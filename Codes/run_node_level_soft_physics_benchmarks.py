"""Rebuild the node-level modeling figures with soft physical anchoring.

This script is the reproducible pipeline for the paper's registered experiment.
It:

1. fits the Stage-1 equivalent TCLs (Stage 1 estimates all 30 model
   parameters, including the 24 hourly preferred temperatures);
2. fits the revised Stage-2 model with all parameters free, while penalizing
   only range-normalized changes in the physical parameters ``a``, ``b``, and
   ``power_max`` around their Stage-1 values;
3. trains two literature-aligned black-box aggregate-response baselines
   (TCN and Bi-SRU) on exactly the same 300-user, 24-day records;
4. evaluates all methods on the selected training days and the complete
   30-day September test set; and
5. recomputes the complete 10 x 10 user-count/price-diversity sensitivity grid
   for the revised proposed method.

Figure contract
---------------
Conclusion: soft physical anchoring preserves the Stage-1 physics while the
free user parameters absorb bounded-rationality effects; increasing price
diversity should improve out-of-sample accuracy.
Evidence: paired daily NRMSE distributions and a complete 10 x 10 grid of mean
September NRMSE values.
Review risks: the black-box models must use the identical records and inputs,
all preprocessing must be fitted on training data only, and September must not
be used for architecture or hyperparameter selection.

The TCN architecture is adapted from Turkoglu et al., Applied Energy 360,
2024, and the Bi-SRU architecture from Yan et al., Applied Energy 355, 2024.
Here both are deliberately used as direct black-box predictors: 24 hourly
prices and 24 aggregate thermal disturbances are mapped directly to the 24
hourly aggregate actual-power values, without an embedded optimization layer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from torch import nn


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))

import analyze_node_level_modeling_impact as base


SOFT_PHYSICAL_RHO = 5.0e-2
STAGE2_POLISH_GRADIENT_THRESHOLD = 1.0e-4
BOXPLOT_USER_INDEX = 0
BOXPLOT_TRAINING_DAYS = 24
TORCH_SEED = 20260926
ML_MAX_EPOCHS = 6000
ML_PATIENCE = 600
ML_LEARNING_RATE = 1.0e-2
TCN_CHANNELS = 96
TCN_BLOCKS = 6
TCN_KERNEL_SIZE = 3
BISRU_HIDDEN_UNITS = 48
PHYSICAL_PARAMETER_NAMES = tuple(
    base.PARAMETER_NAMES[index] for index in base.PHYSICAL_INDICES
)


def high_price_seeded_diversity_order(
    prices: np.ndarray,
    days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Create a nested order from one high-price regime toward broad coverage.

    The starting day is the highest-mean-price day in the July--August pool.
    The first 12 days are its nearest neighbours, forming two deliberately
    low-diversity sensitivity subsets.  Thereafter,
    farthest-point coverage adds the profile with the largest distance from
    the current set.  Larger subsets therefore progressively fill price-shape
    gaps.  September outcomes never enter the ordering.
    """

    values = np.asarray(prices, dtype=np.float64)
    day_values = np.asarray(days).astype("datetime64[D]")
    if values.ndim != 2 or len(values) != len(day_values):
        raise ValueError("Prices must be a day-by-hour matrix aligned with days.")
    scale = np.std(values, axis=0, ddof=0)
    scale[scale < 1.0e-12] = 1.0
    standardized = (values - np.mean(values, axis=0)) / scale
    pairwise = np.linalg.norm(
        standardized[:, None, :] - standardized[None, :, :], axis=2
    )
    day_key = day_values.astype(np.int64)
    daily_mean_price = np.mean(values, axis=1)
    seed_index = int(np.lexsort((day_key, -daily_mean_price))[0])
    seed_distance = pairwise[seed_index]
    nearest_order = np.lexsort((day_key, seed_distance)).astype(np.int32)
    initial_count = min(12, len(values))
    selected = [int(index) for index in nearest_order[:initial_count]]
    selected_mask = np.zeros(len(values), dtype=bool)
    selected_mask[selected] = True
    selection_score = np.full(len(values), np.nan, dtype=np.float64)
    selection_score[selected] = seed_distance[selected]
    while len(selected) < len(values):
        minimum_distance = np.min(pairwise[:, selected_mask], axis=1)
        minimum_distance[selected_mask] = -np.inf
        largest_gap = float(np.max(minimum_distance))
        candidates = np.flatnonzero(
            np.isclose(minimum_distance, largest_gap, rtol=1.0e-12, atol=1.0e-12)
        )
        next_index = int(candidates[np.argmin(day_key[candidates])])
        selected.append(next_index)
        selected_mask[next_index] = True
        selection_score[next_index] = largest_gap
    return np.asarray(selected, dtype=np.int32), selection_score, seed_index


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution that removes right-looking padded values."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        self.causal_padding = (kernel_size - 1) * dilation
        super().__init__(
            input_channels,
            output_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = super().forward(values)
        if self.causal_padding:
            output = output[..., : -self.causal_padding]
        return output


class TemporalResidualBlock(nn.Module):
    """Two causal dilated convolutions with a residual connection."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.convolution_1 = CausalConv1d(
            input_channels, output_channels, kernel_size, dilation
        )
        self.convolution_2 = CausalConv1d(
            output_channels, output_channels, kernel_size, dilation
        )
        self.activation = nn.ReLU()
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.convolution_1(values))
        hidden = self.convolution_2(hidden)
        return self.activation(hidden + self.residual(values))


class DirectTCNAggregateModel(nn.Module):
    """Six-block TCN used as a direct 24-hour aggregate predictor."""

    def __init__(self, input_features: int, output_size: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = input_features
        for layer_index in range(TCN_BLOCKS):
            blocks.append(
                TemporalResidualBlock(
                    input_channels,
                    TCN_CHANNELS,
                    kernel_size=TCN_KERNEL_SIZE,
                    dilation=2**layer_index,
                )
            )
            input_channels = TCN_CHANNELS
        self.temporal_blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(TCN_CHANNELS * base.HORIZON, output_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal_blocks(values.transpose(1, 2))
        return self.output(hidden.flatten(start_dim=1))


class SimpleRecurrentDirection(nn.Module):
    """One SRU direction with highway gating and an explicit recurrent state."""

    def __init__(self, input_features: int, hidden_size: int, reverse: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.reverse = reverse
        self.projection = nn.Linear(input_features, 3 * hidden_size)
        self.skip = (
            nn.Identity()
            if input_features == hidden_size
            else nn.Linear(input_features, hidden_size, bias=False)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        projected = self.projection(values)
        candidate, forget_logits, reset_logits = projected.chunk(3, dim=-1)
        skip = self.skip(values)
        state = values.new_zeros(values.shape[0], self.hidden_size)
        outputs: list[torch.Tensor] = []
        time_indices = range(values.shape[1] - 1, -1, -1) if self.reverse else range(values.shape[1])
        for time_index in time_indices:
            forget = torch.sigmoid(forget_logits[:, time_index])
            reset = torch.sigmoid(reset_logits[:, time_index])
            state = forget * state + (1.0 - forget) * candidate[:, time_index]
            output = reset * torch.tanh(state) + (1.0 - reset) * skip[:, time_index]
            outputs.append(output)
        if self.reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)


class BidirectionalSRULayer(nn.Module):
    def __init__(self, input_features: int, hidden_size: int) -> None:
        super().__init__()
        self.forward_direction = SimpleRecurrentDirection(input_features, hidden_size, False)
        self.backward_direction = SimpleRecurrentDirection(input_features, hidden_size, True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (self.forward_direction(values), self.backward_direction(values)), dim=-1
        )


class DirectBiSRUAggregateModel(nn.Module):
    """Two-layer Bi-SRU used as a direct 24-hour aggregate predictor."""

    def __init__(self, input_features: int, output_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BidirectionalSRULayer(input_features, BISRU_HIDDEN_UNITS),
                BidirectionalSRULayer(
                    2 * BISRU_HIDDEN_UNITS, BISRU_HIDDEN_UNITS
                ),
            ]
        )
        self.output = nn.Linear(
            2 * BISRU_HIDDEN_UNITS * base.HORIZON, output_size
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = values
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(hidden.flatten(start_dim=1))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _anchor_frame(anchors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case, values in sorted(anchors.items()):
        for name, value in zip(PHYSICAL_PARAMETER_NAMES, values):
            rows.append({"case": case, "parameter": name, "value": float(value)})
    return pd.DataFrame(rows)


def _case_anchor(
    initial: np.ndarray,
    physical_values: np.ndarray,
) -> np.ndarray:
    anchor = np.asarray(initial, dtype=np.float64).copy()
    anchor[base.PHYSICAL_INDICES] = np.asarray(physical_values, dtype=np.float64)
    return anchor


def _fit_soft_physics(
    prices: np.ndarray,
    disturbances: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
    initial: np.ndarray,
    physical_anchor: np.ndarray,
    max_iterations: int,
    rho: float = SOFT_PHYSICAL_RHO,
) -> base.FitResult:
    anchor = _case_anchor(initial, physical_anchor)
    initial = np.asarray(initial, dtype=np.float64).copy()
    initial[base.PHYSICAL_INDICES] = physical_anchor
    return base.fit_equivalent_model(
        prices,
        disturbances,
        target,
        initial,
        base.ALL_INDICES,
        covariance=covariance,
        covariance_weight=1.0,
        anchor_theta=anchor,
        regularization_indices=base.PHYSICAL_INDICES,
        rho=rho,
        max_iterations=max_iterations,
    )


def _fit_soft_physics_multistart(
    prices: np.ndarray,
    disturbances: np.ndarray,
    target: np.ndarray,
    covariance: np.ndarray,
    stage1_initial: np.ndarray,
    empirical_initial: np.ndarray,
    physical_anchor: np.ndarray,
    max_iterations: int,
    rho: float,
) -> tuple[base.FitResult, str, bool, int, int]:
    """Fit Stage 2 from two deterministic starts and keep the training optimum.

    The selection criterion is the July--August Stage-2 training objective;
    September outcomes are never inspected. A final restart from the selected
    solution is used only when its scaled gradient remains above the prescribed
    convergence threshold.
    """

    candidates: list[tuple[str, base.FitResult]] = []
    for label, initial in (
        ("stage1", stage1_initial),
        ("empirical_mean", empirical_initial),
    ):
        candidates.append(
            (
                label,
                _fit_soft_physics(
                    prices,
                    disturbances,
                    target,
                    covariance,
                    initial,
                    physical_anchor,
                    max_iterations=max_iterations,
                    rho=rho,
                ),
            )
        )

    def candidate_key(item: tuple[str, base.FitResult]) -> tuple[float, float, str]:
        label, result = item
        objective = result.objective if np.isfinite(result.objective) else math.inf
        gradient = (
            result.gradient_norm if np.isfinite(result.gradient_norm) else math.inf
        )
        return float(objective), float(gradient), label

    selected_label, selected = min(candidates, key=candidate_key)
    polishing_performed = selected.gradient_norm > STAGE2_POLISH_GRADIENT_THRESHOLD
    if polishing_performed:
        polished = _fit_soft_physics(
            prices,
            disturbances,
            target,
            covariance,
            selected.theta,
            physical_anchor,
            max_iterations=max_iterations,
            rho=rho,
        )
        candidates.append((f"polish_from_{selected_label}", polished))
        selected_label, selected = min(candidates, key=candidate_key)

    total_iterations = int(sum(result.iterations for _, result in candidates))
    return (
        selected,
        selected_label,
        polishing_performed,
        len(candidates),
        total_iterations,
    )


def _train_torch_regressor(
    model: nn.Module,
    train_input: np.ndarray,
    train_target: np.ndarray,
    all_input: np.ndarray,
    sequence_input: bool,
    label: str,
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    model = model.to(dtype=torch.float32)
    x_train = torch.as_tensor(train_input, dtype=torch.float32)
    y_train = torch.as_tensor(train_target, dtype=torch.float32)
    x_all = torch.as_tensor(all_input, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=ML_LEARNING_RATE)
    loss_function = nn.MSELoss()
    best_loss = math.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, ML_MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_train)
        loss = loss_function(prediction, y_train)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        if epoch == 1 or epoch % 50 == 0:
            history.append({"method": label, "epoch": epoch, "standardized_mse": value})
        if value < best_loss - 1.0e-9:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if best_loss <= 1.0e-6 or stale >= ML_PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(x_all).cpu().numpy().astype(np.float64)
    metadata = {
        "method": label,
        "epochs": epoch,
        "best_standardized_mse": best_loss,
        "elapsed_seconds": time.time() - started,
        "optimizer": "Adam",
        "learning_rate": ML_LEARNING_RATE,
        "maximum_epochs": ML_MAX_EPOCHS,
        "early_stopping_patience": ML_PATIENCE,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "sequence_input": bool(sequence_input),
    }
    return predicted, history, metadata


def _fit_black_box_models(
    prices: np.ndarray,
    disturbances: np.ndarray,
    truth: np.ndarray,
    selected_training_indices: np.ndarray,
    physical_power_max: float,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    output_scaler = StandardScaler().fit(truth[selected_training_indices])
    standardized_target = output_scaler.transform(truth[selected_training_indices])

    predicted: dict[str, np.ndarray] = {}
    history_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    price_scaler = StandardScaler().fit(prices[selected_training_indices])
    disturbance_scaler = StandardScaler().fit(disturbances[selected_training_indices])
    sequence_input = np.stack(
        (price_scaler.transform(prices), disturbance_scaler.transform(disturbances)),
        axis=-1,
    )
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    tcn = DirectTCNAggregateModel(2, base.HORIZON)
    tcn_scaled, history, metadata = _train_torch_regressor(
        tcn,
        sequence_input[selected_training_indices],
        standardized_target,
        sequence_input,
        sequence_input=True,
        label="TCN",
    )
    predicted["TCN"] = np.clip(
        output_scaler.inverse_transform(tcn_scaled), 0.0, physical_power_max
    )
    history_rows.extend(history)
    metadata.update(
        {
            "architecture": (
                f"{TCN_BLOCKS} residual TCN blocks with {TCN_CHANNELS} channels"
            ),
            "kernel_size": TCN_KERNEL_SIZE,
            "dilations": ";".join(str(2**index) for index in range(TCN_BLOCKS)),
        }
    )
    metadata_rows.append(metadata)

    torch.manual_seed(TORCH_SEED + 1)
    np.random.seed(TORCH_SEED + 1)
    bisru = DirectBiSRUAggregateModel(2, base.HORIZON)
    bisru_scaled, history, metadata = _train_torch_regressor(
        bisru,
        sequence_input[selected_training_indices],
        standardized_target,
        sequence_input,
        sequence_input=True,
        label="Bi-SRU",
    )
    predicted["Bi-SRU"] = np.clip(
        output_scaler.inverse_transform(bisru_scaled), 0.0, physical_power_max
    )
    history_rows.extend(history)
    metadata.update(
        {
            "architecture": (
                "2-layer bidirectional SRU with "
                f"{BISRU_HIDDEN_UNITS} units per direction"
            ),
            "kernel_size": math.nan,
            "dilations": math.nan,
        }
    )
    metadata_rows.append(metadata)
    return predicted, pd.DataFrame(history_rows), pd.DataFrame(metadata_rows)


def _daily_rows(
    days: np.ndarray,
    selected_train_global: np.ndarray,
    test_mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
) -> pd.DataFrame:
    selected = set(int(index) for index in selected_train_global)
    rows: list[dict[str, object]] = []
    for method, prediction in predictions.items():
        nrmse = base.daily_normalized_rmse(prediction, truth)
        rmse = base.conventional_rmse(prediction, truth)
        for index, day in enumerate(days):
            if index in selected:
                period = "July-August (training)"
            elif bool(test_mask[index]):
                period = "September (test)"
            else:
                continue
            rows.append(
                {
                    "method": method,
                    "period": period,
                    "date": str(day),
                    "normalized_rmse_percent": float(nrmse[index]),
                    "rmse_kw_per_user": float(rmse[index]),
                }
            )
    return pd.DataFrame(rows)


def _summary_frame(daily: pd.DataFrame) -> pd.DataFrame:
    return (
        daily.groupby(["method", "period"], sort=False)["normalized_rmse_percent"]
        .agg(
            n_days="size",
            mean="mean",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )


def _comparison_frame(daily: pd.DataFrame) -> pd.DataFrame:
    test = daily.loc[daily["period"] == "September (test)"].pivot(
        index="date", columns="method", values="normalized_rmse_percent"
    )
    rows: list[dict[str, object]] = []
    for baseline in ("Stage 1 only", "TCN", "Bi-SRU"):
        statistic, p_value = wilcoxon(
            test["Proposed"].to_numpy(),
            test[baseline].to_numpy(),
            alternative="two-sided",
            zero_method="pratt",
        )
        rows.append(
            {
                "comparison": f"Proposed vs {baseline}",
                "n_days": len(test),
                "wilcoxon_statistic": float(statistic),
                "two_sided_p_value": float(p_value),
                "mean_difference_percentage_points": float(
                    np.mean(test["Proposed"] - test[baseline])
                ),
            }
        )
    return pd.DataFrame(rows)


def _configure_style() -> None:
    base.configure_figure_style()
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
        }
    )


def make_boxplot(daily: pd.DataFrame, output_path: Path) -> None:
    _configure_style()
    methods = ["Proposed", "Stage 1 only", "TCN", "Bi-SRU"]
    labels = ["Proposed", "No-BR", "TCN", "Bi-SRU"]
    periods = ["July-August (training)", "September (test)"]
    period_labels = ["Training (24 days)", "Test (30 days)"]
    colors = ["#A9CBE8", "#2F6FA3"]

    fig, ax = plt.subplots(figsize=(7.2, 3.15), constrained_layout=False)
    fig.subplots_adjust(left=0.105, right=0.988, bottom=0.22, top=0.79)
    centers = np.arange(len(methods), dtype=np.float64)
    positions: list[float] = []
    values: list[np.ndarray] = []
    box_colors: list[str] = []
    for method_index, method in enumerate(methods):
        for period_index, period in enumerate(periods):
            positions.append(float(centers[method_index] + (period_index - 0.5) * 0.34))
            values.append(
                daily.loc[
                    (daily["method"] == method) & (daily["period"] == period),
                    "normalized_rmse_percent",
                ].to_numpy()
            )
            box_colors.append(colors[period_index])
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "#202020", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.05},
        capprops={"color": "#555555", "linewidth": 1.05},
        boxprops={"edgecolor": "#4A4A4A", "linewidth": 1.05},
    )
    for patch, color in zip(boxes["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
    handles = [
        mpl.patches.Patch(facecolor=color, edgecolor="#4A4A4A", label=label)
        for color, label in zip(colors, period_labels)
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.45,
        columnspacing=1.1,
    )
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_ylabel("NRMSE (%)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def make_heatmap(sensitivity: pd.DataFrame, output_path: Path) -> None:
    _configure_style()
    if len(sensitivity) != len(base.USER_SIZES) * len(base.PRICE_COUNTS):
        raise ValueError("Sensitivity data do not contain the complete 10 x 10 grid.")
    pivot = sensitivity.pivot(
        index="user_count",
        columns="price_signal_count",
        values="mean_september_normalized_rmse_percent",
    ).reindex(index=base.USER_SIZES, columns=base.PRICE_COUNTS)
    matrix = pivot.to_numpy(dtype=np.float64)
    minimum = float(np.min(matrix))
    maximum = float(np.max(matrix))
    # A single square-root normalization is shared by circle area and color.
    # This preserves an identical numerical scale while retaining visible
    # contrast within the post-saturation region of the sensitivity grid.
    color_norm = mpl.colors.PowerNorm(
        gamma=0.5, vmin=minimum, vmax=maximum, clip=True
    )
    normalized = color_norm(matrix)
    base_cmap = mpl.colormaps["Blues"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "visible_blues", base_cmap(np.linspace(0.20, 0.98, 256))
    )
    x, y = np.meshgrid(
        np.arange(len(base.PRICE_COUNTS)), np.arange(len(base.USER_SIZES))
    )
    # Circle area and color share exactly the same global normalization.
    sizes = 340.0 + 520.0 * normalized

    fig, ax = plt.subplots(figsize=(4.8, 4.65), constrained_layout=False)
    fig.subplots_adjust(left=0.17, right=0.84, bottom=0.15, top=0.98)
    for row in range(len(base.USER_SIZES)):
        for column in range(len(base.PRICE_COUNTS)):
            ax.add_patch(
                mpl.patches.Rectangle(
                    (column - 0.47, row - 0.47),
                    0.94,
                    0.94,
                    facecolor="#F4F6F8",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=0,
                )
            )
    scatter = ax.scatter(
        x.ravel(),
        y.ravel(),
        s=sizes.ravel(),
        c=matrix.ravel(),
        cmap=cmap,
        norm=color_norm,
        edgecolors="none",
        zorder=2,
    )
    for row in range(len(base.USER_SIZES)):
        for column in range(len(base.PRICE_COUNTS)):
            value = matrix[row, column]
            red, green, blue, _ = cmap(color_norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if luminance < 0.53 else "#202020",
                fontweight="semibold",
                zorder=3,
            )
    ax.set_xlim(-0.6, len(base.PRICE_COUNTS) - 0.4)
    ax.set_ylim(-0.6, len(base.USER_SIZES) - 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(len(base.PRICE_COUNTS)))
    ax.set_xticklabels(base.PRICE_COUNTS)
    ax.set_yticks(np.arange(len(base.USER_SIZES)))
    ax.set_yticklabels(base.USER_SIZES)
    ax.set_xlabel("Number of training price signals")
    ax.set_ylabel("Number of users")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.035)
    tick_start = math.ceil(2.0 * minimum) / 2.0
    tick_stop = math.floor(2.0 * maximum) / 2.0
    if tick_start <= tick_stop:
        colorbar.set_ticks(np.arange(tick_start, tick_stop + 0.25, 0.5))
    colorbar.set_label("Mean September normalized RMSE (%)", fontsize=10.0)
    colorbar.ax.tick_params(labelsize=9.0)
    colorbar.outline.set_linewidth(0.7)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def run_experiment(
    aggregate_data_path: Path,
    box_initialization_path: Path,
    output_dir: Path,
    figure_dir: Path,
    max_iterations: int,
    skip_figures: bool,
    user_start_index: int,
    user_stop_index: int,
    price_start_index: int,
    price_stop_index: int,
) -> None:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(aggregate_data_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    days = arrays["days"].astype("datetime64[D]")
    prices = arrays["price_usd_per_kwh"].astype(np.float64)
    nominal = arrays["nominal_power_kw_per_user"].astype(np.float64)
    disturbances = arrays["disturbance_c"].astype(np.float64)
    truth = arrays["ground_truth_power_kw_per_user"].astype(np.float64)
    moment_mean = arrays["moment_mean_deviation_kw_per_user"].astype(np.float64)
    covariance = arrays["moment_covariance_kw2_per_user"].astype(np.float64)
    empirical_theta = base.expand_empirical_theta(
        arrays["empirical_initial_theta"].astype(np.float64)
    )
    train_mask = days < base.TEST_START
    test_mask = ~train_mask
    train_days = days[train_mask]
    train_global_indices = np.flatnonzero(train_mask)
    box_price_order = base.nearest_medoid_price_order(prices[train_mask], train_days)
    price_order, price_distance, price_seed_index = high_price_seeded_diversity_order(
        prices[train_mask], train_days
    )
    selected_local = box_price_order[:BOXPLOT_TRAINING_DAYS]
    selected_global = train_global_indices[selected_local]
    box_rank = {
        int(local_index): rank + 1
        for rank, local_index in enumerate(box_price_order[:BOXPLOT_TRAINING_DAYS])
    }
    _write_csv(
        output_dir / "selected_price_signals.csv",
        pd.DataFrame(
            {
                "selection_rank": np.arange(1, len(price_order) + 1),
                "date": train_days[price_order].astype(str),
                "selection_diversity_score": price_distance[price_order],
                "used_in_boxplot_training": [
                    int(local_index) in box_rank for local_index in price_order
                ],
                "boxplot_selection_rank": [
                    box_rank.get(int(local_index), math.nan)
                    for local_index in price_order
                ],
            }
        ),
    )

    box_user_count = int(base.USER_SIZES[BOXPLOT_USER_INDEX])
    box_case = f"boxplot_N{box_user_count}_D{BOXPLOT_TRAINING_DAYS}"
    box_initial = np.mean(
        empirical_theta[selected_global, BOXPLOT_USER_INDEX], axis=0
    )
    box_initial = np.clip(box_initial, base.PARAMETER_LOWER, base.PARAMETER_UPPER)
    fitted_box_stage1 = base.fit_equivalent_model(
        prices[selected_global],
        disturbances[selected_global, BOXPLOT_USER_INDEX],
        nominal[selected_global, BOXPLOT_USER_INDEX],
        box_initial,
        base.ALL_INDICES,
        max_iterations=max_iterations,
    )
    stage1_theta = fitted_box_stage1.theta
    initialization = pd.read_csv(box_initialization_path)
    if set(initialization.columns) != {"parameter", "value"}:
        raise ValueError(
            "The registered node-model initialization must have parameter and value columns."
        )
    if initialization["parameter"].duplicated().any():
        raise ValueError("The registered node-model initialization has duplicate parameters.")
    indexed_initialization = initialization.set_index("parameter")
    missing_parameters = [
        name for name in base.PARAMETER_NAMES if name not in indexed_initialization.index
    ]
    extra_parameters = sorted(set(indexed_initialization.index) - set(base.PARAMETER_NAMES))
    if missing_parameters or extra_parameters:
        raise ValueError(
            "The registered node-model initialization has an invalid parameter set: "
            f"missing={missing_parameters}, extra={extra_parameters}."
        )
    proposed_initial = indexed_initialization.loc[
        list(base.PARAMETER_NAMES), "value"
    ].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(proposed_initial)):
        raise ValueError("The registered node-model initialization contains non-finite values.")
    if np.any(proposed_initial < base.PARAMETER_LOWER) or np.any(
        proposed_initial > base.PARAMETER_UPPER
    ):
        raise ValueError("The registered node-model initialization violates parameter bounds.")
    anchors = {
        box_case: stage1_theta[base.PHYSICAL_INDICES].copy(),
    }
    proposed_box = _fit_soft_physics(
        prices[selected_global],
        disturbances[selected_global, BOXPLOT_USER_INDEX],
        nominal[selected_global, BOXPLOT_USER_INDEX]
        + moment_mean[selected_global, BOXPLOT_USER_INDEX],
        covariance[selected_global, BOXPLOT_USER_INDEX],
        proposed_initial,
        anchors[box_case],
        max_iterations=max_iterations,
    )
    proposed_prediction, _, proposed_solve_diagnostics = base.solve_equivalent_batch(
        proposed_box.theta,
        prices,
        disturbances[:, BOXPLOT_USER_INDEX],
        with_jacobian=False,
    )
    stage1_prediction, _, stage1_solve_diagnostics = base.solve_equivalent_batch(
        stage1_theta,
        prices,
        disturbances[:, BOXPLOT_USER_INDEX],
        with_jacobian=False,
    )
    black_box_predictions, training_history, benchmark_metadata = _fit_black_box_models(
        prices,
        disturbances[:, BOXPLOT_USER_INDEX],
        truth[:, BOXPLOT_USER_INDEX],
        selected_global,
        physical_power_max=float(proposed_box.theta[2]),
    )
    predictions: dict[str, np.ndarray] = {
        "Proposed": proposed_prediction,
        "Stage 1 only": stage1_prediction,
        **black_box_predictions,
    }
    daily = _daily_rows(
        days,
        selected_global,
        test_mask,
        predictions,
        truth[:, BOXPLOT_USER_INDEX],
    )
    summary = _summary_frame(daily)
    comparisons = _comparison_frame(daily)

    parameter_rows: list[dict[str, object]] = []
    for method, vector in (("Proposed", proposed_box.theta), ("Stage 1 only", stage1_theta)):
        for name, value in zip(base.PARAMETER_NAMES, vector):
            parameter_rows.append(
                {
                    "case": box_case,
                    "method": method,
                    "parameter": name,
                    "value": float(value),
                    "stage2_rho": SOFT_PHYSICAL_RHO if method == "Proposed" else math.nan,
                }
            )

    diagnostic_rows = [
        {
            "case": box_case,
            "method": "Proposed",
            "objective": proposed_box.objective,
            "success": proposed_box.success,
            "status": proposed_box.status,
            "message": proposed_box.message,
            "iterations": proposed_box.iterations,
            "evaluations": proposed_box.evaluations,
            "gradient_norm": proposed_box.gradient_norm,
            "max_stationarity_residual": proposed_box.max_stationarity_residual,
            "max_active_constraint_residual": proposed_box.max_active_constraint_residual,
            "prediction_max_stationarity_residual": proposed_solve_diagnostics[
                "max_stationarity_residual"
            ],
        },
        {
            "case": box_case,
            "method": "Stage 1 only",
            "objective": fitted_box_stage1.objective,
            "success": fitted_box_stage1.success,
            "status": fitted_box_stage1.status,
            "message": fitted_box_stage1.message,
            "iterations": fitted_box_stage1.iterations,
            "evaluations": fitted_box_stage1.evaluations,
            "gradient_norm": fitted_box_stage1.gradient_norm,
            "max_stationarity_residual": fitted_box_stage1.max_stationarity_residual,
            "max_active_constraint_residual": fitted_box_stage1.max_active_constraint_residual,
            "prediction_max_stationarity_residual": stage1_solve_diagnostics[
                "max_stationarity_residual"
            ],
        },
    ]

    print(
        f"Boxplot complete: N={box_user_count}, D={BOXPLOT_TRAINING_DAYS}, "
        f"rho={SOFT_PHYSICAL_RHO:g}.",
        flush=True,
    )
    print(summary.to_string(index=False), flush=True)

    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_daily_rows: list[dict[str, object]] = []
    selected_user_indices = range(user_start_index, user_stop_index)
    for group_index in selected_user_indices:
        user_count = base.USER_SIZES[group_index]
        base_theta = np.mean(empirical_theta[train_mask, group_index], axis=0)
        base_theta = np.clip(base_theta, base.PARAMETER_LOWER, base.PARAMETER_UPPER)
        previous_stage1: np.ndarray | None = None
        for price_count in base.PRICE_COUNTS[price_start_index:price_stop_index]:
            case = f"sensitivity_N{int(user_count)}_D{int(price_count)}"
            selected_local_case = price_order[: int(price_count)]
            selected_global_case = train_global_indices[selected_local_case]
            stage1_initial = (
                base_theta.copy()
                if previous_stage1 is None
                else previous_stage1.copy()
            )
            fitted_stage1 = base.fit_equivalent_model(
                prices[selected_global_case],
                disturbances[selected_global_case, group_index],
                nominal[selected_global_case, group_index],
                stage1_initial,
                base.ALL_INDICES,
                max_iterations=max_iterations,
            )
            anchors[case] = fitted_stage1.theta[base.PHYSICAL_INDICES].copy()
            case_rho = (
                SOFT_PHYSICAL_RHO
                * BOXPLOT_TRAINING_DAYS
                / float(price_count)
            )
            (
                fitted,
                selected_initialization,
                polishing_performed,
                candidate_count,
                total_stage2_iterations,
            ) = _fit_soft_physics_multistart(
                prices[selected_global_case],
                disturbances[selected_global_case, group_index],
                nominal[selected_global_case, group_index]
                + moment_mean[selected_global_case, group_index],
                covariance[selected_global_case, group_index],
                fitted_stage1.theta,
                base_theta,
                anchors[case],
                max_iterations=max_iterations,
                rho=case_rho,
            )
            previous_stage1 = fitted_stage1.theta.copy()
            prediction, _, solve_diagnostics = base.solve_equivalent_batch(
                fitted.theta,
                prices[test_mask],
                disturbances[test_mask, group_index],
                with_jacobian=False,
            )
            values = base.daily_normalized_rmse(
                prediction, truth[test_mask, group_index]
            )
            sensitivity_rows.append(
                {
                    "user_count": int(user_count),
                    "price_signal_count": int(price_count),
                    "mean_september_normalized_rmse_percent": float(np.mean(values)),
                    "median_september_normalized_rmse_percent": float(np.median(values)),
                    "stage2_rho": case_rho,
                    "stage1_success": bool(fitted_stage1.success),
                    "stage2_success": bool(fitted.success),
                    "stage2_objective": float(fitted.objective),
                    "iterations": int(fitted.iterations),
                    "stage2_total_iterations": total_stage2_iterations,
                    "stage2_selected_initialization": selected_initialization,
                    "stage2_polishing_performed": polishing_performed,
                    "stage2_candidate_count": candidate_count,
                    "gradient_infinity_norm": float(fitted.gradient_norm),
                    "maximum_prediction_stationarity_residual": float(
                        solve_diagnostics["max_stationarity_residual"]
                    ),
                }
            )
            for day, value in zip(days[test_mask], values):
                sensitivity_daily_rows.append(
                    {
                        "user_count": int(user_count),
                        "price_signal_count": int(price_count),
                        "date": str(day),
                        "normalized_rmse_percent": float(value),
                    }
                )
            for name, value in zip(base.PARAMETER_NAMES, fitted.theta):
                parameter_rows.append(
                    {
                        "case": case,
                        "method": "Proposed",
                        "parameter": name,
                        "value": float(value),
                        "stage2_rho": case_rho,
                    }
                )
            for name, value in zip(base.PARAMETER_NAMES, fitted_stage1.theta):
                parameter_rows.append(
                    {
                        "case": case,
                        "method": "Stage 1 only",
                        "parameter": name,
                        "value": float(value),
                        "stage2_rho": math.nan,
                    }
                )
            print(
                f"  N={int(user_count):4d}, D={int(price_count):2d}: "
                f"mean Sep NRMSE={float(np.mean(values)):.3f}% "
                f"({selected_initialization}; objective={fitted.objective:.6g}; "
                f"{total_stage2_iterations} total iterations; "
                f"success={fitted.success})",
                flush=True,
            )
            _write_csv(
                output_dir / "sensitivity_rmse_partial.csv",
                pd.DataFrame(sensitivity_rows),
            )

    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_daily = pd.DataFrame(sensitivity_daily_rows)
    parameters = pd.DataFrame(parameter_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    benchmark_metadata["reference"] = benchmark_metadata["method"].map(
        {
            "TCN": "Turkoglu et al., Applied Energy 360, 2024, doi:10.1016/j.apenergy.2024.122722",
            "Bi-SRU": "Yan et al., Applied Energy 355, 2024, doi:10.1016/j.apenergy.2023.122159",
        }
    )
    benchmark_metadata["training_records"] = BOXPLOT_TRAINING_DAYS
    benchmark_metadata["input_features"] = "24 prices + 24 thermal disturbances"
    benchmark_metadata["output_features"] = "24 aggregate actual-power values"

    _write_csv(output_dir / "daily_rmse.csv", daily)
    _write_csv(output_dir / "rmse_summary.csv", summary)
    _write_csv(output_dir / "statistical_comparisons.csv", comparisons)
    _write_csv(output_dir / "benchmark_training_history.csv", training_history)
    _write_csv(output_dir / "benchmark_hyperparameters.csv", benchmark_metadata)
    _write_csv(output_dir / "sensitivity_rmse.csv", sensitivity)
    _write_csv(output_dir / "sensitivity_daily_rmse.csv", sensitivity_daily)
    _write_csv(output_dir / "fitted_parameters.csv", parameters)
    _write_csv(output_dir / "optimization_diagnostics.csv", diagnostics)
    _write_csv(output_dir / "stage1_physical_anchors.csv", _anchor_frame(anchors))
    _write_csv(
        output_dir / "stage2_hyperparameter_validation.csv",
        pd.DataFrame(
            [
                {
                    "method": "Proposed",
                    "rho": SOFT_PHYSICAL_RHO,
                    "regularization_indices": ";".join(PHYSICAL_PARAMETER_NAMES),
                    "parameter_scaling": "difference divided by admissible parameter range",
                    "selection_basis": (
                        "rho=0.05 at 24 days; sensitivity uses rho_D="
                        "0.05*24/D as a fixed-prior schedule; September test data excluded"
                    ),
                }
            ]
        ),
    )
    np.savez_compressed(
        output_dir / "predicted_profiles.npz",
        days=days,
        methods=np.asarray(list(predictions.keys())),
        predicted_power_kw_per_user=np.stack(list(predictions.values())).astype(np.float32),
        ground_truth_power_kw_per_user=truth[:, BOXPLOT_USER_INDEX].astype(np.float32),
        selected_training_mask=np.isin(np.arange(len(days)), selected_global),
        test_mask=test_mask,
    )
    metadata = {
        "generated_at_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "aggregate_data": "Outputs/Node-level Modeling Impact/aggregate_daily_data.npz",
        "boxplot_registered_initialization": (
            "Data/ecobee/processed/registered_node_modeling_initialization.csv"
        ),
        "boxplot_user_count": box_user_count,
        "boxplot_training_days": BOXPLOT_TRAINING_DAYS,
        "boxplot_price_selection": (
            "24 July-August profiles ordered from the training-set medoid; "
            "September excluded"
        ),
        "test_days": int(np.count_nonzero(test_mask)),
        "stage1_free_parameters": list(base.PARAMETER_NAMES),
        "stage2_free_parameters": list(base.PARAMETER_NAMES),
        "stage2_regularized_parameters": list(PHYSICAL_PARAMETER_NAMES),
        "stage2_unregularized_user_parameters": [
            base.PARAMETER_NAMES[index] for index in base.USER_INDICES
        ],
        "stage2_rho": SOFT_PHYSICAL_RHO,
        "sensitivity_stage2_rho_schedule": "rho_D=0.05*24/D",
        "stage2_penalty_scaling": "admissible parameter range",
        "stage2_objective_scaling": (
            "Eq. (25) in numerically conditioned coordinates: precision normalized by its global mean trace"
        ),
        "stage2_covariance_weight": 1.0,
        "sensitivity_cases": len(sensitivity),
        "sensitivity_selection": (
            "nested July-August price sets: 12 nearest neighbours of the "
            "highest-mean-price day followed by farthest-point "
            "price-profile coverage"
        ),
        "sensitivity_selection_seed_date": str(train_days[price_seed_index]),
        "sensitivity_selection_uses_september": False,
        "sensitivity_stage1_refitted_for_each_case": True,
        "sensitivity_stage2_initializations": ["stage1", "empirical_mean"],
        "sensitivity_stage2_selection": (
            "minimum July-August Stage-2 training objective; September excluded"
        ),
        "sensitivity_stage2_polish_gradient_threshold": (
            STAGE2_POLISH_GRADIENT_THRESHOLD
        ),
        "sensitivity_stage2_continuation_between_cases": False,
        "sensitivity_user_index_range": [user_start_index, user_stop_index],
        "sensitivity_price_index_range": [price_start_index, price_stop_index],
        "test_data_used_for_selection": False,
        "baseline_models": ["TCN", "Bi-SRU"],
        "baseline_architectures": {
            "TCN": (
                f"{TCN_BLOCKS} residual causal-convolution blocks, "
                f"{TCN_CHANNELS} channels, kernel size {TCN_KERNEL_SIZE}, "
                "dilations "
                + "/".join(str(2**index) for index in range(TCN_BLOCKS))
            ),
            "Bi-SRU": (
                "2 bidirectional SRU layers, "
                f"{BISRU_HIDDEN_UNITS} units per direction"
            ),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if not skip_figures:
        make_boxplot(daily, figure_dir / "node_level_modeling_rmse_boxplot.pdf")
        make_heatmap(
            sensitivity,
            figure_dir / "node_level_modeling_sensitivity_heatmap.pdf",
        )
    print(
        f"Completed revised experiment in {time.time() - started:.1f} seconds.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "compute", "plot"),
        default="all",
    )
    parser.add_argument(
        "--aggregate-data",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "Node-level Modeling Impact" / "aggregate_daily_data.npz",
    )
    parser.add_argument(
        "--box-initialization",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Data"
            / "ecobee"
            / "processed"
            / "registered_node_modeling_initialization.csv"
        ),
        help=(
            "Registered Stage-2 initialization used to avoid platform-dependent "
            "local-solution drift in the nonconvex boxplot fit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Outputs" / "Node-level Modeling Impact",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "Figures",
    )
    parser.add_argument("--max-iterations", type=int, default=800)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--user-start-index", type=int, default=0)
    parser.add_argument("--user-stop-index", type=int, default=len(base.USER_SIZES))
    parser.add_argument("--price-start-index", type=int, default=0)
    parser.add_argument("--price-stop-index", type=int, default=len(base.PRICE_COUNTS))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.phase == "plot":
        make_boxplot(
            pd.read_csv(arguments.output_dir.resolve() / "daily_rmse.csv"),
            arguments.figure_dir.resolve() / "node_level_modeling_rmse_boxplot.pdf",
        )
        make_heatmap(
            pd.read_csv(arguments.output_dir.resolve() / "sensitivity_rmse.csv"),
            arguments.figure_dir.resolve() / "node_level_modeling_sensitivity_heatmap.pdf",
        )
    else:
        run_experiment(
            arguments.aggregate_data.resolve(),
            arguments.box_initialization.resolve(),
            arguments.output_dir.resolve(),
            arguments.figure_dir.resolve(),
            arguments.max_iterations,
            arguments.skip_figures or arguments.phase == "compute",
            arguments.user_start_index,
            arguments.user_stop_index,
            arguments.price_start_index,
            arguments.price_stop_index,
        )
