# Aggregation-bounded-rationality

_This work develops a flexibility-aggregation framework that incorporates the probabilistic bounded rationality of residential air-conditioning users. It derives a Gaussian approximation of aggregate power deviations, identifies bounded-rationality-aware node-level equivalent models, propagates the deviations through a distribution network, and evaluates their impact on day-ahead market clearing._

Codes for the paper "Bounded Rationality Integrated Flexibility Aggregation of Residential Air Conditioning Loads."

Authors: Xueyuan Cui, Yi Wang, and Audun Botterud.

## Requirements

The registered experiments use Python 3.11 and Gurobi 12.0. A working Gurobi license is required for the optimization-based node-level, network-level, and market-clearing experiments.

Create an isolated environment and install the required packages by running

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scripts are CPU-compatible. The full 3,000-user workflow contains many optimization and Monte Carlo runs and can take several hours.

## Experiments

### Data

The data are not stored in this repository. Download the complete `Data` folder from [Google Drive](https://drive.google.com/drive/folders/1N-7cOkxTYllnyyKiIpaO1Bsj4eFLDSJY?usp=sharing) and place it in the repository root.

An optional cached directory, `Outputs/Bounded Rationality Jul-Sep 3000`, is distributed in the same Google Drive folder. If downloaded, place it under the repository-level `Outputs/` directory and run `python Codes/run_all.py --resume`. The runner will reuse its registered individual optimization and user-distribution files.

The resulting layout must contain at least

```text
Data/
|-- Inputs_33kV_25MW_3000_JulSep/
|   |-- metadata.json
|   |-- network_33kv.csv
|   |-- node_assignments.csv
|   |-- node_01.npy ... node_32.npy
|   `-- nyiso_capitl_dam_lbmp.csv
|-- NYISO Price/
|   `-- RTLBMP_202509/
|       `-- 30 daily real-time price CSV files
`-- ecobee/
    `-- processed/
        |-- template_conditional_pmf.npz
        `-- registered_node_modeling_initialization.csv
```

`Inputs_33kV_25MW_3000_JulSep` is the registered 92-day, 3,000-user cooling-only case used by the paper. The small processed ecobee PMF is the fixed empirical input used to reproduce the bounded-rationality distribution. The registered node-model initialization prevents local-solution drift in the nonconvex Stage-2 calibration across solver platforms; the code still reruns and audits the calibration. The raw ecobee files and building-prototype data in the shared `Data` folder are retained for provenance; the registered figure pipeline starts from the prepared case, PMF, and initialization above.

### Reproduction

To validate the downloaded inputs and reproduce every case-study figure in the order used by the paper, run

```bash
python Codes/run_all.py
```

Intermediate numerical results are written to `Outputs/`, and the publication figures are written to `Figures/`. Both paths are constructed relative to the repository root; no machine-specific path is required.

The workflow is checkpoint-friendly. To inspect the ordered steps or resume without rerunning completed steps, use

```bash
python Codes/run_all.py --list-steps
python Codes/run_all.py --resume
python Codes/run_all.py --from-step node-density --to-step fit-node-models
```

The exact manual running order is

```bash
python Codes/plot_ieee33_network_julsep_3000.py
python Codes/plot_data_preparation_profiles_julsep_3000.py
python Codes/optimize_cooling_only_julsep_3000.py
python Codes/prepare_bounded_rationality_julsep_3000.py
python Codes/analyze_node_level_distribution_accuracy.py --analysis-users 300 --density-only --output-dir "Outputs/Node-level Aggregate Distribution Direct Dynamics Density"
python Codes/analyze_node_level_distribution_accuracy.py --analysis-users 3000 --wasserstein-only --output-dir "Outputs/Node-level Aggregate Distribution Direct Dynamics Sensitivity"
python Codes/analyze_node_level_modeling_impact.py --phase prepare
python Codes/run_node_level_soft_physics_benchmarks.py
python Codes/analyze_network_level_flexibility.py --scenario market
python Codes/analyze_network_level_flexibility.py --scenario relaxed
python Codes/analyze_market_clearing_result1.py
python Codes/finalize_market_clearing_result1.py
```

All registered random seeds, dates, sample sizes, network limits, and training/test splits are defined in the corresponding scripts.

## File description

`Codes/run_all.py`: validates the downloaded data and runs the complete registered workflow in dependency order.

`Codes/case_config.py`: stores the shared cooling-only case constants and the daily reachability audit.

`Codes/plot_ieee33_network_julsep_3000.py`: draws the modified IEEE 33-bus network with 26 user buses and six PV-only buses.

`Codes/plot_data_preparation_profiles_julsep_3000.py`: selects a representative day and draws the PV, non-flexible-load, and equivalent thermal-disturbance profiles.

`Codes/optimize_cooling_only_julsep_3000.py`: solves the individual 24-hour cooling-only optimal-response problems for all 3,000 users over July--September 2025.

`Codes/prepare_bounded_rationality_julsep_3000.py`: validates the empirical ecobee conditional PMF, expands it reproducibly to all users, and draws the conditional-distribution figure.

`Codes/analyze_node_level_distribution_accuracy.py`: compares the proposed aggregate Gaussian distribution with independent Monte Carlo ground truth and evaluates the 300-to-3,000-user Wasserstein sensitivity.

`Codes/analyze_node_level_modeling_impact.py`: prepares common aggregate daily records and implements the Stage-1/Stage-2 equivalent-TCL identification routines used by later experiments.

`Codes/run_node_level_soft_physics_benchmarks.py`: fits the proposed model, the No-BR ablation, TCN, and Bi-SRU baselines; it draws the NRMSE boxplot and the 10-by-10 sensitivity heatmap.

`Codes/analyze_network_level_flexibility.py`: fits node-equivalent models and compares analytical and Monte Carlo root-node flexibility boundaries under 7.5-MW and 15-MW branch limits.

`Codes/analyze_market_clearing_result1.py`: runs the 30 September day-ahead market-clearing cases for Ground-truth, Proposed, and No-BR methods.

`Codes/finalize_market_clearing_result1.py`: applies the registered security-recourse valuation, refreshes the market-clearing tables, and draws the daily-cost figure.

## License

This project is released under the MIT License.
