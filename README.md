Policy Optimization for Financial Decision-Making

Author: Ankit Aggarwal
Date: 30/10/2025

Project summary:
Compare a supervised deep learning (MLP) classifier and an offline RL agent (CQL) on LendingClub loans to optimize business value (loan approvals → profit/loss). Reproducible code and notebooks are in this repository.

Repository structure (top-level):
/SHODH... (project root)
├─ .venv/                     # (optional) local virtualenv used during development (not checked in)
├─ artifacts/                 # saved models, preprocessor and other output artifacts
│  ├─ dl_model.keras
│  ├─ discrete_cql_model.d3
│  └─ preprocessed_data.joblib
├─ d3rlpy_logs/               # logs created by d3rlpy during RL runs
├─ accepted_2007_to_2018.csv  # LendingClub source CSV (large - not included)
├─ Data_Sampling.ipynb
├─ EDA_PreProcessing.ipynb
├─ model_dl_classifier.ipynb
├─ model_RL_Agent.ipynb
├─ analysis_comparison.ipynb
├─ Final_report.md (or Final_report.pdf)
└─ README.md


Requirements & recommended environment:
-Python: 3.10 (recommended). Some packages (tensorflow, mediapipe in other projects) work best on 3.10/3.11.
-CPU is sufficient for reproducing evaluation and small training runs; GPU recommended for faster DL/RL training (tensorflow GPU & CUDA drivers must be configured separately).

Create a fresh virtual environment:
# unix / macOS
python3.10 -m venv .venv
source .venv/bin/activate

# windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1


These versions were used during development; slight version differences may work but may also change API behavior (especially d3rlpy).
# Core
numpy>=1.24
pandas>=2.0
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.12

# Deep learning
tensorflow==2.12.0    # match your GPU/CUDA if using GPU

# Offline RL
d3rlpy==2.8.1         # current dev version used in experiments

# optional: if you want exact legacy EPV API, use d3rlpy==1.1.1 in a separate env

# Utilities
joblib
jupyterlab
notebook
ipykernel
tqdm


Note about d3rlpy and EPV
Development used d3rlpy==2.8.1. Newer 2.x releases removed some legacy OPE helper APIs; the notebook contains code that computes EPV via the critic (predict_value) and also documents how to switch to the older API if desired.
If you prefer the original OffPolicyEvaluator API (legacy examples), create a separate environment and install d3rlpy==1.1.1:
--pip install "d3rlpy==1.1.1"
Then you can use evaluate_on_environment() and OffPolicyEvaluator

How to run everything (recommended order)
Important: place the LendingClub CSV accepted_2007_to_2018.csv in the repository root (or edit the path in the Data_Sampling notebook).

1. Start Jupyter
   jupyter lab
2. (Optional) Kernel setup
   If you created .venv above, create an ipykernel so notebooks use the same env:
   python -m ipykernel install --user --name=shodh-env --display-name "shodh-env"
   Then select shodh-env kernel in Jupyter.
3. Run notebooks in this exact order

   1) Data_Sampling.ipynb
      Purpose: read raw accepted_2007_to_2018.csv, sample/filter to Fully Paid and Charged Off, save data_sample.csv and preprocessed_data.joblib (or a data snapshot).
      Output: data_sample.csv (smaller), artifacts/preprocessed_data.joblib (used later).

   2) EDA_PreProcessing.ipynb
      Purpose: EDA, missingness report, feature selection, encode & scale features, save fitted preprocessor pipeline to artifacts/preprocessor.joblib.
      Action: run all cells to produce exploratory figures and final preprocessed arrays.

   3) model_dl_classifier.ipynb
      Purpose: train the supervised MLP, save artifacts/dl_model.keras. Evaluates test AUC and F1, produces ROC curve and threshold sweep.
      Action: run training cell(s) or, if artifacts exist, skip training and load model from artifacts/dl_model.keras.

   4) model_RL_Agent.ipynb
      Purpose: construct RL dataset (MDPDataset), train CQL agent using d3rlpy, save artifacts/discrete_cql_model.d3. Contains training hyperparameters and logs.
      Action: training may be time-consuming; you can load pre-trained model from artifacts/discrete_cql_model.d3.

   5) analysis_comparison.ipynb
      Purpose: produce DL vs RL policy comparison, find disagreement cases, compute EPV (see below), generate summary tables and figures. This notebook uses artifacts/preprocessed_data.joblib, artifacts/dl_model.keras, and artifacts/discrete_cql_model.d3.

   6) Final_report.md or Final_report.pdf
      Generate PDF from the markdown or include the provided PDF.

Tip: If training is expensive, the notebooks include cells to load saved artifacts instead of retraining. This reproduces the evaluation only.


Reproducible metrics & EPV (how we computed Estimated Policy Value)
We provide two ways to compute EPV — both are included in analysis_comparison.ipynb:

A. Preferred (no downgrade required) — d3rlpy 2.x approach (critic-based EPV)
This method uses the RL critic to estimate the value function for each test state and reports the mean predicted value (plus bootstrap CI). This is what we used when OffPolicyEvaluator was unavailable.
# snippet (already present in analysis_comparison.ipynb)
import numpy as np
from d3rlpy.dataset import MDPDataset
import pandas as pd

# Load data and models (use your variable names)
data = joblib.load('artifacts/preprocessed_data.joblib')   # contains X_test, y_test, test_data_rl
X_test = data['X_test']
test_data_rl = data['test_data_rl']

# Build MDPDataset if needed (fallback uses X_test and dummy rewards)
if not isinstance(test_data_rl, MDPDataset):
    observations = X_test
    actions = rl_policy_action
    rewards = np.zeros(len(actions))
    terminals = np.zeros(len(actions)); terminals[-1] = 1
    test_mdp_dataset = MDPDataset(observations, actions, rewards, terminals)
else:
    test_mdp_dataset = test_data_rl

obs = np.array(test_mdp_dataset.observations)
values = cql.predict_value(obs).squeeze()
epv_mean = float(np.mean(values))

# Bootstrap CI
n_boot = 100
rng = np.random.default_rng(42)
boots = [np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
ci_lower, ci_upper = np.percentile(boots, [2.5, 97.5])

print(f"EPV (critic mean): {epv_mean:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

Notes
This method gives an estimate of the expected return per loan based on the learned Q-function. We explicitly document this method in the report (and contrast it with FQE when applicable).
We recommend reporting the bootstrap CI to show uncertainty.

B. Legacy / exact Off-Policy Evaluator (if you want the original API)
   If you prefer the older evaluator APIs (e.g., OffPolicyEvaluator / evaluate_on_environment), use a separate environment and install: pip install "d3rlpy==1.1.1"

   Then use the legacy evaluation call:
   from d3rlpy.metrics.scorer import evaluate_on_environment
   epv = evaluate_on_environment(cql, test_mdp_dataset)
   print("EPV (legacy evaluator):", epv)
   We provide both options in the notebooks. If using the legacy package, note that other APIs may differ.


Reproducing the report figures & tables
   ROC curve for the DL model: generated in model_dl_classifier.ipynb using roc_curve and matplotlib.
   Threshold sweep (best F1) and confusion matrix: included in model_dl_classifier.ipynb.
   RL training logs: visible in d3rlpy_logs/ and printed during training in model_RL_Agent.ipynb.
   Disagreement statistics and example rows: produced in analysis_comparison.ipynb (look for the cell titled "Find and Analyze Disagreements").
   EPV and bootstrap CI: analysis_comparison.ipynb has a cell named "RL EPV estimation".

Reproducibility notes & tips
   Random seeds: notebooks set random_state=42 for reproducibility where applicable. There are still stochastic aspects (GPU nondeterminism, parallelism). Expect small numeric differences on repeated runs.
   Large dataset: accepted_2007_to_2018.csv is large (multiple GB). Use Data_Sampling.ipynb to produce a smaller working sample for development. Final experiments used a filtered dataset of 149,998 loans.
   Artifacts: the artifacts/ folder contains model files and preprocessed_data.joblib so you can skip long training and directly reproduce evaluation results.
   EPV methodology: because d3rlpy’s OPE API changed across versions, the notebooks include both the critic-based EPV (works with 2.x) and legacy OffPolicyEvaluator calls (works with 1.1.1). The report documents exactly which method produced the reported EPV number.   
   
