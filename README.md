# RetainAI

Predict telecom customer churn, compare two models, and turn the predictions
into a ranked contact list with explicit campaign assumptions.

The project uses four notebooks and a small set of shared helpers. It runs
locally with the included public dataset; no account, API key, or cloud service
is required. Read the [problem statement](PROBLEM_STATEMENT.md) for the business
question, intended users, and scope.

## Run it

The reference environment is **Python 3.14.4**. Use Python 3.14 to match the
tested setup. Direct library versions are pinned in `requirements.txt`;
transitive dependencies are resolved by pip rather than fully locked.

Clone the repository, then run these commands one line at a time.

**Windows / PowerShell**

```powershell
git clone https://github.com/DazedSaturn07/RetainAi.git
cd RetainAi
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m jupyter lab
```

If `python` selects another version, use `py -3.14 -m venv .venv` when creating
the environment. These commands do not require PowerShell script activation.

**macOS / Linux**

```bash
git clone https://github.com/DazedSaturn07/RetainAi.git
cd RetainAi
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m jupyter lab
```

Installation needs internet access. The analysis runs offline afterward. The
commands above assume Git and Python are installed. Local verification used
Windows; the GitHub Actions workflow also checks the project on Ubuntu.

In JupyterLab, open each notebook in the order below. Select the environment's
Python kernel and run cells from top to bottom with **Shift+Enter**, or use
**Kernel → Restart Kernel and Run All Cells**. Finish a notebook before moving
to the next. If a cell fails, resolve it before continuing.

| Notebook | What happens | Files written under `output/` |
| --- | --- | --- |
| `01_data.ipynb` | Verify the source, inspect records, validate inputs, split by profile | Three split CSVs, data report, sample batch |
| `02_model.ipynb` | Train two candidates, select using validation, evaluate on test | `model.joblib`, `metrics.json` |
| `03_scoring.ipynb` | Reload the model and rank an example batch | `scores.csv` |
| `04_drift.ipynb` | Compare batch distributions with training data | `drift.json` |

Each notebook can start in a fresh kernel. They share files rather than
in-memory variables. If you rerun preparation, rerun all later stages. If you
retrain, rerun scoring and drift. Earlier stages remove known downstream
results to avoid presenting an older model's outputs as current.

To run without opening JupyterLab, use the same environment:

```powershell
.\.venv\Scripts\python.exe run_notebooks.py 01_data.ipynb
.\.venv\Scripts\python.exe run_notebooks.py 02_model.ipynb
.\.venv\Scripts\python.exe run_notebooks.py 03_scoring.ipynb
.\.venv\Scripts\python.exe run_notebooks.py 04_drift.ipynb
```

Or run `run_notebooks.py` without a notebook argument to execute all four in
order. On macOS/Linux, replace the Python path with `.venv/bin/python`.
The runner starts a fresh kernel for each stage, stops on the first failure,
and saves executed copies under `output/executed/`. It leaves source notebooks
unchanged. Those copies include tables, the plot, and any error traceback.

## Repository layout

```text
RetainAi/
  README.md
  PROBLEM_STATEMENT.md
  LICENSE
  requirements.txt
  run_notebooks.py          Optional command-line notebook runner
  .github/workflows/       Dependency, test, and notebook checks
  data/
    customer_churn.csv     Unmodified public source
    source.json            Attribution and SHA-256 fingerprint
  notebooks/
    01_data.ipynb
    02_model.ipynb
    03_scoring.ipynb
    04_drift.ipynb
  retainai/
    features.py            Input validation and two usage ratios
    scoring.py             Model loading and contact rules
    __init__.py
  tests/
    test_helpers.py
```

The helpers keep shared rules in one place. The saved model also needs an
importable feature function when loaded in another session. Data preparation,
training, evaluation, and drift calculations remain visible in the notebooks.

## Data and evaluation choices

The [UCI Iranian Churn dataset](https://doi.org/10.24432/C5JW3Z) contains
3,150 anonymous customer records. Inputs summarize the first nine months;
churn is measured at month twelve, leaving a three-month planning gap. The
source CSV has no dates for a chronological backtest.

The file has no missing cells, 300 exact duplicate rows, and 314 repeated rows
when comparing only the 12 model predictors. That leaves 2,836 unique predictor
profiles. Records are retained because the anonymous source does not establish
whether matching rows represent different customers or duplicate entries.

Identical predictor profiles stay in one partition, reducing overlap between
training and evaluation. With seed 42, the split is:

| Partition | Rows | Use |
| --- | ---: | --- |
| Training | 2,224 | Fit preprocessing, models, and calibration |
| Validation | 458 | Select a candidate |
| Test | 468 | Evaluate the selected candidate |

The split targets 70% / 15% / 15% of profiles, so row percentages differ.
Calibration folds keep identical profiles together too. The model uses the
12 predictors plus two smoothed ratios: seconds per call and call failure
rate. Churn, customer ID, and customer value are excluded from model inputs.

Notebook 02 compares logistic regression with sigmoid-calibrated histogram
gradient boosting. Higher validation average precision selects the winner;
lower Brier score breaks a tie. Test results do not select the model.
Preprocessing fits inside each training pipeline and calibration fold.

## Reference results

The full workflow was rerun in separate Python 3.14.4 kernels on 24 October 2026.
The selected model was `calibrated_gradient_boosting`.

| Test metric | Result |
| --- | ---: |
| Average precision | 0.8978 |
| ROC-AUC | 0.9807 |
| Recall in the highest-risk 20% | 0.9583 |
| Lift in the highest-risk 10% | 5.9468 |
| Brier score | 0.0353 |

Average precision summarizes precision across recall increments. Recall at 20%
measures how many observed churners appear in the highest-risk fifth. Lift
compares the churn concentration in the highest-risk tenth with the full test
set. Brier score measures overall squared probability error; lower is better.

These top-fraction metrics rank by probability. The campaign list below ranks
by probability times value. Check your own `output/metrics.json`; library and
platform differences can change results. Repeatedly tuning against the test
score would turn the holdout into another validation set.

## Score your own batch

Run notebooks 01 and 02 first. Copy the format of
`output/sample_customers.csv` into a separate file such as
`data/my_customers.csv`. Use the clean column names below. Change `INPUT_PATH`
in notebook 03 to `ROOT / "data/my_customers.csv"`; use the same path in
notebook 04 if you want a drift report for that batch.

| Required column | Accepted values |
| --- | --- |
| `customer_id` | Unique nonempty string within the batch |
| `call_failure` | Nonnegative integer |
| `subscription_length` | Nonnegative integer, months |
| `charge_amount` | Integer tier 0–10 |
| `seconds_of_use` | Nonnegative integer |
| `frequency_of_use` | Nonnegative integer |
| `frequency_of_sms` | Nonnegative integer |
| `distinct_called_numbers` | Nonnegative integer |
| `age` | Integer 0–120 |
| `complains` | 0: no complaint, 1: complaint |
| `age_group` | Integer 1–5, younger to older |
| `tariff_plan` | 1: pay as you go, 2: contractual |
| `status` | 1: active, 2: non-active |
| `customer_value` | Finite nonnegative number; source currency is unspecified |

The churn label is optional for scoring and is ignored if present. UCI describes
charge tiers as 0–9, but seven source rows contain 10. Validation allows 0–10
and preserves that discrepancy. It rejects missing required fields, invalid
codes, nonfinite values, duplicate IDs, and empty batches.

Notebook 03 exposes three settings: `CAPACITY=100`, `COST=20.0`, and
`RETENTION_RATE=0.25`. They are hypothetical campaign assumptions:

```text
expected_churn_loss = churn_probability × customer_value
scenario_net_benefit = expected_churn_loss × RETENTION_RATE − COST
contact_recommended = within capacity AND scenario_net_benefit > 0
```

Probability 0.8 and value 1,000 imply expected loss 800 and assumed net benefit
180 at these settings. Value and cost share hypothetical source units. There
is no measured retention effect or established currency in this dataset.

`scores.csv` is ordered by expected loss, highest first. Ties keep input order.
Decisions use full precision before display rounding. The 0/1 prediction uses
0.5, and illustrative risk tiers use 0.3 and 0.7; contact recommendations use
the separate capacity and benefit rule. No customers are contacted.

New files under `data/` are ignored by default. Keep private batches local and
preserve the original source CSV and its fingerprint. To use another dataset
for training, adapt notebook 01, the feature schema, validation bounds, source
attribution, and evaluation design together. Do not simply replace the source
file or change its hash to bypass a check.

## Drift and limitations

Notebook 04 compares inputs and predicted probabilities with the training
reference using Population Stability Index (PSI). The largest PSI gives a
heuristic label: below 0.10 is `stable`, 0.10 to below 0.25 is `watch`, and
0.25 or above is `review`. Bins use training quantiles; smoothing avoids
division by zero. Results depend on binning and sample size.

A separate synthetic shift checks that the code reacts to an extreme change.
It does not overwrite the saved batch report. Investigate collection errors,
units, and customer mix before retraining. Drift does not prove accuracy loss;
stable distributions do not prove accuracy either.

This is an offline study on a small historical dataset. It does not establish
future performance, fairness across groups, campaign savings, or production
reliability. A high churn score does not mean contact will change the outcome.
Campaign uplift needs intervention data and an appropriate evaluation. There
is no deployed prediction service or automatic retraining.

## Checks and local files

Run the helper tests with the setup command above after changing validation,
features, or scoring. They check arithmetic, capacity, invalid inputs, ties,
and precision-sensitive decisions using known probabilities. Notebook assertions
check the source hash, partition and calibration separation, row preservation,
excluded columns, saved-model reload, contact limits, and synthetic drift.
GitHub Actions runs dependency checks, helper tests, and all four notebooks.

Generated models, reports, executed notebooks, the virtual environment, editor
settings, private data files, and local Markdown guides are ignored. Only the
root README and problem statement are published among Markdown files. Source
notebooks are committed with cell outputs cleared. Clear outputs before
committing notebook experiments; saved outputs can contain private inputs.

Stop kernels before removing `output/` to reset results, then run the notebooks
again in order. Only load joblib files you created or trust: model loading can
execute Python code. Stop JupyterLab with Ctrl+C when finished.

| Problem | Next step |
| --- | --- |
| Import fails or the wrong kernel appears | Install requirements and launch Jupyter using the same `.venv` Python |
| Source fingerprint mismatch | Restore the original CSV; do not bypass the hash check |
| Split files or model missing | Run the earlier numbered notebooks completely |
| An invalid row is rejected | Check the schema above, including IDs and category ranges |
| Results look stale | Restart the kernel and rerun downstream stages after upstream changes |

## License and source

Project code is available under the [MIT license](LICENSE). The unmodified
dataset has its own [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

Iranian Churn [Dataset]. (2020). UCI Machine Learning Repository.
[DOI: 10.24432/C5JW3Z](https://doi.org/10.24432/C5JW3Z).
`data/source.json` records the source download URL and SHA-256 fingerprint.
