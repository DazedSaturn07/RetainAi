# Problem statement: choosing a telecom retention contact list

A telecom team has more customers than it can contact. Some are likely to leave,
but contacting everyone costs time and money. A useful first step is to estimate
churn risk and build a short, explainable list for review.

RetainAI asks two questions:

1. Can historical customer activity help rank customers who later leave?
2. Given a contact limit and stated cost assumptions, which customers would be
   worth considering for a retention campaign?

## Who this is for

This is a small, reproducible project for someone learning tabular machine
learning or exploring an offline retention workflow. The notebooks show the
decisions and intermediate tables so a reader can change the inputs, models,
or campaign assumptions and understand what changed.

## Available evidence

The public [UCI Iranian Churn dataset](https://doi.org/10.24432/C5JW3Z) contains
3,150 anonymous records. Inputs summarize the first nine months; the churn
label records customer status at month twelve. Inputs include calls, SMS use,
complaints, subscription length, charge tier, tariff, account status, and age.

There are repeated input profiles, so splitting individual rows at random could
place the same profile in both training and evaluation. RetainAI keeps identical
profiles together. It also leaves the churn label, generated ID, and customer
value out of the model inputs. Value is reserved for the later ranking scenario.

The data has no event dates for a chronological backtest, no documented currency
for customer value, and no retention campaign outcomes.

## What the project does

| Stage | Decision | Output |
| --- | --- | --- |
| Prepare | Validate records and keep repeated profiles together | Training, validation, and test CSVs |
| Model | Compare logistic regression and calibrated gradient boosting using validation average precision | Selected model and holdout metrics |
| Score | Combine churn probability with value and explicit contact assumptions | Ranked CSV with contact recommendations |
| Monitor | Compare an incoming batch with training distributions | PSI drift report for investigation |

The contact rule uses these assumptions:

```text
expected churn loss = probability of churn × customer value
scenario net benefit = expected churn loss × assumed saved fraction − contact cost
```

Customers are ranked by expected loss. A contact is suggested only within the
capacity limit and when the unrounded scenario benefit is positive. Cost and
saved fraction are constant across the batch. They are editable assumptions,
not learned treatment effects.

## What counts as a useful result

- The data stays traceable to its public source, with its original bytes intact.
- Repeated predictor profiles do not cross partitions or calibration folds.
- Model selection uses validation data before the test results are inspected.
- A saved model reloads in a fresh session and scores a compatible batch.
- Capacity limits and campaign arithmetic behave as documented.
- Drift checks detect a deliberately shifted example without triggering retraining.
- Another reader can clone the repository and rerun the same workflow.

The README reports the observed metrics and explains how to reproduce them.
There is no preset score that establishes business success.

## What remains outside this study

A high churn score does not show that a customer will respond to contact. The
scenario cannot establish saved revenue, profit, or campaign uplift. Those
claims would need intervention data and a controlled evaluation.

This repository also does not establish future accuracy, fairness across age
groups, or reliability in a deployed service. A real rollout would need current,
authorized data, an appropriate time-based evaluation, segment checks, measured
campaign outcomes, and an agreed operating process.

Start with [the README](README.md), then run the four numbered notebooks. To try
your own batch, follow its input-schema instructions and keep customer files
local.
