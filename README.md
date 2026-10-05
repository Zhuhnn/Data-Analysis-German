# Data Mining Analysis: Statlog (German Credit Data)

A data-mining study of the [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
dataset (UCI id 144). The data is loaded with the [`ucimlrepo`](https://github.com/uci-ml-repo/ucimlrepo) package.

**Report:** [`reports/German_Credit_Data_Mining_Report.pdf`](reports/German_Credit_Data_Mining_Report.pdf)

## How the data is loaded

```python
from ucimlrepo import fetch_ucirepo

statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets
print(statlog_german_credit_data.metadata)
print(statlog_german_credit_data.variables)
```

`src/load_data.py` calls `fetch_ucirepo(id=144)` first. If the UCI API cannot be reached, it rebuilds the same
object (`data` / `metadata` / `variables`, same column names) from the bundled copy of
`statlog+german+credit+data.zip` in `data/statlog_german_credit/`.

## What the analysis covers

1. Metadata and variable information from ucimlrepo
2. Data quality checks and descriptive statistics
3. Bivariate association tests (chi-square with Cramer's V, Mann-Whitney U)
4. Classification with 7 models, stratified 10-fold CV and a 70/30 hold-out split, evaluated with the
   dataset's 5:1 cost matrix and cost-optimal threshold tuning
5. Feature importance (permutation importance, logistic-regression odds ratios)
6. k-means customer segmentation with a PCA view
7. Apriori association rules for bad and good credit
8. Fairness check on sensitive attributes

## Reproduce

```bash
pip install -r requirements.txt
python src/analysis.py       # writes reports/results.json and reports/figures/*.png
python src/build_report.py   # writes reports/German_Credit_Data_Mining_Report.pdf
```
