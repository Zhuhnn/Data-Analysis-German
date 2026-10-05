"""Load the Statlog (German Credit Data) dataset the ucimlrepo way.

The primary path is exactly what the ucimlrepo README prescribes:

    from ucimlrepo import fetch_ucirepo
    statlog_german_credit_data = fetch_ucirepo(id=144)

If the UCI API cannot be reached (e.g. no internet access), the same
`dotdict` object (data / metadata / variables) is rebuilt from the local copy
of `statlog+german+credit+data.zip`, so the rest of the analysis is
identical either way.
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo
from ucimlrepo.dotdict import dotdict

UCI_ID = 144
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "statlog_german_credit"

# (name, type, demographic, description, units) for each column, as in german.doc
VARIABLES = [
    ("Attribute1", "Categorical", None, "Status of existing checking account", None),
    ("Attribute2", "Integer", None, "Duration", "months"),
    ("Attribute3", "Categorical", None, "Credit history", None),
    ("Attribute4", "Categorical", None, "Purpose", None),
    ("Attribute5", "Integer", None, "Credit amount", "DM"),
    ("Attribute6", "Categorical", None, "Savings account/bonds", None),
    ("Attribute7", "Categorical", "Other", "Present employment since", None),
    ("Attribute8", "Integer", None, "Installment rate in percentage of disposable income", None),
    ("Attribute9", "Categorical", "Marital Status", "Personal status and sex", None),
    ("Attribute10", "Categorical", None, "Other debtors / guarantors", None),
    ("Attribute11", "Integer", None, "Present residence since", None),
    ("Attribute12", "Categorical", None, "Property", None),
    ("Attribute13", "Integer", "Age", "Age", "years"),
    ("Attribute14", "Categorical", None, "Other installment plans", None),
    ("Attribute15", "Categorical", "Other", "Housing", None),
    ("Attribute16", "Integer", None, "Number of existing credits at this bank", None),
    ("Attribute17", "Categorical", "Occupation", "Job", None),
    ("Attribute18", "Integer", None, "Number of people being liable to provide maintenance for", None),
    ("Attribute19", "Binary", None, "Telephone", None),
    ("Attribute20", "Binary", "Other", "Foreign worker", None),
    ("class", "Binary", None, "1 = Good, 2 = Bad", None),
]

# Human-readable short names used in tables and charts
SHORT_NAMES = {
    "Attribute1": "checking_status", "Attribute2": "duration", "Attribute3": "credit_history",
    "Attribute4": "purpose", "Attribute5": "credit_amount", "Attribute6": "savings",
    "Attribute7": "employment", "Attribute8": "installment_rate", "Attribute9": "personal_status_sex",
    "Attribute10": "other_debtors", "Attribute11": "residence_since", "Attribute12": "property",
    "Attribute13": "age", "Attribute14": "other_installment_plans", "Attribute15": "housing",
    "Attribute16": "existing_credits", "Attribute17": "job", "Attribute18": "num_dependents",
    "Attribute19": "telephone", "Attribute20": "foreign_worker",
}

# Code book from german.doc
CODES = {
    "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking acct",
    "A30": "no credits / all paid", "A31": "all paid at this bank", "A32": "existing paid duly",
    "A33": "past delays", "A34": "critical account",
    "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/TV",
    "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
    "A48": "retraining", "A49": "business", "A410": "others",
    "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM", "A64": ">= 1000 DM",
    "A65": "unknown / none",
    "A71": "unemployed", "A72": "< 1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">= 7 yrs",
    "A91": "male: div/sep", "A92": "female: div/sep/mar", "A93": "male: single",
    "A94": "male: mar/wid", "A95": "female: single",
    "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    "A121": "real estate", "A122": "savings agr/life ins", "A123": "car or other",
    "A124": "unknown / none",
    "A141": "bank", "A142": "stores", "A143": "none",
    "A151": "rent", "A152": "own", "A153": "for free",
    "A171": "unemployed/unskilled non-res", "A172": "unskilled resident",
    "A173": "skilled employee", "A174": "management/highly qualified",
    "A191": "none", "A192": "yes",
    "A201": "yes", "A202": "no",
}


def _load_local():
    names = [v[0] for v in VARIABLES]
    df = pd.read_csv(DATA_DIR / "german.data", sep=r"\s+", header=None, names=names)
    variables = pd.DataFrame(
        [
            {"name": n, "role": "Target" if n == "class" else "Feature", "type": t,
             "demographic": d, "description": desc, "units": u, "missing_values": "no"}
            for n, t, d, desc, u in VARIABLES
        ]
    )
    feature_cols = variables.loc[variables.role == "Feature", "name"].tolist()
    metadata = {
        "uci_id": UCI_ID,
        "name": "Statlog (German Credit Data)",
        "repository_url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "dataset_doi": "10.24432/C5NC77",
        "creators": ["Hans Hofmann"],
        "task": "Classification",
        "characteristics": "Multivariate",
        "num_instances": len(df),
        "num_features": len(feature_cols),
        "feature_types": ["Categorical", "Integer"],
        "target_col": ["class"],
        "index_col": None,
        "has_missing_values": "no",
        "year_of_dataset_creation": 1994,
        "additional_info": dotdict({
            "summary": "This dataset classifies people described by a set of attributes as good "
                       "or bad credit risks. Comes in two formats (one all numeric). Also comes "
                       "with a cost matrix.",
            "variable_info": open(DATA_DIR / "german.doc", encoding="latin-1").read(),
        }),
        "source": "local copy of statlog+german+credit+data.zip",
    }
    data = {
        "ids": None,
        "features": df[feature_cols],
        "targets": df[["class"]],
        "original": df,
        "headers": df.columns,
    }
    return dotdict({"data": dotdict(data), "metadata": dotdict(metadata), "variables": variables})


def load_german_credit():
    """Return (dataset, source) where dataset has the fetch_ucirepo structure."""
    try:
        ds = fetch_ucirepo(id=UCI_ID)
        ds.metadata["source"] = "UCI ML Repository API (fetch_ucirepo)"
        return ds, "fetch_ucirepo(id=144)"
    except Exception as exc:  # no network / API unavailable
        print(f"fetch_ucirepo(id={UCI_ID}) failed ({exc}); using local zip copy.")
        return _load_local(), f"local fallback (fetch_ucirepo failed: {exc})"
