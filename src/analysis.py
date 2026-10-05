"""Data-mining analysis of the Statlog (German Credit Data) dataset (UCI id 144).

Steps (CRISP-DM style):
  1. Data acquisition with ucimlrepo (fetch_ucirepo)
  2. Data understanding: metadata, variables, class balance, descriptive stats
  3. Bivariate analysis: chi-square / Cramer's V, Mann-Whitney U
  4. Pre-processing: one-hot encoding + scaling
  5. Classification models with stratified 10-fold CV and a hold-out test set,
     evaluated on accuracy, ROC-AUC, F1 and the dataset's 5:1 cost matrix
  6. Cost-sensitive threshold tuning
  7. Feature importance (permutation importance, logistic coefficients)
  8. Clustering (k-means) and association-rule mining (Apriori)

Run:  python src/analysis.py
Outputs go to reports/figures/ and reports/results.json
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import CODES, SHORT_NAMES, load_german_credit  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42

# Chart styling (validated reference palette, light mode)
GOOD, BAD = "#2a78d6", "#eb6834"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
})
COST_FN, COST_FP = 5, 1  # bad predicted good costs 5, good predicted bad costs 1


def savefig(name):
    plt.savefig(FIG / name)
    plt.close()


def cost(y_true, y_pred):
    """Total misclassification cost; y = 1 means Bad."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(COST_FN * fn + COST_FP * fp)


def cramers_v(table):
    chi2, p, dof, _ = chi2_contingency(table)
    n = table.values.sum()
    r, k = table.shape
    return chi2, p, np.sqrt(chi2 / (n * (min(r, k) - 1)))


def main():
    R = {}

    # 1. Acquisition ---------------------------------------------------------
    ds, source = load_german_credit()
    X_raw = ds.data.features
    y_raw = ds.data.targets
    meta = ds.metadata
    R["source"] = source
    R["metadata"] = {k: meta.get(k) for k in [
        "uci_id", "name", "repository_url", "dataset_doi", "creators", "task",
        "characteristics", "num_instances", "num_features", "feature_types",
        "target_col", "has_missing_values", "year_of_dataset_creation"]}
    R["metadata"]["summary"] = (meta.additional_info or {}).get("summary")
    R["variables"] = ds.variables.fillna("").to_dict(orient="records")

    df = X_raw.rename(columns=SHORT_NAMES).copy()
    y = (y_raw["class"] == 2).astype(int).rename("bad")  # 1 = bad credit risk
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    R["num_cols"], R["cat_cols"] = num_cols, cat_cols
    R["shape"] = list(df.shape)
    R["missing_total"] = int(df.isna().sum().sum())
    R["duplicates"] = int(pd.concat([df, y], axis=1).duplicated().sum())
    R["class_counts"] = {"good": int((y == 0).sum()), "bad": int((y == 1).sum())}

    # 2. Data understanding -------------------------------------------------
    desc = df[num_cols].describe().T
    desc["skew"] = df[num_cols].skew()
    R["numeric_summary"] = desc.round(2).reset_index().rename(columns={"index": "variable"}).to_dict("records")

    # class balance
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    counts = [R["class_counts"]["good"], R["class_counts"]["bad"]]
    bars = ax.barh(["Good (1)", "Bad (2)"], counts, color=[GOOD, BAD], height=0.55)
    for b, c in zip(bars, counts):
        ax.text(b.get_width() + 10, b.get_y() + b.get_height() / 2, f"{c} ({c/10:.0f}%)",
                va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 820); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_title("Target class distribution (n = 1000)", loc="left")
    ax.set_xlabel("Number of applicants")
    savefig("class_balance.png")

    # numeric distributions by class
    fig, axes = plt.subplots(2, 4, figsize=(9, 4.4))
    for ax, c in zip(axes.ravel(), num_cols):
        bins = np.histogram_bin_edges(df[c], bins=min(20, df[c].nunique()))
        ax.hist(df.loc[y == 0, c], bins=bins, alpha=0.75, color=GOOD, label="Good", density=True)
        ax.hist(df.loc[y == 1, c], bins=bins, alpha=0.65, color=BAD, label="Bad", density=True)
        ax.set_title(c, loc="left", fontsize=9); ax.set_yticks([])
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Numeric attributes: density by class", x=0.01, ha="left", fontweight="bold", color=INK)
    plt.tight_layout()
    savefig("numeric_distributions.png")

    # correlation heatmap
    corr = pd.concat([df[num_cols], y], axis=1).corr(method="spearman")
    R["spearman_corr"] = corr.round(3).to_dict()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns); ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.55 else INK)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Spearman correlation (numeric + bad flag)", loc="left")
    savefig("correlation.png")

    # 3. Bivariate tests ----------------------------------------------------
    assoc = []
    for c in cat_cols:
        tab = pd.crosstab(df[c], y)
        chi2, p, v = cramers_v(tab)
        assoc.append({"variable": c, "type": "categorical", "test": "chi-square",
                      "statistic": round(chi2, 2), "p_value": p, "effect": round(v, 3)})
    for c in num_cols:
        u, p = mannwhitneyu(df.loc[y == 0, c], df.loc[y == 1, c])
        # rank-biserial correlation as effect size
        rbc = 1 - 2 * u / ((y == 0).sum() * (y == 1).sum())
        assoc.append({"variable": c, "type": "numeric", "test": "Mann-Whitney U",
                      "statistic": round(float(u), 1), "p_value": p, "effect": round(abs(rbc), 3),
                      "median_good": float(df.loc[y == 0, c].median()),
                      "median_bad": float(df.loc[y == 1, c].median())})
    assoc = sorted(assoc, key=lambda d: d["p_value"])
    R["bivariate"] = assoc

    fig, ax = plt.subplots(figsize=(6.5, 5))
    a = pd.DataFrame(assoc).sort_values("p_value", ascending=False)
    lp = -np.log10(a["p_value"].clip(lower=1e-30))
    colors = [SERIES[0] if t == "categorical" else SERIES[2] for t in a["type"]]
    ax.barh(a["variable"], lp, color=colors, height=0.65)
    ax.axvline(-np.log10(0.05), color=INK2, lw=1, ls="--")
    ax.text(-np.log10(0.05) + 0.2, 0.2, "p = 0.05", color=INK2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("-log10(p-value)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=SERIES[0], label="categorical (chi-square)"),
                       Patch(color=SERIES[2], label="numeric (Mann-Whitney U)")], loc="lower right")
    ax.set_title("Strength of association with credit risk", loc="left")
    savefig("association_strength.png")

    # bad rate by category for the top categorical drivers
    top_cats = [d["variable"] for d in assoc if d["type"] == "categorical"][:6]
    bad_rate_tables = {}
    fig, axes = plt.subplots(3, 2, figsize=(9, 8))
    for ax, c in zip(axes.ravel(), top_cats):
        g = pd.DataFrame({"cat": df[c].map(CODES).fillna(df[c]), "bad": y}).groupby("cat")["bad"].agg(["mean", "count"])
        g = g.sort_values("mean")
        bad_rate_tables[c] = g.round(3).reset_index().to_dict("records")
        ax.barh(g.index, g["mean"] * 100, color=BAD, height=0.6)
        for i, (m, n) in enumerate(zip(g["mean"], g["count"])):
            ax.text(m * 100 + 1, i, f"{m*100:.0f}%  (n={n})", va="center", fontsize=7.5, color=INK)
        ax.axvline(30, color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0, max(g["mean"] * 100) + 25); ax.grid(axis="y", visible=False)
        ax.set_title(c, loc="left", fontsize=9); ax.tick_params(axis="y", labelsize=7.5)
    fig.suptitle("Bad-credit rate by category (dashed line = 30% overall rate)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    plt.tight_layout()
    savefig("bad_rate_by_category.png")
    R["bad_rate_tables"] = bad_rate_tables

    # 4. Pre-processing & 5. Modelling --------------------------------------
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
        "SVM (RBF)": SVC(C=1.0, probability=True, random_state=SEED),
        "k-NN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
    }
    X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.3, stratify=y, random_state=SEED)
    R["split"] = {"train": len(X_tr), "test": len(X_te)}
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    cv_rows, test_rows, roc_data, fitted = [], [], {}, {}
    thr_opt = COST_FP / (COST_FP + COST_FN)  # Bayes-optimal threshold = 1/6
    R["bayes_threshold"] = thr_opt
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("clf", m)])
        # 10-fold CV on whole dataset (out-of-fold probabilities)
        p_cv = cross_val_predict(pipe, df, y, cv=skf, method="predict_proba")[:, 1]
        fold_auc = [roc_auc_score(y.iloc[te], p_cv[te]) for _, te in skf.split(df, y)]
        cv_rows.append({
            "model": name,
            "accuracy": accuracy_score(y, p_cv >= 0.5),
            "roc_auc": roc_auc_score(y, p_cv), "auc_sd": float(np.std(fold_auc)),
            "f1_bad": f1_score(y, p_cv >= 0.5),
            "recall_bad": recall_score(y, p_cv >= 0.5),
            "cost_at_0.5": cost(y, p_cv >= 0.5),
            "cost_at_1/6": cost(y, p_cv >= thr_opt),
            "accuracy_at_1/6": accuracy_score(y, p_cv >= thr_opt),
        })
        # hold-out test
        pipe.fit(X_tr, y_tr)
        fitted[name] = pipe
        p = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_te, p))
        for thr, tag in [(0.5, "0.5"), (thr_opt, "1/6")]:
            pred = (p >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
            test_rows.append({
                "model": name, "threshold": tag, "accuracy": accuracy_score(y_te, pred),
                "precision_bad": precision_score(y_te, pred, zero_division=0),
                "recall_bad": recall_score(y_te, pred), "f1_bad": f1_score(y_te, pred),
                "roc_auc": roc_auc_score(y_te, p), "cost": cost(y_te, pred),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
    # baselines
    R["baseline"] = {
        "all_good_cost_full": int(COST_FN * (y == 1).sum()),
        "all_bad_cost_full": int(COST_FP * (y == 0).sum()),
        "all_good_accuracy": float((y == 0).mean()),
        "all_good_cost_test": int(COST_FN * (y_te == 1).sum()),
        "all_bad_cost_test": int(COST_FP * (y_te == 0).sum()),
    }
    cv_df = pd.DataFrame(cv_rows).sort_values("roc_auc", ascending=False)
    R["cv_results"] = cv_df.round(4).to_dict("records")
    R["test_results"] = pd.DataFrame(test_rows).round(4).to_dict("records")

    # ROC chart
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    order = cv_df["model"].tolist()
    for i, name in enumerate(order):
        fpr, tpr, auc = roc_data[name]
        ax.plot(fpr, tpr, color=SERIES[i], lw=1.8, label=f"{name} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("False positive rate (good flagged as bad)")
    ax.set_ylabel("True positive rate (bad caught)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("ROC curves on hold-out test set (n = 300)", loc="left")
    savefig("roc_curves.png")

    # model comparison chart (CV AUC and cost)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    c = cv_df.sort_values("roc_auc")
    axes[0].barh(c["model"], c["roc_auc"], xerr=c["auc_sd"], color=GOOD, height=0.6,
                 error_kw={"ecolor": INK2, "lw": 0.8})
    for i, v in enumerate(c["roc_auc"]):
        axes[0].text(0.505, i, f"{v:.3f}", va="center", fontsize=7.5, color="white")
    axes[0].set_xlim(0.5, 0.85); axes[0].grid(axis="y", visible=False)
    axes[0].set_title("10-fold CV ROC-AUC (higher is better)", loc="left")
    w = 0.38; idx = np.arange(len(c))
    axes[1].barh(idx + w / 2, c["cost_at_0.5"], height=w, color=SERIES[3], label="threshold 0.5")
    axes[1].barh(idx - w / 2, c["cost_at_1/6"], height=w, color=SERIES[1], label="threshold 1/6 (cost-optimal)")
    axes[1].axvline(R["baseline"]["all_bad_cost_full"], color=INK2, ls="--", lw=0.8)
    axes[1].text(R["baseline"]["all_bad_cost_full"] + 8, -0.95, "reject everyone = 700", fontsize=7, color=INK2)
    axes[1].set_yticks(idx); axes[1].set_yticklabels([]); axes[1].grid(axis="y", visible=False)
    axes[1].set_title("10-fold CV total cost (lower is better)", loc="left")
    axes[1].set_ylim(-1.1, len(c) - 0.4)
    axes[1].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.tight_layout()
    savefig("model_comparison.png")

    # 6. Cost vs threshold for best model --------------------------------
    best = cv_df.sort_values("cost_at_1/6").iloc[0]["model"]
    R["best_model"] = best
    pipe = Pipeline([("pre", pre), ("clf", models[best])])
    p_cv = cross_val_predict(pipe, df, y, cv=skf, method="predict_proba")[:, 1]
    thrs = np.linspace(0.02, 0.9, 89)
    costs = [cost(y, p_cv >= t) for t in thrs]
    t_best = float(thrs[int(np.argmin(costs))])
    R["threshold_curve"] = {"best_threshold_empirical": t_best, "min_cost": int(min(costs))}
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(thrs, costs, color=GOOD, lw=2)
    ax.axvline(thr_opt, color=BAD, ls="--", lw=1)
    ax.text(thr_opt + 0.01, max(costs) * 0.95, "1/6 (theory)", color=INK2, fontsize=8)
    ax.axvline(0.5, color=INK2, ls=":", lw=1)
    ax.text(0.51, max(costs) * 0.95, "0.5 (default)", color=INK2, fontsize=8)
    ax.set_xlabel("Decision threshold on P(bad)"); ax.set_ylabel("Total cost (1000 applicants)")
    ax.set_title(f"Misclassification cost vs threshold - {best} (10-fold CV)", loc="left")
    savefig("cost_threshold.png")

    # 7. Feature importance -------------------------------------------------
    rf = fitted["Random Forest"]
    pi = permutation_importance(rf, X_te, y_te, scoring="roc_auc", n_repeats=20, random_state=SEED)
    imp = pd.DataFrame({"feature": df.columns, "mean": pi.importances_mean, "sd": pi.importances_std})
    imp = imp.sort_values("mean", ascending=False)
    R["permutation_importance_rf"] = imp.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(6, 4.6))
    s = imp.sort_values("mean")
    ax.barh(s["feature"], s["mean"], xerr=s["sd"], color=GOOD, height=0.6, error_kw={"ecolor": INK2, "lw": 0.7})
    ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in test ROC-AUC when feature is shuffled")
    ax.set_title("Permutation importance - Random Forest", loc="left")
    savefig("feature_importance.png")

    lr = fitted["Logistic Regression"]
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = lr.named_steps["clf"].coef_[0]
    co = pd.DataFrame({"term": names, "coef": coefs})

    def pretty(t):
        t = t.replace("num__", "").replace("cat__", "")
        if "_A" in t:
            base, code = t.rsplit("_", 1)
            return f"{base} = {CODES.get(code, code)}"
        return t + " (std)"
    co["term"] = co["term"].map(pretty)
    co["odds_ratio"] = np.exp(co["coef"])
    co = co.reindex(co["coef"].abs().sort_values(ascending=False).index)
    R["logit_top_terms"] = co.head(16).round(3).to_dict("records")
    top = co.head(14).sort_values("coef")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(top["term"], top["coef"], color=[BAD if v > 0 else GOOD for v in top["coef"]], height=0.6)
    ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("Log-odds coefficient (positive = raises risk of bad credit)")
    ax.set_title("Logistic regression: strongest terms", loc="left")
    savefig("logit_coefficients.png")

    # interpretable decision rules
    dt = fitted["Decision Tree"]
    dt_names = [pretty(n) for n in dt.named_steps["pre"].get_feature_names_out()]
    R["tree_rules"] = export_text(dt.named_steps["clf"], feature_names=dt_names, max_depth=3, decimals=2)

    # 8a. Clustering --------------------------------------------------------
    Xs = pre.fit_transform(df)
    sil = {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
        sil[k] = float(silhouette_score(Xs, km.labels_))
    R["silhouette"] = sil
    K = 4
    km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(Xs)
    lab = km.labels_
    prof = df[num_cols].groupby(lab).median()
    prof["size"] = pd.Series(lab).value_counts().sort_index()
    prof["bad_rate"] = y.groupby(lab).mean()
    for cc in ["checking_status", "savings", "housing", "purpose"]:
        prof[f"top_{cc}"] = df[cc].groupby(lab).agg(lambda s: CODES.get(s.mode()[0], s.mode()[0]))
    R["cluster_k"] = K
    R["cluster_profiles"] = prof.round(3).reset_index().rename(columns={"index": "cluster"}).to_dict("records")
    pca = PCA(n_components=2, random_state=SEED).fit(Xs)
    Z = pca.transform(Xs)
    R["pca_var"] = pca.explained_variance_ratio_.round(4).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for k in range(K):
        mk = lab == k
        axes[0].scatter(Z[mk, 0], Z[mk, 1], s=10, color=SERIES[k], alpha=0.7,
                        label=f"Cluster {k} (bad {prof.loc[k, 'bad_rate']*100:.0f}%)", edgecolors="white", linewidths=0.3)
    axes[0].legend(fontsize=7.5, markerscale=1.8)
    axes[0].set_title("k-means clusters (k = 4) in PCA space", loc="left")
    axes[0].set_xlabel(f"PC1 ({R['pca_var'][0]*100:.1f}% var)"); axes[0].set_ylabel(f"PC2 ({R['pca_var'][1]*100:.1f}% var)")
    for cls, col, lbl in [(0, GOOD, "Good"), (1, BAD, "Bad")]:
        mk = (y == cls).values
        axes[1].scatter(Z[mk, 0], Z[mk, 1], s=10, color=col, alpha=0.6, label=lbl, edgecolors="white", linewidths=0.3)
    axes[1].legend(fontsize=7.5, markerscale=1.8)
    axes[1].set_title("Same projection coloured by true class", loc="left")
    axes[1].set_xlabel("PC1")
    plt.tight_layout()
    savefig("clusters_pca.png")

    # 8b. Association rules -------------------------------------------------
    bins = df.copy()
    bins["duration"] = pd.cut(df["duration"], [0, 12, 24, 36, 100], labels=["<=12m", "13-24m", "25-36m", ">36m"])
    bins["credit_amount"] = pd.qcut(df["credit_amount"], 3, labels=["low", "mid", "high"])
    bins["age"] = pd.cut(df["age"], [0, 25, 35, 50, 100], labels=["<=25", "26-35", "36-50", ">50"])
    items = []
    for c in ["checking_status", "credit_history", "purpose", "savings", "employment", "property",
              "housing", "other_installment_plans", "duration", "credit_amount", "age"]:
        items.append(pd.get_dummies(bins[c].map(lambda v: CODES.get(v, v)).astype(str), prefix=c, prefix_sep="="))
    items.append(pd.DataFrame({"class=BAD": y == 1, "class=GOOD": y == 0}))
    basket = pd.concat(items, axis=1).astype(bool)
    fi = apriori(basket, min_support=0.03, use_colnames=True, max_len=4)
    rules = association_rules(fi, metric="confidence", min_threshold=0.5)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset({"class=BAD"}))]
    rules = rules[rules["lift"] >= 1.5].sort_values(["confidence", "support"], ascending=False)
    # drop redundant rules whose antecedent is a superset of a stronger rule
    kept = []
    for _, r in rules.iterrows():
        if not any(k["antecedents"] <= r["antecedents"] and k["confidence"] >= r["confidence"] for k in kept):
            kept.append(r)
    R["n_rules_bad"] = len(rules)
    R["assoc_rules_bad"] = [{
        "if": " AND ".join(sorted(r["antecedents"])), "support": round(r["support"], 3),
        "confidence": round(r["confidence"], 3), "lift": round(r["lift"], 2),
        "count": int(round(r["support"] * len(df)))} for r in kept[:12]]
    rules_g = association_rules(fi, metric="confidence", min_threshold=0.85)
    rules_g = rules_g[rules_g["consequents"].apply(lambda s: s == frozenset({"class=GOOD"}))]
    rules_g = rules_g.sort_values(["confidence", "support"], ascending=False)
    kept_g = []
    for _, r in rules_g.iterrows():
        if not any(k["antecedents"] <= r["antecedents"] and k["confidence"] >= r["confidence"] for k in kept_g):
            kept_g.append(r)
    R["assoc_rules_good"] = [{
        "if": " AND ".join(sorted(r["antecedents"])), "support": round(r["support"], 3),
        "confidence": round(r["confidence"], 3), "lift": round(r["lift"], 2),
        "count": int(round(r["support"] * len(df)))} for r in kept_g[:8]]

    # fairness glance: bad rate and model flag rate by sensitive groups
    p_best = cross_val_predict(Pipeline([("pre", pre), ("clf", models[best])]), df, y, cv=skf,
                               method="predict_proba")[:, 1]
    flag = p_best >= thr_opt
    grp = {
        "age <= 25": df["age"] <= 25, "age > 25": df["age"] > 25,
        "foreign worker = yes": df["foreign_worker"] == "A201", "foreign worker = no": df["foreign_worker"] == "A202",
        "female (A92/A95)": df["personal_status_sex"].isin(["A92", "A95"]),
        "male (A91/A93/A94)": df["personal_status_sex"].isin(["A91", "A93", "A94"]),
    }
    R["fairness"] = [{"group": g, "n": int(m.sum()), "actual_bad_rate": round(float(y[m].mean()), 3),
                      "flagged_bad_rate": round(float(flag[m.values].mean()), 3)} for g, m in grp.items()]

    with open(ROOT / "reports" / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("source:", source)
    print(cv_df.round(3).to_string(index=False))
    print("best (cost):", best, "empirical best thr", t_best, "min cost", min(costs))


if __name__ == "__main__":
    main()
