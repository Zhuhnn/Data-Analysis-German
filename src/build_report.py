"""Build the PDF report from reports/results.json and reports/figures/*.png.

Run after analysis.py:  python src/build_report.py
"""
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "German_Credit_Data_Mining_Report.pdf"
R = json.load(open(ROOT / "reports" / "results.json"))

ACCENT = colors.HexColor("#1c5cab")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#d9d8d3")
ZEBRA = colors.HexColor("#f4f3ef")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, textColor=ACCENT, spaceBefore=6, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.black, spaceBefore=8, spaceAfter=4, keepWithNext=1)
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13.5, alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10.5, textColor=INK2)
CELL = ParagraphStyle("C", parent=BODY, fontSize=7.8, leading=9.6, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=12, bulletIndent=2)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=7.8, leading=9.8, backColor=ZEBRA,
                      borderPadding=6, leftIndent=4, rightIndent=4, spaceBefore=4, spaceAfter=8)
CAP = ParagraphStyle("CAP", parent=SMALL, spaceBefore=2, spaceAfter=10)

story = []
P = lambda t, s=BODY: story.append(Paragraph(t, s))  # noqa: E731


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))


def fig(name, width_cm, caption):
    path = FIG / name
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    width = width_cm * cm
    story.append(KeepTogether([Image(str(path), width=width, height=width * h / w),
                               Paragraph(caption, CAP)]))


def table(rows, widths, header=True, font=7.8):
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
          ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for i in range(2, len(rows), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 6))


def fmt_p(p):
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(2 * cm, 1.2 * cm, "Data Mining Report - Statlog (German Credit Data), UCI id 144")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


cv = R["cv_results"]
test = R["test_results"]
best = R["best_model"]
cv_by = {r["model"]: r for r in cv}
test_by = {(r["model"], r["threshold"]): r for r in test}
top_auc = cv[0]
bl = R["baseline"]
biv = R["bivariate"]
local = R["source"].startswith("local")

# ---------------------------------------------------------------- Title
story.append(Spacer(1, 2.2 * cm))
P("Data Mining Analysis of the<br/>Statlog (German Credit Data) Dataset",
  ParagraphStyle("T", parent=H1, fontSize=22, leading=27, textColor=colors.black))
P("Using the <font face='Courier'>ucimlrepo</font> Python package (UCI Machine Learning Repository, dataset id 144)",
  ParagraphStyle("ST", parent=BODY, fontSize=11.5, textColor=INK2, spaceAfter=18))
P(f"Report date: {date.today().strftime('%d %B %Y')}", SMALL)
story.append(Spacer(1, 0.6 * cm))

P("Executive summary", H2)
chk = R["bad_rate_tables"]["checking_status"]
chk_rate = {d["cat"]: d["mean"] for d in chk}
bullets([
    f"<b>Dataset.</b> 1,000 loan applicants, 20 attributes (7 numeric, 13 categorical), binary target "
    f"<i>good</i> ({R['class_counts']['good']}) vs <i>bad</i> ({R['class_counts']['bad']}) credit risk. "
    f"There are no missing values and no duplicate rows. The classes are imbalanced 70:30.",
    f"<b>Strongest single driver: checking-account status.</b> The bad-credit rate is "
    f"{chk_rate['< 0 DM']*100:.0f}% for an overdrawn account (&lt; 0 DM) and {chk_rate['no checking acct']*100:.0f}% "
    f"for applicants with no checking account at this bank (Cramer's V = {biv[0]['effect']:.2f}). "
    "Credit history, loan duration, savings and property follow.",
    f"<b>Longer, larger loans are riskier.</b> The median duration is 24 months for bad loans vs 18 months for good loans. "
    "Loans over 36 months combined with savings under 100 DM default 72% of the time (association rule, lift 2.4).",
    f"<b>Predictive models.</b> Seven classifiers were compared with stratified 10-fold cross-validation. "
    f"{top_auc['model']}, SVM and logistic regression lead with a ROC-AUC of about 0.79-0.80. Accuracy at the "
    f"default 0.5 threshold is about 75-77%, only modestly above the 70% obtained by labelling everyone \"good\".",
    f"<b>The cost matrix changes the decision rule.</b> Because a missed bad customer costs 5 and a rejected good customer "
    f"costs 1, the cost-optimal threshold is P(bad) &gt; 1/6, not 0.5. This cuts cross-validated cost by "
    f"roughly 40-50% for most models (e.g. {best}: {cv_by[best]['cost_at_0.5']} to {cv_by[best]['cost_at_1/6']}). It also beats the "
    f"trivial \"reject everyone\" policy (cost {bl['all_bad_cost_full']}).",
    "<b>Fairness.</b> Age, sex/marital status and foreign-worker status are linked to the target. "
    "Any operational model should be audited for disparate impact before use (Section 9).",
])
story.append(PageBreak())

# ---------------------------------------------------------------- 1. Method
P("1. Methodology and data acquisition with ucimlrepo", H1)
P("The study follows the CRISP-DM data-mining process: business understanding, then data acquisition and "
  "understanding, pre-processing, modelling (classification, clustering, association rules), evaluation "
  "and interpretation. The data are loaded with the <font face='Courier'>ucimlrepo</font> package, following "
  "the usage instructions in its README (version 0.0.7):")
story.append(Preformatted(
    "pip install -U ucimlrepo\n\n"
    "from ucimlrepo import fetch_ucirepo, list_available_datasets\n\n"
    "# import dataset (Statlog German Credit Data = id 144)\n"
    "statlog_german_credit_data = fetch_ucirepo(id=144)\n"
    "# alternatively: fetch_ucirepo(name='Statlog (German Credit Data)')\n\n"
    "# access data\n"
    "X = statlog_german_credit_data.data.features   # 20 feature columns\n"
    "y = statlog_german_credit_data.data.targets    # 'class' column (1 = good, 2 = bad)\n\n"
    "# access metadata and variable information\n"
    "print(statlog_german_credit_data.metadata.uci_id)\n"
    "print(statlog_german_credit_data.metadata.num_instances)\n"
    "print(statlog_german_credit_data.metadata.additional_info.summary)\n"
    "print(statlog_german_credit_data.variables)", CODE))
if local:
    P("<b>Note on data source.</b> The analysis code calls <font face='Courier'>fetch_ucirepo(id=144)</font> first. "
      "The environment that produced this report had no access to archive.ics.uci.edu, so the call raised "
      "<i>ConnectionError</i>. The code then rebuilt the same <font face='Courier'>dotdict</font> object "
      "(<font face='Courier'>data.features</font>, <font face='Courier'>data.targets</font>, "
      "<font face='Courier'>data.original</font>, <font face='Courier'>metadata</font>, "
      "<font face='Courier'>variables</font>) from the supplied <i>statlog+german+credit+data.zip</i> "
      "(file <i>german.data</i>). It used the same column names (Attribute1-Attribute20, class) as the UCI API. "
      "The data are therefore identical, and with internet access the script uses the API automatically.", SMALL)

P("Business objective", H2)
P("A bank wants to decide, from the information on a loan application, whether the applicant is a good or a bad "
  "credit risk. The dataset includes a <b>cost matrix</b>: classifying a bad customer as good costs 5, and "
  "classifying a good customer as bad costs 1. The data-mining goals are to (i) describe which applicant "
  "characteristics are associated with bad credit, (ii) build and evaluate predictive models under this cost "
  "structure, and (iii) find natural customer segments and frequent risk patterns.")

# ---------------------------------------------------------------- 2. Metadata
P("2. Dataset metadata and variables", H1)
m = R["metadata"]
table([
    ["Field", "Value"],
    ["uci_id / name", f"{m['uci_id']} / {m['name']}"],
    ["repository_url", m.get("repository_url") or ""],
    ["dataset_doi", m.get("dataset_doi") or ""],
    ["creators", ", ".join(m.get("creators") or [])],
    ["task / characteristics", f"{m.get('task')} / {m.get('characteristics')}"],
    ["num_instances / num_features", f"{m['num_instances']} / {m['num_features']}"],
    ["feature_types", ", ".join(m.get("feature_types") or [])],
    ["target_col", ", ".join(m.get("target_col") or [])],
    ["has_missing_values", m.get("has_missing_values")],
    ["summary", m.get("summary") or ""],
], [4.2, 12.8])

P("The <font face='Courier'>variables</font> table (name, role, type, demographic, description, units) "
  "is shown below with the short names used in the rest of this report.", BODY)
from load_data import SHORT_NAMES  # noqa: E402
rows = [["UCI name", "Short name", "Role", "Type", "Demographic", "Description", "Units"]]
for v in R["variables"]:
    rows.append([v["name"], SHORT_NAMES.get(v["name"], "target"), v["role"], v["type"],
                 v.get("demographic") or "", v["description"], v.get("units") or ""])
table(rows, [2.0, 3.0, 1.5, 2.1, 2.2, 4.8, 1.4])
story.append(PageBreak())

# ---------------------------------------------------------------- 3. Data understanding
P("3. Data quality and descriptive statistics", H1)
P(f"<b>Shape:</b> {R['shape'][0]} rows x {R['shape'][1]} features. <b>Missing values:</b> {R['missing_total']}. "
  f"<b>Duplicate rows:</b> {R['duplicates']}. All categorical attributes use the A-codes defined in "
  "<i>german.doc</i>. They were decoded to readable labels for the tables and charts. No cleaning or "
  "imputation was required.")
fig("class_balance.png", 9, "Figure 1. Target distribution. A majority-class classifier would score 70% accuracy, so accuracy alone is a weak metric here.")
rows = [["Variable", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max", "Skew"]]
for d in R["numeric_summary"]:
    rows.append([d["variable"], d["mean"], d["std"], d["min"], d["25%"], d["50%"], d["75%"], d["max"], d["skew"]])
table(rows, [3.4, 1.6, 1.6, 1.3, 1.4, 1.5, 1.4, 1.6, 1.3])
P("Credit amount (median 2,320 DM, max 18,424 DM) and duration are strongly right-skewed. Age ranges from 19 to 75 "
  "(median 33). The ordinal attributes take only a few distinct values: installment rate, residence and number of credits range "
  "from 1 to 4, and dependents from 1 to 2.")
fig("numeric_distributions.png", 16.5, "Figure 2. Density of each numeric attribute for good (blue) and bad (orange) applicants. Bad loans shift towards longer durations, higher amounts and younger ages.")
fig("correlation.png", 10, "Figure 3. Spearman rank correlations. Duration and credit amount are strongly correlated (rho = "
    f"{R['spearman_corr']['duration']['credit_amount']:.2f}). Duration has the largest correlation with the bad flag.")

# ---------------------------------------------------------------- 4. Bivariate
P("4. Which attributes are associated with bad credit?", H1)
P("Each attribute was tested against the target. Categorical attributes used a chi-square test of independence, "
  "with Cramer's V as the effect size. Numeric attributes used a Mann-Whitney U test, with the rank-biserial "
  "correlation as the effect size.")
rows = [["#", "Attribute", "Test", "Statistic", "p-value", "Effect size", "Median good / bad"]]
for i, d in enumerate(biv, 1):
    med = f"{d['median_good']:g} / {d['median_bad']:g}" if d["type"] == "numeric" else ""
    rows.append([i, d["variable"], d["test"], d["statistic"], fmt_p(d["p_value"]), d["effect"], med])
table(rows, [1.0, 3.8, 2.8, 2.0, 1.7, 1.7, 3.0])
sig = [d["variable"] for d in biv if d["p_value"] < 0.05]
nsig = [d["variable"] for d in biv if d["p_value"] >= 0.05]
P(f"<b>{len(sig)} of 20</b> attributes are significant at the 5% level. "
  f"<b>Not significant:</b> {', '.join(nsig)}. These carry little standalone information about credit risk.")
fig("association_strength.png", 12.5, "Figure 4. Strength of association (-log10 p) for every attribute.")
fig("bad_rate_by_category.png", 16.5, "Figure 5. Bad-credit rate within each category of the six most associated categorical attributes.")
P("Key observations", H2)
ch = {d["cat"]: d for d in R["bad_rate_tables"]["credit_history"]}
bullets([
    "<b>Checking account:</b> risk falls steadily from an overdrawn account (49% bad) to &gt;= 200 DM (22%). "
    "It is lowest for applicants with no checking account at this bank (12%).",
    f"<b>Credit history behaves counter-intuitively:</b> applicants coded \"critical account / other credits elsewhere\" have "
    f"the lowest bad rate ({ch['critical account']['mean']*100:.0f}%). Those with \"no credits / all paid back\" have the highest "
    f"({ch['no credits / all paid']['mean']*100:.0f}%, n = {ch['no credits / all paid']['count']}). "
    "This probably reflects how the bank selected applicants in the past (the data contain only granted loans), "
    "so it should not be read as a causal effect.",
    "<b>Savings and property act as a buffer:</b> savings &gt;= 1000 DM (12% bad) and real estate ownership (21%) are protective. "
    "No known property raises the bad rate to 44%.",
    "<b>Housing:</b> owners default less (26%) than renters (39%) or people housed for free (41%).",
    "<b>Purpose:</b> education (44%) and new cars (38%) are the riskiest large groups. Used cars (17%) and radio/TV (22%) are the safest.",
])
story.append(PageBreak())

# ---------------------------------------------------------------- 5. Modelling
P("5. Predictive modelling (classification)", H1)
P("<b>Pre-processing.</b> Numeric attributes were standardised (z-scores). Categorical attributes were one-hot encoded, "
  "giving 61 model inputs. All transformations were placed inside a scikit-learn Pipeline, so they are fitted "
  "only on training folds and do not leak information.")
P("<b>Evaluation design.</b> (a) Stratified 10-fold cross-validation on all 1,000 records, using out-of-fold "
  "probabilities. (b) A stratified 70/30 hold-out split (700 train / 300 test). Metrics: accuracy, ROC-AUC, "
  "F1 and recall for the <i>bad</i> class, and the <b>total misclassification cost</b> = 5 x FN + 1 x FP, "
  "where FN is a bad customer predicted good and FP is a good customer predicted bad. Bayes decision theory "
  "says to predict <i>bad</i> when P(bad) &gt; c<sub>FP</sub>/(c<sub>FP</sub>+c<sub>FN</sub>) = 1/6. "
  "Both the default 0.5 threshold and the cost-optimal 1/6 threshold are reported.")
P("<b>Models:</b> logistic regression; decision tree (depth 4, min leaf 20); random forest (500 trees); "
  "gradient boosting (200 stages, depth 3); SVM with RBF kernel; k-nearest neighbours (k = 15); Gaussian naive Bayes.")
rows = [["Model", "Accuracy @0.5", "ROC-AUC (sd)", "F1 bad @0.5", "Recall bad @0.5", "Cost @0.5", "Cost @1/6", "Accuracy @1/6"]]
for r in cv:
    rows.append([r["model"], f"{r['accuracy']:.3f}", f"{r['roc_auc']:.3f} ({r['auc_sd']:.3f})", f"{r['f1_bad']:.3f}",
                 f"{r['recall_bad']:.3f}", r["cost_at_0.5"], r["cost_at_1/6"], f"{r['accuracy_at_1/6']:.3f}"])
rows.append(["Baseline: accept all", f"{bl['all_good_accuracy']:.3f}", "0.500", "0.000", "0.000", bl["all_good_cost_full"], "", ""])
rows.append(["Baseline: reject all", "0.300", "0.500", "", "1.000", bl["all_bad_cost_full"], "", ""])
table(rows, [3.4, 1.8, 2.2, 1.8, 2.0, 1.6, 1.6, 1.9])
P("Table: 10-fold cross-validation results on 1,000 applicants, sorted by ROC-AUC.", CAP)
fig("model_comparison.png", 16.5, "Figure 6. Left: cross-validated ROC-AUC (error bar = sd across folds). Right: total cost at the default and the cost-optimal threshold.")

rows = [["Model", "Thr", "Accuracy", "Prec. bad", "Recall bad", "F1 bad", "AUC", "TN", "FP", "FN", "TP", "Cost"]]
for r in sorted(test, key=lambda r: (-cv_by[r["model"]]["roc_auc"], r["threshold"])):
    rows.append([r["model"], r["threshold"], f"{r['accuracy']:.3f}", f"{r['precision_bad']:.3f}",
                 f"{r['recall_bad']:.3f}", f"{r['f1_bad']:.3f}", f"{r['roc_auc']:.3f}",
                 r["tn"], r["fp"], r["fn"], r["tp"], r["cost"]])
P("Hold-out test set (n = 300: 210 good, 90 bad)", H2)
table(rows, [2.8, 1.0, 1.8, 1.6, 1.4, 1.4, 1.2, 0.9, 0.9, 0.9, 0.9, 1.1])
P(f"Test-set baselines: accept everyone costs {bl['all_good_cost_test']}; reject everyone costs {bl['all_bad_cost_test']}.", SMALL)
fig("roc_curves.png", 11, "Figure 7. ROC curves on the hold-out test set.")
tc = R["threshold_curve"]
fig("cost_threshold.png", 13, f"Figure 8. Cross-validated total cost of the {best} model as a function of the threshold. The empirical minimum "
    f"(cost {tc['min_cost']} at threshold {tc['best_threshold_empirical']:.2f}) lies close to the theoretical 1/6.")
P("Interpretation of the modelling results", H2)
lr_t = test_by[("Logistic Regression", "1/6")]
rf_t = test_by[("Random Forest", "1/6")]
bullets([
    "Ensembles (random forest), the RBF-SVM and plain logistic regression perform equally well (CV AUC 0.785-0.797, "
    "overlapping fold-to-fold spreads). The simple, transparent logistic model gives up almost nothing in accuracy.",
    "The single decision tree, naive Bayes and k-NN are clearly weaker. One-hot encoded categorical data suits "
    "distance-based and independence-assuming learners poorly.",
    "At the default 0.5 threshold every model misses more than half of the bad customers (recall 0.25-0.47, except naive Bayes). "
    "Every model except naive Bayes then costs <i>more</i> than simply rejecting everyone (700).",
    f"At the cost-optimal threshold of 1/6, recall for bad customers rises to 0.82-0.96 (naive Bayes 0.74). On the test set, logistic regression "
    f"catches {lr_t['tp']} of 90 bad applicants (cost {lr_t['cost']}) and random forest catches {rf_t['tp']} of 90 (cost {rf_t['cost']}). "
    f"That is less than {bl['all_bad_cost_test']} for rejecting everyone and {bl['all_good_cost_test']} for accepting everyone. "
    "The price is that roughly 40-65% of good applicants are declined. The bank should decide explicitly whether the 5:1 cost ratio reflects its real economics.",
])
story.append(PageBreak())

# ---------------------------------------------------------------- 6. Interpretation
P("6. Model interpretation: which features drive the predictions?", H1)
fig("feature_importance.png", 12.5, "Figure 9. Permutation importance (random forest, hold-out test set, 20 repeats): the drop in ROC-AUC when one attribute is shuffled.")
pi = R["permutation_importance_rf"]
P(f"Checking-account status dominates the random forest (AUC drop {pi[0]['mean']:.3f}). It is followed by "
  f"{pi[1]['feature']}, {pi[2]['feature']}, {pi[3]['feature']}, {pi[4]['feature']} and {pi[5]['feature']}. "
  "This agrees closely with the univariate tests in Section 4.")
fig("logit_coefficients.png", 14, "Figure 10. Largest logistic-regression coefficients (log-odds of being bad). Orange terms raise risk; blue terms lower it.")
rows = [["Term", "Coefficient", "Odds ratio"]]
for d in R["logit_top_terms"][:12]:
    rows.append([d["term"], f"{d['coef']:+.3f}", f"{d['odds_ratio']:.2f}"])
table(rows, [8.5, 2.5, 2.5])
P("Holding the other attributes constant, the odds of bad credit are about 2.7x higher for education loans. "
  "They are about 2x higher for applicants with no known property or savings under 100 DM, and about 1.9x higher "
  "with an overdrawn checking account. Having no checking account at this bank, a \"critical\" credit history "
  "(credits elsewhere), 4-7 years of employment or a guarantor all lower the odds.")
P("The shallow decision tree (depth 4) makes its first split on checking-account status. It then splits on "
  "duration, savings and property, matching the variables identified above.", BODY)

# ---------------------------------------------------------------- 7. Clustering
P("7. Customer segmentation (k-means clustering)", H1)
sil = R["silhouette"]
P("Unsupervised k-means was run on the same standardised, one-hot encoded data, without using the target. "
  "Silhouette scores for k = 2..7 were " + ", ".join(f"k={k}: {v:.3f}" for k, v in sil.items()) +
  ". These low values (&lt; 0.15) show there are <b>no strongly separated natural groups</b>: applicants form a "
  "continuum. k = 4 was kept because it gives segments that are easy to interpret and differ clearly in risk.")
rows = [["Cluster", "Size", "Bad rate", "Median age", "Median duration (m)", "Median amount (DM)", "Inst. rate", "Typical purpose", "Typical checking"]]
for d in sorted(R["cluster_profiles"], key=lambda d: d["bad_rate"]):
    rows.append([d["cluster"], d["size"], f"{d['bad_rate']*100:.0f}%", f"{d['age']:g}", f"{d['duration']:g}",
                 f"{d['credit_amount']:,.0f}", f"{d['installment_rate']:g}", d["top_purpose"], d["top_checking_status"]])
table(rows, [1.5, 1.1, 1.4, 1.6, 2.2, 2.2, 1.4, 2.5, 2.5])
fig("clusters_pca.png", 16.5, f"Figure 11. Clusters (left) and true classes (right) projected on the first two principal components "
    f"({R['pca_var'][0]*100:.1f}% + {R['pca_var'][1]*100:.1f}% of variance).")
cp = sorted(R["cluster_profiles"], key=lambda d: d["bad_rate"])
bullets([
    f"<b>Low risk segment (cluster {cp[0]['cluster']}, {cp[0]['bad_rate']*100:.0f}% bad):</b> older applicants (median {cp[0]['age']:g}) "
    "with several existing credits, small-to-medium loans and long residence.",
    f"<b>High risk segment (cluster {cp[-1]['cluster']}, {cp[-1]['bad_rate']*100:.0f}% bad):</b> long (median {cp[-1]['duration']:g} months), "
    f"large (median {cp[-1]['credit_amount']:,.0f} DM) loans. Half of these applicants turn out bad.",
    "The two middle clusters, one dominated by younger applicants, sit near the 30% average.",
    "In the right panel the good and bad classes overlap heavily. This explains why no classifier exceeds an AUC of about 0.80.",
])
story.append(PageBreak())

# ---------------------------------------------------------------- 8. Association rules
P("8. Association-rule mining (Apriori)", H1)
P("Numeric attributes were discretised (duration: &lt;=12, 13-24, 25-36, &gt;36 months; credit amount: terciles; "
  "age: &lt;=25, 26-35, 36-50, &gt;50). Eleven attributes plus the class label were then converted to a "
  "transaction table. Frequent itemsets were mined with Apriori (min support 3% = 30 applicants, up to 4 items). "
  f"Rules were kept if their consequent is <b>class = BAD</b>, confidence &gt;= 50% and lift &gt;= 1.5 "
  f"({R['n_rules_bad']} rules). Redundant rules, whose condition extends a rule that is at least as strong, were removed.")
rows = [["IF (antecedent)", "THEN", "Support", "Count", "Confidence", "Lift"]]
for r in R["assoc_rules_bad"]:
    rows.append([r["if"].replace(" AND ", " AND<br/>"), "BAD", f"{r['support']:.3f}", r["count"], f"{r['confidence']:.2f}", f"{r['lift']:.2f}"])
table(rows, [8.0, 1.4, 1.6, 1.3, 2.0, 1.3])
P("Strongest rules leading to GOOD credit (confidence &gt;= 85%)", H2)
rows = [["IF (antecedent)", "THEN", "Support", "Count", "Confidence", "Lift"]]
for r in R["assoc_rules_good"]:
    rows.append([r["if"].replace(" AND ", " AND<br/>"), "GOOD", f"{r['support']:.3f}", r["count"], f"{r['confidence']:.2f}", f"{r['lift']:.2f}"])
table(rows, [8.0, 1.4, 1.6, 1.3, 2.0, 1.3])
P("The rules tell the same story in a form loan officers can read directly. Bad credit concentrates where an "
  "<b>overdrawn checking account or long duration</b> is combined with <b>low savings, renting or no property</b>; "
  "these groups default at roughly twice the base rate (lift about 2). Almost risk-free groups (97-100% good) "
  "combine <b>no checking account at this bank</b> with real estate, short loans, used-car or radio/TV purposes, "
  "or a critical credit history.")

# ---------------------------------------------------------------- 9. Fairness
P("9. Sensitive attributes and fairness", H1)
P("The UCI variable table flags several demographic attributes: personal status and sex, age, job, employment, "
  "housing and foreign-worker status. Several of these are significantly associated with the target "
  "(age p &lt; 0.001; foreign worker p = 0.016; personal status/sex p = 0.022). The share of applicants flagged "
  f"as bad by the cost-optimal {best} model (threshold 1/6, out-of-fold) differs by group:")
rows = [["Group", "n", "Actual bad rate", "Flagged as bad by model"]]
for d in R["fairness"]:
    rows.append([d["group"], d["n"], f"{d['actual_bad_rate']*100:.1f}%", f"{d['flagged_bad_rate']*100:.1f}%"])
table(rows, [5.5, 1.5, 3.5, 4.5])
P("Applicants aged 25 or under and women are flagged noticeably more often. Part of this follows from their "
  "higher observed bad rates, but the cost-sensitive threshold amplifies the gap. Only 37 applicants are "
  "non-foreign workers, so estimates for that group are unreliable. In most jurisdictions sex and nationality "
  "may not be used for credit decisions. A deployable model should drop or neutralise these attributes and be "
  "re-validated with fairness metrics (demographic parity, equal opportunity).")

# ---------------------------------------------------------------- 10. Conclusions
P("10. Conclusions, recommendations and limitations", H1)
P("Information gathered", H2)
bullets([
    "Data: 1,000 complete records with no missing or duplicate values. 20 attributes, 70% good / 30% bad.",
    "Most informative attributes: checking-account status, credit history, duration, savings, property, purpose and housing. "
    "Residence duration, number of dependents, job and telephone carry almost no signal.",
    "Risk profile of a bad applicant: overdrawn or low-balance checking account, long and large loan, little savings, "
    "no property, renting, often a new car or education purpose, and younger age.",
    "Best predictive performance: ROC-AUC of about 0.80 (random forest, SVM, logistic regression). The data overlap "
    "strongly between classes, which limits what any model can achieve.",
    "Using the cost matrix is essential. Lowering the threshold to P(bad) &gt; 1/6 cuts the expected cost by roughly 40% "
    "compared with the default 0.5 rule.",
    "Segments: k-means finds a low-risk segment of older, established customers (19% bad) and a high-risk segment of large, long-term loans (50% bad).",
])
P("Recommendations", H2)
bullets([
    "Use a <b>cost-sensitive logistic regression</b> or random forest scorecard with a threshold near 1/6. "
    "Logistic regression matches the ensembles and is easier to explain to regulators and customers.",
    "Add manual review, or require collateral or a guarantor, for the high-risk patterns found by association mining: "
    "overdrawn account with low savings, and loans over 36 months with savings under 100 DM.",
    "Remove protected attributes (sex, foreign-worker status) and monitor age effects before deployment.",
    "Re-estimate the 5:1 cost ratio from actual loss and profit data, because the threshold depends directly on it.",
])
P("Limitations", H2)
bullets([
    "Small sample (1,000 records), collected in Germany in the 1970s-1990s in Deutsche Mark. The results may not transfer to current lending.",
    "Only approved loans are observed (reject-inference problem). This explains counter-intuitive effects such as the \"critical account\" history.",
    "The personal-status attribute mixes sex and marital status. One category (single female) has no records.",
    "The hyper-parameters are reasonable defaults rather than exhaustively tuned. Small AUC differences (&lt; 0.01) between the top models are not significant.",
])
P("Reproducibility", H2)
P("All code is in the repository: <font face='Courier'>src/load_data.py</font> (ucimlrepo loading), "
  "<font face='Courier'>src/analysis.py</font> (all computations and figures, random seed 42) and "
  "<font face='Courier'>src/build_report.py</font> (this PDF). Run <font face='Courier'>pip install -r requirements.txt</font>, "
  "then <font face='Courier'>python src/analysis.py &amp;&amp; python src/build_report.py</font>.", BODY)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="German Credit Data Mining Report",
                        author="Data mining analysis", subject="Statlog (German Credit Data), UCI id 144")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
