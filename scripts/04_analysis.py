"""
04_analysis.py
Statistical analysis testing H1–H5 from the Psych-AI research paper.

H1: AI assistants show significantly lower positive valence than human counselors (ESConv supporters)
H2: ESConv users show significantly higher negative affect than AI-conversation users
H3: ESConv supporters show significantly higher empathy marker density than AI assistants
H4: AI assistants display lower emotional variability (entropy) than human participants
H5: ESConv conversations show a more positive valence arc slope than AI-mediated conversations

Tests used:
  - Kruskal-Wallis (omnibus across all groups, non-parametric)
  - Mann-Whitney U (pairwise, non-parametric)
  - Bonferroni correction for multiple comparisons
  - Rank-biserial correlation as effect size (r = 1 - 2U/(n1*n2))

Outputs:
  outputs/tables/descriptive_stats.csv
  outputs/tables/hypothesis_results.csv
  outputs/tables/pairwise_comparisons.csv
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy import stats

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
INPUT  = ROOT / "data/processed/linguistic_features.csv"
TABLES = ROOT / "outputs/tables"
TABLES.mkdir(parents=True, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Effect size for Mann-Whitney U: ranges -1 to +1."""
    return 1 - (2 * u_stat) / (n1 * n2)


def mannwhitney(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, feature: str) -> dict:
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = rank_biserial(u, len(a), len(b))
    return {
        "feature": feature,
        "group_a": label_a,
        "group_b": label_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "U": float(u),
        "p_raw": float(p),
        "effect_r": float(r),
    }


def kruskal(*arrays, feature: str, labels: list[str]) -> dict:
    h, p = stats.kruskal(*arrays)
    return {"feature": feature, "groups": " | ".join(labels), "H": float(h), "p": float(p)}


def group_vec(df: pd.DataFrame, dataset: str, speaker: str, col: str) -> np.ndarray:
    mask = (df["dataset"] == dataset) & (df["speaker"] == speaker)
    return df.loc[mask, col].dropna().values


# ── load data ──────────────────────────────────────────────────────────────────

print(f"Loading {INPUT} ...")
df = pd.read_csv(INPUT, low_memory=False)
print(f"  {len(df):,} rows")

# conversation-level slope is repeated per row; deduplicate for arc-slope tests
conv_df = df.drop_duplicates(subset=["conversation_id"])[
    ["conversation_id", "dataset", "valence_arc_slope_pos", "valence_arc_slope_neg"]
].copy()

# Assign dataset-level group for arc slope (no speaker split — conversation level)
def conv_group(row):
    if row["dataset"] == "esconv":
        return "ESConv"
    elif row["dataset"] == "lmsys":
        return "LMSYS"
    else:
        return "WildChat"
conv_df["group"] = conv_df.apply(conv_group, axis=1)


# ── descriptive stats ──────────────────────────────────────────────────────────

print("Computing descriptive statistics ...")

feat_cols = [
    "valence_pos", "valence_neg", "empathy_markers", "emotion_entropy",
    "self_disclosure", "hedging", "cognitive_style", "question_rate",
]

desc = (
    df.groupby(["dataset", "speaker"])[feat_cols]
    .agg(["count", "mean", "median", "std"])
    .round(4)
)
desc.columns = ["_".join(c) for c in desc.columns]
desc = desc.reset_index()
desc.to_csv(TABLES / "descriptive_stats.csv", index=False)
print(f"  Saved descriptive_stats.csv")


# ── extract key groups ─────────────────────────────────────────────────────────

esconv_supporter = group_vec(df, "esconv",   "supporter", "valence_pos")
esconv_user      = group_vec(df, "esconv",   "user",      "valence_pos")
lmsys_asst       = group_vec(df, "lmsys",    "assistant", "valence_pos")
wild_asst        = group_vec(df, "wildchat", "assistant", "valence_pos")
wild_user        = group_vec(df, "wildchat", "user",      "valence_pos")

esconv_user_neg  = group_vec(df, "esconv",   "user",      "valence_neg")
lmsys_user_neg   = group_vec(df, "lmsys",    "user",      "valence_neg")
wild_user_neg    = group_vec(df, "wildchat", "user",      "valence_neg")

esconv_emp  = group_vec(df, "esconv",   "supporter", "empathy_markers")
lmsys_emp   = group_vec(df, "lmsys",    "assistant", "empathy_markers")
wild_emp    = group_vec(df, "wildchat", "assistant", "empathy_markers")

esconv_ent_sup  = group_vec(df, "esconv",   "supporter", "emotion_entropy")
esconv_ent_usr  = group_vec(df, "esconv",   "user",      "emotion_entropy")
lmsys_ent       = group_vec(df, "lmsys",    "assistant", "emotion_entropy")
wild_ent        = group_vec(df, "wildchat", "assistant", "emotion_entropy")

arc_esconv  = conv_df.loc[conv_df["group"] == "ESConv",   "valence_arc_slope_pos"].values
arc_lmsys   = conv_df.loc[conv_df["group"] == "LMSYS",    "valence_arc_slope_pos"].values
arc_wild    = conv_df.loc[conv_df["group"] == "WildChat", "valence_arc_slope_pos"].values


# ── hypothesis tests ───────────────────────────────────────────────────────────

print("Running hypothesis tests ...")
hyp_rows = []
pair_rows = []


# H1: AI assistants lower positive valence than ESConv supporters
h1_groups = [esconv_supporter, lmsys_asst, wild_asst]
h1_labels = ["ESConv-supporter", "LMSYS-assistant", "WildChat-assistant"]
h1_kw = kruskal(*h1_groups, feature="valence_pos", labels=h1_labels)
h1_kw["hypothesis"] = "H1"
h1_kw["description"] = "AI assistants vs ESConv supporters: positive valence"
hyp_rows.append(h1_kw)

for la, lb, a, b in [
    ("ESConv-supporter", "LMSYS-assistant",    esconv_supporter, lmsys_asst),
    ("ESConv-supporter", "WildChat-assistant",  esconv_supporter, wild_asst),
    ("LMSYS-assistant",  "WildChat-assistant",  lmsys_asst,       wild_asst),
]:
    pair_rows.append(mannwhitney(a, b, la, lb, "valence_pos"))


# H2: ESConv users higher negative affect than WildChat users
# (LMSYS has no user turns — assistant-only dataset)
h2_groups = [esconv_user_neg, wild_user_neg]
h2_labels = ["ESConv-user", "WildChat-user"]
h2_kw = kruskal(*h2_groups, feature="valence_neg", labels=h2_labels)
h2_kw["hypothesis"] = "H2"
h2_kw["description"] = "User negative valence: ESConv vs WildChat (LMSYS has no user turns)"
hyp_rows.append(h2_kw)

pair_rows.append(mannwhitney(esconv_user_neg, wild_user_neg, "ESConv-user", "WildChat-user", "valence_neg"))


# H3: ESConv supporters higher empathy markers than AI assistants
h3_groups = [esconv_emp, lmsys_emp, wild_emp]
h3_labels = ["ESConv-supporter", "LMSYS-assistant", "WildChat-assistant"]
h3_kw = kruskal(*h3_groups, feature="empathy_markers", labels=h3_labels)
h3_kw["hypothesis"] = "H3"
h3_kw["description"] = "Empathy marker density: human supporters vs AI assistants"
hyp_rows.append(h3_kw)

for la, lb, a, b in [
    ("ESConv-supporter", "LMSYS-assistant",   esconv_emp, lmsys_emp),
    ("ESConv-supporter", "WildChat-assistant", esconv_emp, wild_emp),
]:
    pair_rows.append(mannwhitney(a, b, la, lb, "empathy_markers"))


# H4: AI assistants lower emotion entropy than human participants
h4_groups = [esconv_ent_sup, esconv_ent_usr, lmsys_ent, wild_ent]
h4_labels = ["ESConv-supporter", "ESConv-user", "LMSYS-assistant", "WildChat-assistant"]
h4_kw = kruskal(*h4_groups, feature="emotion_entropy", labels=h4_labels)
h4_kw["hypothesis"] = "H4"
h4_kw["description"] = "Emotion entropy: human vs AI"
hyp_rows.append(h4_kw)

for la, lb, a, b in [
    ("ESConv-supporter", "LMSYS-assistant",   esconv_ent_sup, lmsys_ent),
    ("ESConv-supporter", "WildChat-assistant", esconv_ent_sup, wild_ent),
    ("ESConv-user",      "LMSYS-assistant",   esconv_ent_usr, lmsys_ent),
    ("ESConv-user",      "WildChat-assistant", esconv_ent_usr, wild_ent),
]:
    pair_rows.append(mannwhitney(a, b, la, lb, "emotion_entropy"))


# H5: ESConv conversations show more positive valence arc slope
h5_groups = [arc_esconv, arc_lmsys, arc_wild]
h5_labels = ["ESConv", "LMSYS", "WildChat"]
h5_kw = kruskal(*h5_groups, feature="valence_arc_slope_pos", labels=h5_labels)
h5_kw["hypothesis"] = "H5"
h5_kw["description"] = "Valence arc slope (pos) across datasets"
hyp_rows.append(h5_kw)

for la, lb, a, b in [
    ("ESConv",  "LMSYS",    arc_esconv, arc_lmsys),
    ("ESConv",  "WildChat", arc_esconv, arc_wild),
]:
    pair_rows.append(mannwhitney(a, b, la, lb, "valence_arc_slope_pos"))


# ── Bonferroni correction ──────────────────────────────────────────────────────

pair_df = pd.DataFrame(pair_rows)
n_tests = len(pair_df)
pair_df["p_bonferroni"] = (pair_df["p_raw"] * n_tests).clip(upper=1.0)
pair_df["significant_bonferroni"] = pair_df["p_bonferroni"] < 0.05
pair_df["effect_magnitude"] = pair_df["effect_r"].abs().apply(
    lambda r: "large" if r >= 0.5 else ("medium" if r >= 0.3 else "small")
)

hyp_df = pd.DataFrame(hyp_rows)[
    ["hypothesis", "description", "feature", "groups", "H", "p"]
]
hyp_df["significant"] = hyp_df["p"] < 0.05

hyp_df.to_csv(TABLES / "hypothesis_results.csv", index=False)
pair_df.to_csv(TABLES / "pairwise_comparisons.csv", index=False)
print(f"  Saved hypothesis_results.csv")
print(f"  Saved pairwise_comparisons.csv ({n_tests} pairwise tests, Bonferroni corrected)")


# ── print summary ──────────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("HYPOTHESIS RESULTS")
print("═" * 70)
for _, row in hyp_df.iterrows():
    sig = "✓ SUPPORTED" if row["significant"] else "✗ NOT SUPPORTED"
    print(f"\n{row['hypothesis']}: {row['description']}")
    print(f"  Kruskal-Wallis H={row['H']:.2f}, p={row['p']:.2e}  → {sig}")

print("\n" + "═" * 70)
print("PAIRWISE COMPARISONS (Bonferroni corrected)")
print("═" * 70)
for feat, grp in pair_df.groupby("feature"):
    print(f"\n[{feat}]")
    for _, r in grp.iterrows():
        sig = "* " if r["significant_bonferroni"] else "  "
        print(
            f"  {sig}{r['group_a']} vs {r['group_b']}: "
            f"mean {r['mean_a']:.4f} vs {r['mean_b']:.4f}, "
            f"r={r['effect_r']:.3f} ({r['effect_magnitude']}), "
            f"p_adj={r['p_bonferroni']:.2e}"
        )

print("\nDone. Tables saved to outputs/tables/")
