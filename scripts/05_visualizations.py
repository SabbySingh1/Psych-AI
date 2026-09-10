"""
05_visualizations.py
Publication-quality figures for the Psych-AI paper.

Fig 1: Positive & negative valence by group (grouped bars)
Fig 2: Emotion entropy by group (horizontal bars)
Fig 3: Empathy markers by group (bars)
Fig 4: Effect size forest plot (rank-biserial r, all pairwise tests)
Fig 5: Linguistic feature heatmap (z-scored, groups × features)
Fig 6: Valence arc slope by dataset (bars)

Palette: reference categorical slots (light mode)
  blue   #2a78d6  ESConv-supporter
  orange #eb6834  ESConv-user
  aqua   #1baf7a  LMSYS-assistant
  yellow #eda100  WildChat-assistant
  pink   #e87ba4  WildChat-user
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
LF      = ROOT / "data/processed/linguistic_features.csv"
PAIRS   = ROOT / "outputs/tables/pairwise_comparisons.csv"
FIGURES = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── palette (reference categorical, light mode) ────────────────────────────────
PAL = {
    "ESConv-supporter":    "#2a78d6",
    "ESConv-user":         "#eb6834",
    "LMSYS-assistant":     "#1baf7a",
    "WildChat-assistant":  "#eda100",
    "WildChat-user":       "#e87ba4",
}
GROUPS = list(PAL.keys())
COLORS = [PAL[g] for g in GROUPS]

# chart chrome
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_MUT   = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

FONT = "DejaVu Sans"
mpl.rcParams.update({
    "font.family":        FONT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": True,
    "axes.edgecolor":     BASELINE,
    "axes.facecolor":     SURFACE,
    "figure.facecolor":   SURFACE,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.color":         GRID,
    "grid.linewidth":     0.6,
    "xtick.color":        INK_MUT,
    "ytick.color":        INK_MUT,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.labelsize":     10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    INK,
    "axes.labelcolor":    INK,
    "text.color":         INK,
    "legend.fontsize":    9,
    "legend.frameon":     False,
})

def save(fig, name: str, dpi: int = 300):
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {path.name}")
    plt.close(fig)


def group_mean_sem(df, feature, group_col="group"):
    result = {}
    for g in GROUPS:
        vals = df.loc[df[group_col] == g, feature].dropna().values
        result[g] = (np.mean(vals), stats.sem(vals) if len(vals) > 1 else 0)
    return result


# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(LF, low_memory=False)
df["group"] = df["dataset"].str.cat(df["speaker"], sep="-").str.replace(
    "esconv", "ESConv").str.replace("wildchat", "WildChat").str.replace(
    "lmsys", "LMSYS").str.replace("-supporter", "-supporter").str.replace(
    "-assistant", "-assistant").str.replace("-user", "-user")

# normalize group labels
remap = {
    "ESConv-supporter": "ESConv-supporter",
    "ESConv-user":      "ESConv-user",
    "LMSYS-assistant":  "LMSYS-assistant",
    "WildChat-assistant": "WildChat-assistant",
    "WildChat-user":    "WildChat-user",
}
df["group"] = df["group"].map(remap)

pairs_df = pd.read_csv(PAIRS)
print(f"  {len(df):,} turns across {df['group'].nunique()} groups")

# conversation-level for arc slopes
conv_df = df.drop_duplicates("conversation_id")[
    ["conversation_id", "dataset", "valence_arc_slope_pos"]
].copy()
conv_df["dataset_label"] = conv_df["dataset"].map(
    {"esconv": "ESConv", "lmsys": "LMSYS", "wildchat": "WildChat"}
)


# ── Fig 1: Positive & Negative Valence ────────────────────────────────────────
print("Fig 1: Valence comparison ...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
fig.suptitle("Emotional Valence by Group", fontsize=13, fontweight="bold", color=INK, y=1.01)

for ax, feat, title, ylabel in [
    (axes[0], "valence_pos", "Positive Valence", "Mean Positive Valence"),
    (axes[1], "valence_neg", "Negative Valence", "Mean Negative Valence"),
]:
    ms = group_mean_sem(df, feat)
    means = [ms[g][0] for g in GROUPS]
    sems  = [ms[g][1] for g in GROUPS]
    x = np.arange(len(GROUPS))
    bars = ax.bar(x, means, color=COLORS, width=0.6, yerr=sems,
                  error_kw={"elinewidth": 1.2, "ecolor": INK_MUT, "capsize": 3},
                  linewidth=0)
    # 4px rounded look via corner radius
    for bar in bars:
        bar.set_linewidth(0)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_axisbelow(True)
    # direct value labels
    for xi, (m, s) in zip(x, zip(means, sems)):
        ax.text(xi, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom",
                fontsize=7.5, color=INK_MUT)

fig.tight_layout()
save(fig, "fig1_valence")


# ── Fig 2: Emotion Entropy ─────────────────────────────────────────────────────
print("Fig 2: Emotion entropy ...")
ms = group_mean_sem(df, "emotion_entropy")
means = [ms[g][0] for g in GROUPS]
sems  = [ms[g][1] for g in GROUPS]

fig, ax = plt.subplots(figsize=(7, 4))
y = np.arange(len(GROUPS))
hbars = ax.barh(y, means, color=COLORS, height=0.55,
                xerr=sems, error_kw={"elinewidth": 1.2, "ecolor": INK_MUT, "capsize": 3},
                linewidth=0)
ax.set_yticks(y)
ax.set_yticklabels(GROUPS[::-1][::-1], fontsize=9)  # keep order
ax.set_xlabel("Mean Shannon Entropy (bits)")
ax.set_title("Emotion Entropy by Group\n(higher = richer emotional range)", pad=10)
ax.grid(axis="x")
ax.grid(axis="y", linewidth=0)
# value labels
for yi, (m, s) in zip(y, zip(means, sems)):
    ax.text(m + s + 0.01, yi, f"{m:.3f}", va="center", fontsize=8, color=INK_MUT)
ax.set_xlim(0, max(means) * 1.25)
fig.tight_layout()
save(fig, "fig2_entropy")


# ── Fig 3: Linguistic Features Panel ─────────────────────────────────────────
print("Fig 3: Linguistic features panel ...")
feat_info = [
    ("self_disclosure",  "Self-Disclosure\n(1st-person density)"),
    ("hedging",          "Hedging\n(uncertainty keywords)"),
    ("empathy_markers",  "Empathy Markers\n(keyword density)"),
    ("cognitive_style",  "Cognitive/Analytical\n(function words)"),
    ("question_rate",    "Question Rate\n(? per sentence)"),
]

fig, axes = plt.subplots(1, 5, figsize=(15, 4.5))
fig.suptitle("Linguistic Features by Group", fontsize=13, fontweight="bold", color=INK, y=1.01)

for ax, (feat, label) in zip(axes, feat_info):
    ms = group_mean_sem(df, feat)
    means = [ms[g][0] for g in GROUPS]
    sems  = [ms[g][1] for g in GROUPS]
    x = np.arange(len(GROUPS))
    ax.bar(x, means, color=COLORS, width=0.6, yerr=sems,
           error_kw={"elinewidth": 1.0, "ecolor": INK_MUT, "capsize": 2},
           linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels(["ES-sup", "ES-usr", "LM-ast", "WC-ast", "WC-usr"],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_title(label, fontsize=9)
    ax.spines["bottom"].set_color(BASELINE)

fig.tight_layout()
save(fig, "fig3_linguistic_features")


# ── Fig 4: Effect Size Forest Plot ────────────────────────────────────────────
print("Fig 4: Effect size forest plot ...")
p = pairs_df.copy()
p["label"] = p["group_a"].str.replace("-", "\n", 1) + " vs\n" + \
             p["group_b"].str.replace("-", "\n", 1)
p["label"] = p["feature"] + ": " + p["group_a"] + " vs " + p["group_b"]
p = p.sort_values("effect_r", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(p) * 0.42)))
y = np.arange(len(p))
colors_ef = ["#2a78d6" if sig else GRID for sig in p["significant_bonferroni"]]
ax.barh(y, p["effect_r"].abs(), color=colors_ef, height=0.55, linewidth=0)
ax.axvline(0.5, color="#e34948", linewidth=1.0, linestyle="--", label="Large (r=0.5)")
ax.axvline(0.3, color="#eda100", linewidth=1.0, linestyle="--", label="Medium (r=0.3)")
ax.set_yticks(y)
ax.set_yticklabels(p["label"], fontsize=7.5)
ax.set_xlabel("|Rank-biserial r| (effect size)")
ax.set_title("Effect Sizes — All Pairwise Comparisons\n(blue = significant after Bonferroni correction)",
             pad=10)
ax.set_xlim(0, 1.05)
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x")
ax.grid(axis="y", linewidth=0)
# value labels
for yi, (r, sig) in enumerate(zip(p["effect_r"].abs(), p["significant_bonferroni"])):
    ax.text(r + 0.01, yi, f"{r:.3f}", va="center", fontsize=7.5,
            color=INK if sig else INK_MUT)
fig.tight_layout()
save(fig, "fig4_effect_sizes")


# ── Fig 5: Feature Heatmap ────────────────────────────────────────────────────
print("Fig 5: Feature heatmap ...")
feat_cols = [
    "valence_pos", "valence_neg", "emotion_entropy",
    "empathy_markers", "self_disclosure", "hedging",
    "cognitive_style", "question_rate",
]
feat_labels = [
    "Pos. Valence", "Neg. Valence", "Emotion Entropy",
    "Empathy", "Self-Disclosure", "Hedging",
    "Cognitive Style", "Question Rate",
]

matrix = np.zeros((len(GROUPS), len(feat_cols)))
for i, g in enumerate(GROUPS):
    for j, f in enumerate(feat_cols):
        vals = df.loc[df["group"] == g, f].dropna().values
        matrix[i, j] = np.mean(vals) if len(vals) else np.nan

# z-score each column
z = (matrix - np.nanmean(matrix, axis=0)) / np.nanstd(matrix, axis=0)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
ax.set_xticks(range(len(feat_labels)))
ax.set_xticklabels(feat_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(GROUPS)))
ax.set_yticklabels(GROUPS, fontsize=9)
ax.set_title("Linguistic & Emotional Features — Z-scored Means", pad=10)
cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="z-score")
cbar.ax.tick_params(labelsize=8)
# cell value labels
for i in range(len(GROUPS)):
    for j in range(len(feat_cols)):
        val = z[i, j]
        txt_color = "white" if abs(val) > 1.2 else INK
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=txt_color)
ax.spines[:].set_visible(False)
fig.tight_layout()
save(fig, "fig5_heatmap")


# ── Fig 6: Valence Arc Slope ──────────────────────────────────────────────────
print("Fig 6: Valence arc slope ...")
arc_pal = {"ESConv": "#2a78d6", "LMSYS": "#1baf7a", "WildChat": "#eda100"}
ds_labels = ["ESConv", "LMSYS", "WildChat"]

arc_means, arc_sems = [], []
for ds in ds_labels:
    vals = conv_df.loc[conv_df["dataset_label"] == ds, "valence_arc_slope_pos"].dropna().values
    arc_means.append(np.mean(vals))
    arc_sems.append(stats.sem(vals) if len(vals) > 1 else 0)

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(ds_labels))
bars = ax.bar(x, arc_means, color=[arc_pal[d] for d in ds_labels], width=0.5,
              yerr=arc_sems, error_kw={"elinewidth": 1.2, "ecolor": INK_MUT, "capsize": 4},
              linewidth=0)
ax.axhline(0, color=BASELINE, linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels(ds_labels, fontsize=10)
ax.set_ylabel("Mean Positive Valence Arc Slope")
ax.set_title("Valence Trajectory Across Conversation\n(positive slope = conversations get warmer)", pad=10)
for xi, (m, s) in zip(x, zip(arc_means, arc_sems)):
    ax.text(xi, m + s + 0.001, f"{m:.4f}", ha="center", va="bottom",
            fontsize=8.5, color=INK_MUT)
fig.tight_layout()
save(fig, "fig6_valence_arc")


print(f"\nAll figures saved to {FIGURES}/")
print("Files:", [f.name for f in sorted(FIGURES.glob("*.png"))])
