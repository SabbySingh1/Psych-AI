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
ROOT     = Path(__file__).resolve().parents[1]
LF       = ROOT / "data/processed/linguistic_features.csv"
PAIRS    = ROOT / "outputs/tables/pairwise_comparisons.csv"
MIRROR   = ROOT / "outputs/tables/mirroring_coefficients.csv"
FIGURES  = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── palette (reference categorical, light mode) ────────────────────────────────
PAL = {
    "ESConv-supporter":    "#2a78d6",
    "ESConv-user":         "#eb6834",
    "LMSYS-assistant":     "#1baf7a",
    "WildChat-assistant":  "#eda100",
    "WildChat-user":       "#e87ba4",
    "ShareChat-assistant": "#008300",
    "ShareChat-user":      "#4a3aa7",
}
GROUPS = list(PAL.keys())
COLORS = [PAL[g] for g in GROUPS]

SC_PLATFORM_PAL = {
    "chatgpt":    "#2a78d6",
    "claude":     "#eb6834",
    "gemini":     "#1baf7a",
    "grok":       "#eda100",
    "perplexity": "#e87ba4",
}

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

remap = {
    "esconv-supporter":   "ESConv-supporter",
    "esconv-user":        "ESConv-user",
    "lmsys-assistant":    "LMSYS-assistant",
    "wildchat-assistant": "WildChat-assistant",
    "wildchat-user":      "WildChat-user",
}
df["group"] = (df["dataset"] + "-" + df["speaker"]).map(remap)

# merge ShareChat, if available
SC_LF = ROOT / "data/processed/sharechat_linguistic_features.csv"
if SC_LF.exists():
    sc_df = pd.read_csv(SC_LF, low_memory=False)
    sc_remap = {"assistant": "ShareChat-assistant", "user": "ShareChat-user"}
    sc_df["group"] = sc_df["speaker"].map(sc_remap)
    common_cols = [c for c in df.columns if c in sc_df.columns]
    df = pd.concat([df[common_cols], sc_df[common_cols]], ignore_index=True)
    print(f"  ShareChat merged: {len(sc_df):,} turns")
else:
    GROUPS[:] = [g for g in GROUPS if not g.startswith("ShareChat")]
    COLORS[:] = [PAL[g] for g in GROUPS]

pairs_df = pd.read_csv(PAIRS)
print(f"  {len(df):,} turns across {df['group'].nunique()} groups")

# conversation-level for arc slopes
conv_df = df.drop_duplicates("conversation_id")[
    ["conversation_id", "dataset", "valence_arc_slope_pos"]
].copy()
conv_df["dataset_label"] = conv_df["dataset"].map(
    {"esconv": "ESConv", "lmsys": "LMSYS", "wildchat": "WildChat", "sharechat": "ShareChat"}
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
    abbrev = {
        "ESConv-supporter": "ES-sup", "ESConv-user": "ES-usr",
        "LMSYS-assistant": "LM-ast",
        "WildChat-assistant": "WC-ast", "WildChat-user": "WC-usr",
        "ShareChat-assistant": "SC-ast", "ShareChat-user": "SC-usr",
    }
    ax.set_xticklabels([abbrev[g] for g in GROUPS],
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
arc_pal = {"ESConv": "#2a78d6", "LMSYS": "#1baf7a", "WildChat": "#eda100", "ShareChat": "#008300"}
ds_labels = [d for d in ["ESConv", "LMSYS", "WildChat", "ShareChat"]
             if d in conv_df["dataset_label"].unique()]

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


# ── Fig 7: Emotional Mirroring Coefficient ────────────────────────────────────
if MIRROR.exists():
    print("Fig 7: Emotional mirroring coefficient ...")
    m = pd.read_csv(MIRROR)

    # x-axis order: ESConv (human benchmark) first, then the 3 AI datasets,
    # then the 5 ShareChat platforms
    order = ["ESConv (human supporter)", "WildChat", "ShareChat (all)",
             "ShareChat/chatgpt", "ShareChat/claude", "ShareChat/gemini",
             "ShareChat/grok", "ShareChat/perplexity"]
    m = m.set_index("label").loc[[o for o in order if o in m["label"].values]].reset_index()

    disp_labels = {
        "ESConv (human supporter)": "ESConv\n(human)",
        "WildChat":                 "WildChat",
        "ShareChat (all)":          "ShareChat\n(all)",
        "ShareChat/chatgpt":        "ChatGPT",
        "ShareChat/claude":         "Claude",
        "ShareChat/gemini":         "Gemini",
        "ShareChat/grok":           "Grok",
        "ShareChat/perplexity":     "Perplexity",
    }
    bar_colors = {
        "ESConv (human supporter)": "#2a78d6",
        "WildChat":                 "#eda100",
        "ShareChat (all)":          "#008300",
        "ShareChat/chatgpt":        SC_PLATFORM_PAL["chatgpt"],
        "ShareChat/claude":         SC_PLATFORM_PAL["claude"],
        "ShareChat/gemini":         SC_PLATFORM_PAL["gemini"],
        "ShareChat/grok":           SC_PLATFORM_PAL["grok"],
        "ShareChat/perplexity":     SC_PLATFORM_PAL["perplexity"],
    }

    x = np.arange(len(m))
    colors = [bar_colors[label] for label in m["label"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(x, m["r_user_neg_to_ai_pos"], color=colors, width=0.6, linewidth=0)
    ax.axhline(0, color=INK_MUT, linewidth=1.2, linestyle="--", zorder=1,
               label="No mirroring (r = 0)")
    ax.set_xticks(x)
    ax.set_xticklabels([disp_labels[l] for l in m["label"]], fontsize=9)
    ax.set_ylabel("Pearson r: user distress (N) → AI positive valence (N+1)")
    ax.set_title(
        "Emotional Mirroring — Does AI Track User Distress?\n"
        "(negative = AI pulls back positivity like a human supporter; "
        "positive = AI grows warmer despite distress)",
        pad=12, fontsize=11
    )
    for xi, r in zip(x, m["r_user_neg_to_ai_pos"]):
        va = "bottom" if r >= 0 else "top"
        offset = 0.008 if r >= 0 else -0.008
        ax.text(xi, r + offset, f"{r:.3f}", ha="center", va=va, fontsize=8, color=INK_MUT)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(axis="x", linewidth=0)
    fig.tight_layout()
    save(fig, "fig7_mirroring")
else:
    print(f"Fig 7 skipped — {MIRROR} not found (run 06_mirroring_analysis.py first)")


print(f"\nAll figures saved to {FIGURES}/")
print("Files:", [f.name for f in sorted(FIGURES.glob("*.png"))])
