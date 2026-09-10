"""
07_paper_figures.py
Four publication-ready figures for the Psych-AI paper (Computers in Human
Behavior framing): self-disclosure gap, human-vs-AI mirroring, platform
emotional profile, and arc trajectory by turn quintile.

Fig 1: Self-disclosure across ESConv / WildChat / LMSYS / ShareChat platforms
Fig 2: Mirroring — ESConv (human) vs WildChat (AI), side-by-side panels
Fig 3: Platform emotional profile — Cohen's d vs ESConv, horizontal bars
Fig 4: Valence trajectory across turn quintiles, one line per dataset

Palette: reference categorical slots (light mode), same as 05_visualizations.py
Saved as 300 DPI PNG and PDF to outputs/figures/
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
LF       = ROOT / "data/processed/linguistic_features.csv"
SC_LF    = ROOT / "data/processed/sharechat_linguistic_features.csv"
SC_SD    = ROOT / "outputs/tables/sharechat_self_disclosure_by_platform.csv"
SC_ANOVA = ROOT / "outputs/tables/sharechat_platform_anova.csv"
MIRROR   = ROOT / "outputs/tables/mirroring_coefficients.csv"
FIGURES  = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ── palette ─────────────────────────────────────────────────────────────────
DATASET_PAL = {
    "ESConv":    "#2a78d6",
    "WildChat":  "#eda100",
    "LMSYS":     "#1baf7a",
    "ShareChat": "#008300",
}
PLATFORM_PAL = {
    "chatgpt":    "#2a78d6",
    "claude":     "#eb6834",
    "gemini":     "#1baf7a",
    "grok":       "#eda100",
    "perplexity": "#e87ba4",
}

SURFACE, INK, INK_MUT, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
FONT = "DejaVu Sans"
mpl.rcParams.update({
    "font.family": FONT,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.spines.bottom": True, "axes.edgecolor": BASELINE,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": INK_MUT, "ytick.color": INK_MUT,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.labelsize": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "legend.fontsize": 9, "legend.frameon": False,
})


def save(fig, name: str):
    for ext in ("png", "pdf"):
        path = FIGURES / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved {name}.png / {name}.pdf")
    plt.close(fig)


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return 0.0
    return 1.96 * stats.sem(vals)


# ── load data ──────────────────────────────────────────────────────────────
print("Loading data ...")
df = pd.read_csv(LF, low_memory=False)
sc = pd.read_csv(SC_LF, low_memory=False) if SC_LF.exists() else None


# ══════════════════════════════════════════════════════════════════════════
# Fig 1: Self-disclosure — ESConv / WildChat / LMSYS / ShareChat platforms
# ══════════════════════════════════════════════════════════════════════════
print("Fig 1: Self-disclosure by source ...")

labels, means, cis, colors = [], [], [], []

for ds, ds_label in [("esconv", "ESConv\n(human)"), ("wildchat", "WildChat"), ("lmsys", "LMSYS")]:
    vals = df.loc[(df["dataset"] == ds) & (df["speaker"] == "user"), "self_disclosure"].dropna().values
    if len(vals) == 0:
        continue
    labels.append(ds_label)
    means.append(np.mean(vals))
    cis.append(ci95(vals))
    colors.append(DATASET_PAL.get(ds_label.split("\n")[0], "#898781"))

if sc is not None:
    for p_name in ["chatgpt", "claude", "gemini", "grok", "perplexity"]:
        vals = sc.loc[(sc["platform"] == p_name) & (sc["speaker"] == "user"), "self_disclosure"].dropna().values
        labels.append(p_name.capitalize())
        means.append(np.mean(vals))
        cis.append(ci95(vals))
        colors.append(PLATFORM_PAL[p_name])

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
ax.bar(x, means, color=colors, width=0.6, yerr=cis,
       error_kw={"elinewidth": 1.2, "ecolor": INK_MUT, "capsize": 4}, linewidth=0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean Self-Disclosure (1st-person pronoun density)")
ax.set_title("User Self-Disclosure — Human Counseling vs. AI Platforms\n(error bars = 95% CI)", pad=12)
for xi, (m, c) in zip(x, zip(means, cis)):
    ax.text(xi, m + c + 0.0015, f"{m:.4f}", ha="center", va="bottom", fontsize=7.5, color=INK_MUT)
fig.tight_layout()
save(fig, "fig8_self_disclosure_by_source")


# ══════════════════════════════════════════════════════════════════════════
# Fig 2: Mirroring — ESConv (human) vs WildChat (AI), side-by-side panels
# ══════════════════════════════════════════════════════════════════════════
print("Fig 2: Human vs AI mirroring ...")

m = pd.read_csv(MIRROR)
esconv_row  = m[m["label"] == "ESConv (human supporter)"].iloc[0]
wildchat_row = m[m["label"] == "WildChat"].iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
fig.suptitle("Emotional Mirroring: Human Supporters vs. AI Assistants",
             fontsize=13, fontweight="bold", color=INK, y=1.02)

panel_data = [
    (axes[0], esconv_row,  "ESConv (Human Supporter)", "#2a78d6"),
    (axes[1], wildchat_row, "WildChat (AI Assistant)",  "#eda100"),
]
for ax, row, title, color in panel_data:
    vals = [row["r_user_neg_to_ai_pos"], row["r_user_neg_to_ai_neg"]]
    xlabels = ["→ AI positivity", "→ AI empathy"]
    x = np.arange(2)
    bar_colors = [color, color]
    ax.bar(x, vals, color=bar_colors, width=0.5, linewidth=0)
    ax.axhline(0, color=INK_MUT, linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    for xi, v in zip(x, vals):
        va = "bottom" if v >= 0 else "top"
        off = 0.01 if v >= 0 else -0.01
        ax.text(xi, v + off, f"{v:.3f}", ha="center", va=va, fontsize=9, color=INK)

axes[0].set_ylabel("Pearson r (user distress at N)")
fig.tight_layout()
save(fig, "fig9_mirroring_human_vs_ai")


# ══════════════════════════════════════════════════════════════════════════
# Fig 3: Platform emotional profile — Cohen's d vs ESConv, horizontal bars
# ══════════════════════════════════════════════════════════════════════════
print("Fig 3: Platform emotional profile ...")

anova = pd.read_csv(SC_ANOVA)
valence_d = anova[anova["feature"] == "valence_pos"][["platform", "cohens_d_vs_esconv"]].copy()
valence_d = valence_d.sort_values("cohens_d_vs_esconv", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 4))
y = np.arange(len(valence_d))
colors = [PLATFORM_PAL[p] for p in valence_d["platform"]]
ax.barh(y, valence_d["cohens_d_vs_esconv"], color=colors, height=0.55, linewidth=0)
ax.axvline(0, color=BASELINE, linewidth=1.2, label="ESConv baseline (d = 0)")
ax.set_yticks(y)
ax.set_yticklabels([p.capitalize() for p in valence_d["platform"]], fontsize=10)
ax.set_xlabel("Cohen's d vs. ESConv (positive valence)")
ax.set_title("Platform Emotional Profile\n(more negative = further from human-supporter warmth)", pad=10)
for yi, d in zip(y, valence_d["cohens_d_vs_esconv"]):
    ax.text(d - 0.03, yi, f"{d:.2f}", va="center", ha="right", fontsize=8.5, color=INK_MUT)
ax.set_xlim(valence_d["cohens_d_vs_esconv"].min() * 1.28, 0.05)
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(axis="y", linewidth=0)
fig.tight_layout()
save(fig, "fig10_platform_profile")


# ══════════════════════════════════════════════════════════════════════════
# Fig 4: Valence trajectory across turn quintiles — one line per dataset
# ══════════════════════════════════════════════════════════════════════════
print("Fig 4: Arc trajectory by turn quintile ...")

def quintile_trajectory(sub_df: pd.DataFrame) -> pd.DataFrame:
    sub_df = sub_df.copy()
    rows = []
    for conv_id, g in sub_df.groupby("conversation_id"):
        g = g.sort_values("turn_number")
        n = len(g)
        if n < 2:
            continue
        quintile = np.minimum((np.arange(n) * 5 // n), 4)
        g = g.assign(quintile=quintile)
        rows.append(g[["quintile", "valence_pos"]])
    if not rows:
        return pd.DataFrame(columns=["quintile", "valence_pos"])
    return pd.concat(rows, ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 5))

for ds, label in [("esconv", "ESConv"), ("wildchat", "WildChat"), ("lmsys", "LMSYS")]:
    sub = df[df["dataset"] == ds]
    traj = quintile_trajectory(sub)
    if traj.empty:
        continue
    means = traj.groupby("quintile")["valence_pos"].mean()
    ax.plot(means.index + 1, means.values, marker="o", markersize=6,
            color=DATASET_PAL[label], linewidth=2, label=label)

if sc is not None:
    traj_sc = quintile_trajectory(sc)
    means = traj_sc.groupby("quintile")["valence_pos"].mean()
    ax.plot(means.index + 1, means.values, marker="o", markersize=6,
            color=DATASET_PAL["ShareChat"], linewidth=2, label="ShareChat")

ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xlabel("Conversation Quintile (1 = start, 5 = end)")
ax.set_ylabel("Mean Positive Valence")
ax.set_title("Valence Trajectory Across Conversation Progress", pad=10)
ax.legend(loc="best", fontsize=9)
fig.tight_layout()
save(fig, "fig11_arc_trajectory")


print(f"\nAll paper figures saved to {FIGURES}/")
print("Files:", sorted(f.name for f in FIGURES.glob("fig8*") or FIGURES.glob("fig9*")
                        or FIGURES.glob("fig10*") or FIGURES.glob("fig11*")))
print("Files:", sorted(f.name for f in FIGURES.iterdir() if f.stem.startswith(("fig8", "fig9", "fig10", "fig11"))))
