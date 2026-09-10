"""
06_mirroring_analysis.py
Turn-pair emotional mirroring across all four datasets:
  ESConv, LMSYS, WildChat (from emotion_scores.csv)
  ShareChat × 5 platforms      (from sharechat_emotion_scores.csv)

For every consecutive user→assistant turn pair in each dataset/platform:
  - user_valence_neg at turn N
  - assistant_valence_pos / valence_neg at turn N+1

Outputs:
  outputs/tables/mirroring_results.csv      — binned means (distressed/neutral/positive)
  outputs/tables/mirroring_coefficients.csv — Pearson r per dataset and per platform
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
MAIN   = ROOT / "data/processed/emotion_scores.csv"
SC     = ROOT / "data/processed/sharechat_emotion_scores.csv"
TABLES = ROOT / "outputs/tables"
TABLES.mkdir(parents=True, exist_ok=True)


# ── load and combine ───────────────────────────────────────────────────────────

print("Loading datasets ...")
main_df = pd.read_csv(MAIN, low_memory=False)
print(f"  main: {len(main_df):,} rows")

sc_df = pd.read_csv(SC, low_memory=False)
print(f"  ShareChat: {len(sc_df):,} rows")

# tag sources — ShareChat rows get dataset="sharechat" already; split by platform later
combined = pd.concat([main_df, sc_df], ignore_index=True, sort=False)
print(f"  combined: {len(combined):,} rows")


# ── build turn pairs (user at N → assistant at N+1 within same conversation) ──

def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["conversation_id", "turn_number"]).reset_index(drop=True)
    shifted = df.shift(-1)

    mask = (
        (df["speaker"] == "user") &
        (shifted["speaker"].isin(["assistant", "supporter"])) &
        (df["conversation_id"] == shifted["conversation_id"])
    )
    pairs = pd.DataFrame({
        "conversation_id":   df.loc[mask, "conversation_id"].values,
        "dataset":           df.loc[mask, "dataset"].values,
        "platform":          df.loc[mask, "platform"].values if "platform" in df.columns else "—",
        "turn_number":       df.loc[mask, "turn_number"].values,
        "user_speaker":      df.loc[mask, "speaker"].values,
        "user_ekman":        df.loc[mask, "ekman_emotion"].values if "ekman_emotion" in df.columns else None,
        "user_valence_pos":  df.loc[mask, "valence_pos"].values,
        "user_valence_neg":  df.loc[mask, "valence_neg"].values,
        "ai_valence_pos":    shifted.loc[mask, "valence_pos"].values,
        "ai_valence_neg":    shifted.loc[mask, "valence_neg"].values,
        "ai_ekman":          shifted.loc[mask, "ekman_emotion"].values if "ekman_emotion" in shifted.columns else None,
        "ai_empathy":        shifted.loc[mask, "empathy_markers"].values if "empathy_markers" in shifted.columns else np.nan,
    })
    return pairs

print("Building turn pairs ...")
pairs = build_pairs(combined)
print(f"  {len(pairs):,} user→assistant turn pairs")


# ── emotional state bins ───────────────────────────────────────────────────────

def bin_state(row):
    if row["user_valence_neg"] > 0.5:
        return "distressed"
    elif row["user_valence_pos"] > 0.5:
        return "positive"
    else:
        return "neutral"

pairs["user_state"] = pairs.apply(bin_state, axis=1)


# ── per-dataset analysis ───────────────────────────────────────────────────────

DATASETS = [
    ("esconv",    "ESConv (human supporter)",   None),
    ("wildchat",  "WildChat",                    None),
    ("lmsys",     "LMSYS",                      None),
    ("sharechat", "ShareChat (all)",             None),
]
SC_PLATFORMS = ["chatgpt", "claude", "gemini", "grok", "perplexity"]

result_rows  = []   # binned means
coeff_rows   = []   # mirroring coefficients

print("\nComputing mirroring coefficients ...")

def analyze_group(sub: pd.DataFrame, label: str, dataset: str, platform: str = "—"):
    if len(sub) < 10:
        return

    # binned means
    for state in ["distressed", "neutral", "positive"]:
        s = sub[sub["user_state"] == state]
        result_rows.append({
            "dataset":       dataset,
            "platform":      platform,
            "label":         label,
            "user_state":    state,
            "n_pairs":       len(s),
            "user_neg_mean": round(s["user_valence_neg"].mean(), 4) if len(s) else float("nan"),
            "ai_pos_mean":   round(s["ai_valence_pos"].mean(),   4) if len(s) else float("nan"),
            "ai_neg_mean":   round(s["ai_valence_neg"].mean(),   4) if len(s) else float("nan"),
        })

    # ANOVA: does user state predict AI response valence?
    groups = [sub.loc[sub["user_state"] == st, "ai_valence_pos"].dropna().values
              for st in ["distressed", "neutral", "positive"]]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) >= 2:
        f, p_anova = stats.f_oneway(*groups)
    else:
        f, p_anova = float("nan"), float("nan")

    # Pearson r: user_neg at N → ai_valence_pos at N+1  (sycophancy)
    # Pearson r: user_neg at N → ai_valence_neg at N+1  (mirroring/empathy)
    valid = sub[["user_valence_neg", "ai_valence_pos", "ai_valence_neg"]].dropna()
    if len(valid) >= 10:
        r_pos, p_pos = stats.pearsonr(valid["user_valence_neg"], valid["ai_valence_pos"])
        r_neg, p_neg = stats.pearsonr(valid["user_valence_neg"], valid["ai_valence_neg"])
    else:
        r_pos = r_neg = p_pos = p_neg = float("nan")

    coeff_rows.append({
        "dataset":                   dataset,
        "platform":                  platform,
        "label":                     label,
        "n_pairs":                   len(sub),
        "r_user_neg_to_ai_pos":      round(r_pos, 4),
        "p_user_neg_to_ai_pos":      round(p_pos, 4) if not np.isnan(p_pos) else float("nan"),
        "r_user_neg_to_ai_neg":      round(r_neg, 4),
        "p_user_neg_to_ai_neg":      round(p_neg, 4) if not np.isnan(p_neg) else float("nan"),
        "anova_F":                   round(f, 3) if not np.isnan(f) else float("nan"),
        "anova_p":                   round(p_anova, 6) if not np.isnan(p_anova) else float("nan"),
        "interpretation":            (
            "true mirroring (reduces positivity, raises negativity)" if (r_pos < -0.02 and r_neg > 0.05)
            else ("sycophantic (raises positivity despite distress)" if r_pos > 0.05
                  else "flat affect")
        ),
    })

# main datasets
for ds, label, _ in DATASETS:
    sub = pairs[pairs["dataset"] == ds]
    print(f"  {label}: {len(sub):,} pairs")
    if len(sub) < 10:
        # document the null explicitly (e.g. LMSYS has no user turns —
        # assistant-only dataset, structurally excluded from mirroring)
        coeff_rows.append({
            "dataset": ds, "platform": "—", "label": label,
            "n_pairs": len(sub),
            "r_user_neg_to_ai_pos": float("nan"), "p_user_neg_to_ai_pos": float("nan"),
            "r_user_neg_to_ai_neg": float("nan"), "p_user_neg_to_ai_neg": float("nan"),
            "anova_F": float("nan"), "anova_p": float("nan"),
            "interpretation": "N/A — no user turns in this dataset",
        })
    else:
        analyze_group(sub, label, ds)

# ShareChat by platform
for p_name in SC_PLATFORMS:
    sub = pairs[(pairs["dataset"] == "sharechat") & (pairs["platform"] == p_name)]
    print(f"  ShareChat/{p_name}: {len(sub):,} pairs")
    analyze_group(sub, f"ShareChat/{p_name}", "sharechat", p_name)


# ── save outputs ───────────────────────────────────────────────────────────────

results_df = pd.DataFrame(result_rows)
coeffs_df  = pd.DataFrame(coeff_rows)

results_df.to_csv(TABLES / "mirroring_results.csv",      index=False)
coeffs_df.to_csv(TABLES / "mirroring_coefficients.csv",  index=False)
print(f"\nSaved mirroring_results.csv ({len(results_df)} rows)")
print(f"Saved mirroring_coefficients.csv ({len(coeffs_df)} rows)")


# ── print summary ──────────────────────────────────────────────────────────────

print("\n" + "═" * 80)
print("MIRRORING COEFFICIENTS  (r: user distress at N → AI response at N+1)")
print("  r_neg > 0 → AI mirrors distress  |  r_pos > 0 → AI responds with warmth")
print("═" * 80)

disp_cols = ["label", "n_pairs",
             "r_user_neg_to_ai_pos", "r_user_neg_to_ai_neg",
             "anova_F", "anova_p", "interpretation"]

print(coeffs_df[disp_cols].to_string(index=False))

print("\n" + "═" * 80)
print("BINNED MEANS BY USER EMOTIONAL STATE")
print("═" * 80)
pivot = results_df.pivot_table(
    index=["label", "user_state"],
    values=["n_pairs", "ai_pos_mean", "ai_neg_mean"],
    aggfunc="first"
).round(4)
print(pivot.to_string())
