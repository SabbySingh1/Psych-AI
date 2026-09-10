"""
08_crisis_adjacent_analysis.py
Flags user turns containing crisis-adjacent language (hopelessness, giving up,
acute distress — plain-language phrases, NOT explicit self-harm methods) and
measures whether the AI-sycophancy pattern found in 06_mirroring_analysis.py
holds or strengthens in this highest-stakes subset.

This is a conservative keyword flag for aggregate statistical analysis only —
it does not diagnose, does not detect actual crises, and makes no claims
about any individual conversation.

Outputs: outputs/tables/crisis_adjacent_analysis.csv
         outputs/tables/crisis_adjacent_mirroring.csv
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parents[1]
LF     = ROOT / "data/processed/linguistic_features.csv"
SC_LF  = ROOT / "data/processed/sharechat_linguistic_features.csv"
TABLES = ROOT / "outputs/tables"
TABLES.mkdir(parents=True, exist_ok=True)

# ── conservative crisis-adjacent phrase list ───────────────────────────────
# Plain-language hopelessness / giving-up / acute-distress expressions.
# Deliberately excludes explicit self-harm methods or instructional content.
CRISIS_PHRASES = [
    "want to give up", "no point anymore", "no point in anything",
    "can't go on", "cant go on", "not worth living", "life isn't worth",
    "nobody would care", "no one would care", "tired of living",
    "want to disappear", "wish i wasn't here", "wish i weren't here",
    "hate myself", "hate my life", "no reason to live",
    "can't take it anymore", "cant take it anymore",
    "give up on everything", "better off without me", "better off dead",
    "nothing matters anymore", "i give up", "why bother anymore",
    "don't want to be here", "dont want to be here", "end it all",
]


def flag_text(text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = text.lower()
    return any(p in t for p in CRISIS_PHRASES)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled_sd) if pooled_sd > 0 else 0.0


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Same turn-pair construction as 06_mirroring_analysis.py"""
    df = df.sort_values(["conversation_id", "turn_number"]).reset_index(drop=True)
    shifted = df.shift(-1)
    mask = (
        (df["speaker"] == "user") &
        (shifted["speaker"].isin(["assistant", "supporter"])) &
        (df["conversation_id"] == shifted["conversation_id"])
    )
    return pd.DataFrame({
        "conversation_id":  df.loc[mask, "conversation_id"].values,
        "dataset":          df.loc[mask, "dataset"].values,
        "platform":         df.loc[mask, "platform"].values if "platform" in df.columns else "—",
        "user_valence_neg": df.loc[mask, "valence_neg"].values,
        "ai_valence_pos":   shifted.loc[mask, "valence_pos"].values,
        "ai_valence_neg":   shifted.loc[mask, "valence_neg"].values,
    })


# ── load ─────────────────────────────────────────────────────────────────
print("Loading data ...")
main_df = pd.read_csv(LF, low_memory=False)
sc_df   = pd.read_csv(SC_LF, low_memory=False) if SC_LF.exists() else None

combined = pd.concat([main_df, sc_df], ignore_index=True, sort=False) if sc_df is not None else main_df
print(f"  {len(combined):,} total rows")

# ── flag user turns ─────────────────────────────────────────────────────
print("Flagging crisis-adjacent language ...")
user_rows = combined[combined["speaker"] == "user"].copy()
user_rows["crisis_flag"] = user_rows["text"].apply(flag_text)
n_flagged_turns = user_rows["crisis_flag"].sum()
print(f"  {n_flagged_turns:,} user turns flagged out of {len(user_rows):,} ({100*n_flagged_turns/len(user_rows):.3f}%)")

flagged_convs = set(user_rows.loc[user_rows["crisis_flag"], "conversation_id"])
print(f"  {len(flagged_convs):,} unique conversations contain flagged language")

combined["conv_flagged"] = combined["conversation_id"].isin(flagged_convs)


# ── S1: % flagged, AI response quality, conversation length ────────────────
print("\nComputing per-dataset crisis-adjacent statistics ...")

results = []
for ds in combined["dataset"].dropna().unique():
    sub = combined[combined["dataset"] == ds]
    conv_ids = sub["conversation_id"].unique()
    conv_flag_map = sub.drop_duplicates("conversation_id").set_index("conversation_id")["conv_flagged"]
    n_conv = len(conv_ids)
    n_flagged_conv = int(conv_flag_map.sum())
    pct_flagged = 100 * n_flagged_conv / n_conv if n_conv else float("nan")

    ai_rows = sub[sub["speaker"].isin(["assistant", "supporter"])]
    flagged_ai   = ai_rows[ai_rows["conv_flagged"]]
    unflagged_ai = ai_rows[~ai_rows["conv_flagged"]]

    turns_per_conv = sub.groupby("conversation_id")["turn_number"].max()
    flagged_len   = turns_per_conv[turns_per_conv.index.isin(
        conv_flag_map[conv_flag_map].index)]
    unflagged_len = turns_per_conv[turns_per_conv.index.isin(
        conv_flag_map[~conv_flag_map].index)]

    results.append({
        "dataset": ds,
        "n_conversations": n_conv,
        "n_flagged_conversations": n_flagged_conv,
        "pct_flagged": round(pct_flagged, 3),
        "ai_valence_pos_flagged":   round(flagged_ai["valence_pos"].mean(), 4)   if len(flagged_ai) else float("nan"),
        "ai_valence_pos_unflagged": round(unflagged_ai["valence_pos"].mean(), 4) if len(unflagged_ai) else float("nan"),
        "ai_empathy_flagged":       round(flagged_ai["empathy_markers"].mean(), 4)   if len(flagged_ai) else float("nan"),
        "ai_empathy_unflagged":     round(unflagged_ai["empathy_markers"].mean(), 4) if len(unflagged_ai) else float("nan"),
        "mean_turns_flagged":       round(flagged_len.mean(), 2)   if len(flagged_len) else float("nan"),
        "mean_turns_unflagged":     round(unflagged_len.mean(), 2) if len(unflagged_len) else float("nan"),
        "cohens_d_valence_flagged_vs_unflagged": round(
            cohens_d(flagged_ai["valence_pos"].dropna().values,
                     unflagged_ai["valence_pos"].dropna().values), 4
        ),
    })

results_df = pd.DataFrame(results)
results_df.to_csv(TABLES / "crisis_adjacent_analysis.csv", index=False)
print(f"\nSaved crisis_adjacent_analysis.csv")
print(results_df.to_string(index=False))


# ── S2: mirroring coefficient restricted to flagged conversations ──────────
print("\nComputing mirroring coefficients within flagged conversations ...")

pairs = build_pairs(combined)
pairs_flagged = pairs[pairs["conversation_id"].isin(flagged_convs)]

mirror_rows = []
for ds, label in [("esconv", "ESConv (human supporter)"),
                   ("wildchat", "WildChat"),
                   ("lmsys", "LMSYS"),
                   ("sharechat", "ShareChat (all)")]:
    sub = pairs_flagged[pairs_flagged["dataset"] == ds]
    if len(sub) < 5:
        mirror_rows.append({
            "dataset": ds, "label": label, "n_pairs_in_flagged_convs": len(sub),
            "r_user_neg_to_ai_pos": float("nan"), "r_user_neg_to_ai_neg": float("nan"),
            "note": "insufficient flagged-conversation pairs" if len(sub) else "no user turns / no flagged pairs",
        })
        continue
    valid = sub[["user_valence_neg", "ai_valence_pos", "ai_valence_neg"]].dropna()
    r_pos, p_pos = stats.pearsonr(valid["user_valence_neg"], valid["ai_valence_pos"])
    r_neg, p_neg = stats.pearsonr(valid["user_valence_neg"], valid["ai_valence_neg"])
    mirror_rows.append({
        "dataset": ds, "label": label, "n_pairs_in_flagged_convs": len(sub),
        "r_user_neg_to_ai_pos": round(r_pos, 4), "p_user_neg_to_ai_pos": round(p_pos, 5),
        "r_user_neg_to_ai_neg": round(r_neg, 4), "p_user_neg_to_ai_neg": round(p_neg, 5),
        "note": "",
    })

mirror_df = pd.DataFrame(mirror_rows)
mirror_df.to_csv(TABLES / "crisis_adjacent_mirroring.csv", index=False)
print(f"\nSaved crisis_adjacent_mirroring.csv")
print(mirror_df.to_string(index=False))

print(f"\nDone. {len(flagged_convs):,} flagged conversations across all sources.")
