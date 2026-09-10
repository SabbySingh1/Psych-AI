"""
03_linguistic_features.py
Extracts turn-level and conversation-level linguistic features from emotion_scores.csv.

Turn-level features (one row per turn):
  - self_disclosure: 1st-person pronoun density
  - hedging: uncertainty/hedge keyword density
  - empathy_markers: empathy keyword density
  - cognitive_style: analytical function word density
  - question_rate: fraction of sentences ending in '?'

Conversation-level features (aggregated per conversation_id):
  - emotion_entropy: Shannon entropy over GoEmotions distribution (mean across turns)
  - valence_arc_slope: linear slope of valence_pos across turn_number
  - valence_arc_slope_neg: linear slope of valence_neg across turn_number

Output: data/processed/linguistic_features.csv
"""

import re
import math
import numpy as np
import pandas as pd
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/emotion_scores.csv"
OUTPUT = ROOT / "data/processed/linguistic_features.csv"

# ── lexicons ───────────────────────────────────────────────────────────────────
FIRST_PERSON = {"i", "me", "my", "myself", "mine", "i'm", "i've", "i'd", "i'll"}

HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "probably", "might", "could", "should",
    "seem", "seems", "appeared", "appears", "suggest", "suggests", "think",
    "thought", "believe", "believed", "guess", "wonder", "sort of", "kind of",
    "somewhat", "rather", "fairly", "generally", "usually", "often", "sometimes",
    "unclear", "uncertain", "unsure", "not sure", "i'm not", "not certain",
}

EMPATHY_WORDS = {
    "understand", "feel", "feeling", "sorry", "apologize", "support", "here for",
    "care", "listen", "listening", "validate", "acknowledge", "compassion",
    "empathize", "empathy", "concern", "concerned", "worry", "worried",
    "that must", "that sounds", "i hear you", "i see", "makes sense",
    "difficult", "hard", "tough", "struggle", "struggling", "pain", "hurt",
}

ANALYTICAL_WORDS = {
    "because", "therefore", "thus", "hence", "however", "although", "despite",
    "whereas", "furthermore", "moreover", "consequently", "accordingly",
    "analyze", "analysis", "consider", "evaluate", "determine", "conclude",
    "evidence", "reason", "reasoning", "logical", "objective", "systematic",
    "specifically", "particularly", "generally", "typically", "relatively",
    "compared", "contrast", "distinction", "category", "structure", "process",
}

GOEMOTION_COLS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise",
]


# ── feature extractors ─────────────────────────────────────────────────────────

def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def self_disclosure(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in FIRST_PERSON)
    return hits / len(toks)


def hedging(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    t_lower = text.lower()
    hits = sum(1 for w in HEDGE_WORDS if w in t_lower)
    return hits / len(toks)


def empathy_markers(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    t_lower = text.lower()
    hits = sum(1 for w in EMPATHY_WORDS if w in t_lower)
    return hits / len(toks)


def cognitive_style(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in ANALYTICAL_WORDS)
    return hits / len(toks)


def question_rate(text: str) -> float:
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    q_count = sum(1 for s in re.split(r"[.!]+", text) if "?" in s)
    q_count = text.count("?")
    total = max(len(sentences), 1)
    return min(q_count / total, 1.0)


def emotion_entropy(row: pd.Series) -> float:
    probs = row[GOEMOTION_COLS].values.astype(float)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log2(probs)))


def valence_slope(group: pd.DataFrame, col: str) -> float:
    if len(group) < 2:
        return 0.0
    x = group["turn_number"].values.astype(float)
    y = group[col].values.astype(float)
    x -= x.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT} ...")
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # ── turn-level features ────────────────────────────────────────────────────
    print("Computing turn-level features ...")
    texts = df["text"].fillna("").astype(str)

    df["self_disclosure"] = texts.apply(self_disclosure)
    df["hedging"]         = texts.apply(hedging)
    df["empathy_markers"] = texts.apply(empathy_markers)
    df["cognitive_style"] = texts.apply(cognitive_style)
    df["question_rate"]   = texts.apply(question_rate)
    df["emotion_entropy"] = df.apply(emotion_entropy, axis=1)

    print("  Turn-level features done.")

    # ── conversation-level: valence arc slopes ─────────────────────────────────
    print("Computing valence arc slopes per conversation ...")
    slopes_pos = (
        df.groupby("conversation_id")
        .apply(lambda g: valence_slope(g, "valence_pos"), include_groups=False)
        .rename("valence_arc_slope_pos")
    )
    slopes_neg = (
        df.groupby("conversation_id")
        .apply(lambda g: valence_slope(g, "valence_neg"), include_groups=False)
        .rename("valence_arc_slope_neg")
    )
    slopes = pd.concat([slopes_pos, slopes_neg], axis=1).reset_index()
    df = df.merge(slopes, on="conversation_id", how="left")

    print("  Valence arc slopes done.")

    # ── save ───────────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT}")
    print(f"Columns: {list(df.columns)}")

    # ── quick summary ──────────────────────────────────────────────────────────
    feat_cols = [
        "self_disclosure", "hedging", "empathy_markers",
        "cognitive_style", "question_rate", "emotion_entropy",
        "valence_arc_slope_pos", "valence_arc_slope_neg",
    ]
    print("\n── Feature summary by dataset + speaker ──")
    print(df.groupby(["dataset", "speaker"])[feat_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
