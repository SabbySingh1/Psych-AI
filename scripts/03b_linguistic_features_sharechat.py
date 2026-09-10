"""
03b_linguistic_features_sharechat.py
Runs the same linguistic/emotional feature pipeline as 03_linguistic_features.py
on the ShareChat emotion scores, preserving the `platform` column.

Output: data/processed/sharechat_linguistic_features.csv
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
INPUT  = ROOT / "data/processed/sharechat_emotion_scores.csv"
OUTPUT = ROOT / "data/processed/sharechat_linguistic_features.csv"

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


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def self_disclosure(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in FIRST_PERSON) / len(toks)


def hedging(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    t_lower = text.lower()
    return sum(1 for w in HEDGE_WORDS if w in t_lower) / len(toks)


def empathy_markers(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    t_lower = text.lower()
    return sum(1 for w in EMPATHY_WORDS if w in t_lower) / len(toks)


def cognitive_style(text: str) -> float:
    toks = _tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in ANALYTICAL_WORDS) / len(toks)


def question_rate(text: str) -> float:
    sentences = re.split(r"[.!?]+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    q_count = text.count("?")
    return min(q_count / max(len(sentences), 1), 1.0)


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
    return float((x * y).sum() / denom) if denom != 0 else 0.0


def main():
    print(f"Loading {INPUT} ...")
    df = pd.read_csv(INPUT, low_memory=False)
    print(f"  {len(df):,} rows, {df['platform'].nunique()} platforms")

    print("Computing turn-level features ...")
    texts = df["text"].fillna("").astype(str)
    df["self_disclosure"] = texts.apply(self_disclosure)
    df["hedging"]         = texts.apply(hedging)
    df["empathy_markers"] = texts.apply(empathy_markers)
    df["cognitive_style"] = texts.apply(cognitive_style)
    df["question_rate"]   = texts.apply(question_rate)
    df["emotion_entropy"] = df.apply(emotion_entropy, axis=1)
    print("  Turn-level features done.")

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
    df = df.merge(
        pd.concat([slopes_pos, slopes_neg], axis=1).reset_index(),
        on="conversation_id", how="left"
    )
    print("  Valence arc slopes done.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved {len(df):,} rows → {OUTPUT}")

    feat_cols = [
        "self_disclosure", "hedging", "empathy_markers",
        "cognitive_style", "question_rate", "emotion_entropy",
        "valence_pos", "valence_neg",
    ]
    print("\n── Feature summary by platform + speaker ──")
    print(df.groupby(["platform", "speaker"])[feat_cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
