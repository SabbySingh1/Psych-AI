"""
02_emotion_scoring.py
Score every turn in unified_conversations.csv using SamLowe/roberta-base-go_emotions
(27-label GoEmotions) and map to Ekman's 6 basic emotions + neutral.

Outputs:
    data/processed/emotion_scores.csv  — original columns + 27 label scores
                                         + top_emotion, ekman_emotion,
                                           valence_pos, valence_neg

Features:
    - MPS acceleration on Apple Silicon (M5)
    - Batch inference with configurable batch size
    - Checkpoint/resume: saves progress every CHECKPOINT_EVERY rows,
      restarts from last checkpoint so a crash never loses work
    - Truncates text to MAX_TOKENS (128) — covers ~99% of conversational turns;
      longer turns (rare in WildChat/LMSYS) are truncated gracefully

Run:
    source ~/robopsych_env/bin/activate
    python scripts/02_emotion_scoring.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IN_CSV        = PROCESSED_DIR / "unified_conversations.csv"
OUT_CSV       = PROCESSED_DIR / "emotion_scores.csv"
CKPT_CSV      = PROCESSED_DIR / "_emotion_scores_checkpoint.csv"

MODEL_ID       = "SamLowe/roberta-base-go_emotions"
BATCH_SIZE     = 128     # direct forward pass handles larger batches on MPS
MAX_TOKENS     = 128     # truncate — conversational turns avg ~30 tokens
CHECKPOINT_EVERY = 5_000 # rows between checkpoint saves

# ---------------------------------------------------------------------------
# GoEmotions 27 labels (alphabetical — matches model output order)
# ---------------------------------------------------------------------------
GO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "neutral", "optimism", "pride",
    "realization", "relief", "remorse", "sadness", "surprise",
]

# Ekman 6 + neutral mapping (based on GoEmotions paper taxonomy)
EKMAN_MAP = {
    "admiration":    "joy",
    "amusement":     "joy",
    "anger":         "anger",
    "annoyance":     "anger",
    "approval":      "joy",
    "caring":        "joy",
    "confusion":     "surprise",
    "curiosity":     "surprise",
    "desire":        "joy",
    "disappointment":"sadness",
    "disapproval":   "anger",
    "disgust":       "disgust",
    "embarrassment": "sadness",
    "excitement":    "joy",
    "fear":          "fear",
    "gratitude":     "joy",
    "grief":         "sadness",
    "joy":           "joy",
    "love":          "joy",
    "nervousness":   "fear",
    "neutral":       "neutral",
    "optimism":      "joy",
    "pride":         "joy",
    "realization":   "surprise",
    "relief":        "joy",
    "remorse":       "sadness",
    "sadness":       "sadness",
    "surprise":      "surprise",
}

POSITIVE_LABELS = {
    "admiration", "amusement", "approval", "caring", "desire",
    "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief",
}
NEGATIVE_LABELS = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        log.info("Using Apple MPS (Metal) for inference")
        return "mps"
    if torch.cuda.is_available():
        log.info("Using CUDA for inference")
        return "cuda"
    log.info("Using CPU for inference")
    return "cpu"


# ---------------------------------------------------------------------------
# Score a batch of texts → list of dicts {label: score}
# ---------------------------------------------------------------------------
def batch_to_score_dicts(logits: torch.Tensor, id2label: dict) -> list[dict]:
    """Convert raw logits → per-example {label: probability} dicts."""
    probs = F.softmax(logits, dim=-1).cpu().float().numpy()
    out = []
    for row in probs:
        out.append({id2label[i]: float(row[i]) for i in range(len(row))})
    return out


# ---------------------------------------------------------------------------
# Derive top_emotion, ekman_emotion, valence aggregates from a score dict
# ---------------------------------------------------------------------------
def derive_meta(score_dict: dict) -> dict:
    top = max(score_dict, key=score_dict.get)
    valence_pos = sum(score_dict[l] for l in POSITIVE_LABELS)
    valence_neg = sum(score_dict[l] for l in NEGATIVE_LABELS)
    return {
        "top_emotion":   top,
        "ekman_emotion": EKMAN_MAP[top],
        "valence_pos":   round(valence_pos, 6),
        "valence_neg":   round(valence_neg, 6),
    }


# ---------------------------------------------------------------------------
# Load checkpoint (resume support)
# ---------------------------------------------------------------------------
def load_checkpoint() -> pd.DataFrame | None:
    if CKPT_CSV.exists():
        df = pd.read_csv(CKPT_CSV)
        log.info("Resuming from checkpoint: %d rows already scored", len(df))
        return df
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Loading unified dataset from %s", IN_CSV)
    df = pd.read_csv(IN_CSV)
    total = len(df)
    log.info("Total turns to score: %d", total)

    # --- Resume from checkpoint ---
    done = load_checkpoint()
    start_idx = len(done) if done is not None else 0
    scored_rows = done.to_dict("records") if done is not None else []

    if start_idx >= total:
        log.info("All rows already scored. Writing final output.")
        pd.DataFrame(scored_rows).to_csv(OUT_CSV, index=False)
        return

    remaining = df.iloc[start_idx:].reset_index(drop=True)
    log.info("Scoring rows %d → %d", start_idx, total)

    # --- Load model directly (faster than pipeline on MPS) ---
    device_str = get_device()
    device = torch.device(device_str)
    log.info("Loading model: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()
    id2label = model.config.id2label
    log.info("Model loaded on %s\n", device_str)

    # --- Batch inference ---
    texts = remaining["text"].fillna("").tolist()
    n = len(texts)
    since_last_ckpt = 0

    for batch_start in tqdm(range(0, n, BATCH_SIZE), desc="Scoring", unit="batch"):
        batch_texts = texts[batch_start: batch_start + BATCH_SIZE]
        batch_texts = [t if t.strip() else " " for t in batch_texts]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits

        score_dicts = batch_to_score_dicts(logits, id2label)

        for i, score_dict in enumerate(score_dicts):
            meta = derive_meta(score_dict)
            original_row = remaining.iloc[batch_start + i].to_dict()
            scored_rows.append({**original_row, **score_dict, **meta})

        since_last_ckpt += len(batch_texts)

        # Save checkpoint
        if since_last_ckpt >= CHECKPOINT_EVERY:
            pd.DataFrame(scored_rows).to_csv(CKPT_CSV, index=False)
            log.info("Checkpoint saved at %d / %d rows", start_idx + len(scored_rows), total)
            since_last_ckpt = 0

    # --- Final output ---
    result_df = pd.DataFrame(scored_rows)

    # Enforce column order: original schema first, then GO_LABELS, then meta
    original_cols = ["conversation_id", "dataset", "turn_number", "speaker", "text", "model"]
    meta_cols     = ["top_emotion", "ekman_emotion", "valence_pos", "valence_neg"]
    result_df = result_df[original_cols + GO_LABELS + meta_cols]

    result_df.to_csv(OUT_CSV, index=False)

    # Clean up checkpoint now that we have the final file
    if CKPT_CSV.exists():
        CKPT_CSV.unlink()

    # --- Summary ---
    log.info("=" * 60)
    log.info("Emotion scores saved → %s", OUT_CSV)
    log.info("Total rows scored: %d", len(result_df))
    log.info("\nTop emotion distribution (all turns):")
    ekman_counts = result_df["ekman_emotion"].value_counts()
    log.info("\n%s", ekman_counts.to_string())

    log.info("\nTop emotion by dataset:")
    pivot = (
        result_df.groupby(["dataset", "ekman_emotion"])
        .size()
        .unstack(fill_value=0)
    )
    log.info("\n%s", pivot.to_string())

    log.info("\nMean valence by dataset and speaker:")
    valence = (
        result_df.groupby(["dataset", "speaker"])[["valence_pos", "valence_neg"]]
        .mean()
        .round(4)
    )
    log.info("\n%s", valence.to_string())


if __name__ == "__main__":
    main()
