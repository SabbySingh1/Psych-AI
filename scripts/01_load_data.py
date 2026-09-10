"""
01_load_data.py
Load ESConv, WildChat, and LMSYS-Chat-1M from HuggingFace and write a unified
CSV with columns: conversation_id, dataset, turn_number, speaker, text, model

Run:
    source ~/robopsych_env/bin/activate
    python scripts/01_load_data.py
"""

import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = PROCESSED_DIR / "unified_conversations.csv"

ESCONV_SAMPLE = None          # None = all (~1 300 convos)
WILDCHAT_SAMPLE = 50_000
LMSYS_SAMPLE = 50_000

SCHEMA = ["conversation_id", "dataset", "turn_number", "speaker", "text", "model"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stable_id(dataset_name: str, raw_id) -> str:
    """Deterministic conversation ID independent of row order."""
    key = f"{dataset_name}::{raw_id}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def rows_to_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=SCHEMA)
    df["turn_number"] = df["turn_number"].astype(int)
    return df


def save_checkpoint(df: pd.DataFrame, name: str) -> None:
    path = RAW_DIR / f"{name}_raw.csv"
    df.to_csv(path, index=False)
    log.info("Checkpoint saved → %s  (%d rows)", path.name, len(df))


# ---------------------------------------------------------------------------
# ESConv  (counselor / seeker turns)
# ---------------------------------------------------------------------------
def load_esconv() -> pd.DataFrame:
    log.info("── ESConv ──────────────────────────────────────────────────")
    ds = load_dataset("thu-coai/esconv", split="train")
    log.info("Loaded %d ESConv conversations", len(ds))

    import json as _json

    rows = []
    for row_idx, item in enumerate(tqdm(ds, desc="ESConv")):
        # Each row has a single 'text' key containing a JSON-encoded conversation
        raw = item.get("text", "")
        try:
            conv = _json.loads(raw) if isinstance(raw, str) else raw
        except _json.JSONDecodeError:
            log.warning("Row %d: JSON parse failed, skipping", row_idx)
            continue

        conv_id = stable_id("esconv", row_idx)
        dialog = conv.get("dialog", conv.get("conversation", []))

        for turn_idx, turn in enumerate(dialog):
            speaker_raw = turn.get("speaker", turn.get("role", "unknown")).lower()
            # ESConv uses 'sys' (supporter) / 'usr' (seeker)
            if speaker_raw in {"sys", "supporter", "therapist", "counselor"}:
                speaker = "supporter"
            elif speaker_raw in {"usr", "seeker", "user"}:
                speaker = "user"
            else:
                speaker = speaker_raw
            text = turn.get("text", turn.get("content", "")).strip()
            rows.append({
                "conversation_id": conv_id,
                "dataset": "esconv",
                "turn_number": turn_idx,
                "speaker": speaker,
                "text": text,
                "model": "human",   # ESConv is human-human counseling
            })

    df = rows_to_df(rows)
    save_checkpoint(df, "esconv")
    log.info("ESConv → %d turns across %d conversations\n",
             len(df), df["conversation_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# WildChat  (user / assistant turns, model field present)
# ---------------------------------------------------------------------------
def load_wildchat(n: int = WILDCHAT_SAMPLE) -> pd.DataFrame:
    log.info("── WildChat ────────────────────────────────────────────────")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    rows = []
    seen_convs: set[str] = set()

    for item in tqdm(ds, desc="WildChat", total=n):
        if len(seen_convs) >= n:
            break

        conv_raw_id = item.get("conversation_id", item.get("id", id(item)))
        conv_id = stable_id("wildchat", conv_raw_id)
        if conv_id in seen_convs:
            continue
        seen_convs.add(conv_id)

        model = item.get("model", "unknown")
        conversation = item.get("conversation", [])

        for turn_idx, turn in enumerate(conversation):
            role = turn.get("role", "unknown").lower()
            speaker = "user" if role == "user" else "assistant"
            text = turn.get("content", "").strip()
            rows.append({
                "conversation_id": conv_id,
                "dataset": "wildchat",
                "turn_number": turn_idx,
                "speaker": speaker,
                "text": text,
                "model": model,
            })

    df = rows_to_df(rows)
    save_checkpoint(df, "wildchat")
    log.info("WildChat → %d turns across %d conversations\n",
             len(df), df["conversation_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# LMSYS-Chat-1M  (user / assistant turns, model field present)
# ---------------------------------------------------------------------------
def load_lmsys(n: int = LMSYS_SAMPLE) -> pd.DataFrame:
    log.info("── LMSYS-Chat-1M ───────────────────────────────────────────")
    # Requires HF login for gated dataset; uses streaming to avoid 1M download
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    rows = []
    seen_convs: set[str] = set()

    for item in tqdm(ds, desc="LMSYS", total=n):
        if len(seen_convs) >= n:
            break

        conv_raw_id = item.get("conversation_id", item.get("id", id(item)))
        conv_id = stable_id("lmsys", conv_raw_id)
        if conv_id in seen_convs:
            continue
        seen_convs.add(conv_id)

        model = item.get("model", "unknown")
        conversation = item.get("conversation", [])

        for turn_idx, turn in enumerate(conversation):
            role = turn.get("role", "unknown").lower()
            speaker = "user" if role == "human" else "assistant"
            text = turn.get("content", "").strip()
            rows.append({
                "conversation_id": conv_id,
                "dataset": "lmsys",
                "turn_number": turn_idx,
                "speaker": speaker,
                "text": text,
                "model": model,
            })

    df = rows_to_df(rows)
    save_checkpoint(df, "lmsys")
    log.info("LMSYS → %d turns across %d conversations\n",
             len(df), df["conversation_id"].nunique())
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("Starting data load pipeline")
    log.info("Output will be written to: %s\n", OUT_CSV)

    frames = []

    # --- Phase 1: ESConv (fast, validates pipeline) ---
    try:
        frames.append(load_esconv())
    except Exception as exc:
        log.error("ESConv failed: %s", exc, exc_info=True)
        sys.exit(1)

    # --- Phase 2: WildChat ---
    try:
        frames.append(load_wildchat())
    except Exception as exc:
        log.error("WildChat failed: %s", exc, exc_info=True)

    # --- Phase 3: LMSYS ---
    try:
        frames.append(load_lmsys())
    except Exception as exc:
        log.error("LMSYS failed: %s", exc, exc_info=True)
        log.warning("LMSYS requires HF login. Run: huggingface-cli login")

    if not frames:
        log.error("No data loaded. Exiting.")
        sys.exit(1)

    unified = pd.concat(frames, ignore_index=True)
    unified = unified[SCHEMA]  # enforce column order

    # Sanity check
    assert unified["conversation_id"].notna().all(), "Null conversation IDs found"
    assert pd.api.types.is_string_dtype(unified["text"]), "Text column type error"

    unified.to_csv(OUT_CSV, index=False)

    log.info("=" * 60)
    log.info("Unified CSV saved → %s", OUT_CSV)
    log.info("Total rows   : %d", len(unified))
    log.info("Total convos : %d", unified["conversation_id"].nunique())
    log.info("\nBreakdown by dataset:")
    summary = (
        unified.groupby("dataset")
        .agg(conversations=("conversation_id", "nunique"), turns=("text", "count"))
        .reset_index()
    )
    log.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
