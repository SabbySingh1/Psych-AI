"""
01b_load_sharechat.py
Loads ShareChat (tucnguyen/ShareChat) — English conversations only, 10k turns per
platform (chatgpt, claude, gemini, grok, perplexity) = up to 50k turns total.

Schema matches unified_conversations.csv plus one extra column:
  conversation_id, dataset, platform, turn_number, speaker, text, model
  (dataset = "sharechat" for all rows; platform = chatgpt/claude/etc.)

Output: data/raw/sharechat_conversations.csv
        data/processed/sharechat_unified.csv  (same schema as unified_conversations.csv)
"""

import hashlib
import pandas as pd
from pathlib import Path
from datasets import load_dataset

ROOT      = Path(__file__).resolve().parents[1]
RAW_OUT   = ROOT / "data/raw/sharechat_conversations.csv"
PROC_OUT  = ROOT / "data/processed/sharechat_unified.csv"
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
PROC_OUT.parent.mkdir(parents=True, exist_ok=True)

PLATFORMS     = ["chatgpt", "claude", "gemini", "grok", "perplexity"]
TURNS_PER_PLATFORM = 10_000   # English turns per platform → ~50k total
ROLE_MAP      = {"user": "user", "llm": "assistant", "assistant": "assistant", "system": None}


def conv_id(platform: str, url: str) -> str:
    raw = f"sharechat-{platform}-{url}"
    return hashlib.sha1(raw.encode()).hexdigest()


def load_platform(platform: str, target_turns: int) -> list[dict]:
    print(f"  Loading {platform} ...")
    ds = load_dataset("tucnguyen/ShareChat", platform, streaming=True)
    split = list(ds.keys())[0]

    rows = []
    # Group by URL (each URL = one shared conversation)
    current_url = None
    turn_buf: list[dict] = []
    english_turns = 0

    for item in ds[split]:
        if item.get("detected_language_final") != "English":
            continue
        if item.get("role") not in ROLE_MAP or ROLE_MAP[item["role"]] is None:
            continue

        url = item.get("url", "")
        if url != current_url:
            # flush previous conversation
            if turn_buf:
                rows.extend(turn_buf)
            current_url = url
            turn_buf = []

        cid = conv_id(platform, url)
        turn_number = item.get("message_index", len(turn_buf) + 1)

        turn_buf.append({
            "conversation_id": cid,
            "dataset":         "sharechat",
            "platform":        platform,
            "turn_number":     turn_number,
            "speaker":         ROLE_MAP[item["role"]],
            "text":            (item.get("plain_text") or "").strip(),
            "model":           item.get("model") or platform,
            "topic":           item.get("topic") or "",
            "language":        item.get("detected_language_final") or "",
        })
        english_turns += 1

        if english_turns >= target_turns:
            rows.extend(turn_buf)
            break

    print(f"    {english_turns:,} English turns, {len({r['conversation_id'] for r in rows}):,} conversations")
    return rows


def main():
    print("Loading ShareChat — English only, 10k turns per platform ...")
    all_rows = []
    for p in PLATFORMS:
        all_rows.extend(load_platform(p, TURNS_PER_PLATFORM))

    df = pd.DataFrame(all_rows)
    df = df[df["text"].str.strip().astype(bool)].reset_index(drop=True)

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_OUT, index=False)
    print(f"\nRaw saved: {len(df):,} turns → {RAW_OUT}")

    # unified schema (drop topic/language for the shared pipeline)
    unified = df[["conversation_id", "dataset", "platform", "turn_number", "speaker", "text", "model"]].copy()
    unified.to_csv(PROC_OUT, index=False)
    print(f"Unified saved: {len(unified):,} turns → {PROC_OUT}")

    print("\n── Breakdown ──")
    print(unified.groupby(["platform", "speaker"]).size().to_string())
    print(f"\nTotal conversations: {unified['conversation_id'].nunique():,}")
    print(f"Total turns:         {len(unified):,}")


if __name__ == "__main__":
    main()
