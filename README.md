# Psych-AI

A computational psychology research project comparing emotional language patterns across human counseling and AI-generated conversation datasets. The goal is to characterize how emotional expression, empathy, self-disclosure, and sentiment evolve differently in human-to-human support contexts versus human-to-AI chat.

---

## Research Questions

- How do emotion distributions differ between human counselors and AI assistants?
- Do AI models (GPT-4, Claude, Mistral, etc.) mirror user emotional states or respond with consistent affect regardless of user emotion?
- How does self-disclosure and hedging language vary across datasets and speaker roles?
- What does the emotional arc of a conversation look like — and does it differ between ESConv support sessions and open-domain AI chat?

---

## Datasets

| Dataset | Conversations | Turns | Language | Focus |
|---|---|---|---|---|
| [ESConv](https://huggingface.co/datasets/thu-coai/esconv) | 910 | 26,648 | English | Human emotional support counseling |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 50,000 | ~290,000 | English | Real-world GPT-3/4 conversations |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 50,000 | ~202,000 | English | Multi-model chat (gated) |
| [Talk2AI](https://arxiv.org/abs/2604.04354) | 3,080 | 30,800 | Italian | Human-AI persuasion (side comparison) |

**Total primary corpus:** ~100,910 conversations / ~518,962 turns

Talk2AI is used as a secondary cross-linguistic comparison only. Because the primary emotion classifier is English-only, Talk2AI is scored with a separate multilingual model and treated as supplementary evidence rather than a core dataset.

---

## Pipeline

```
01_load_data.py          Pull all datasets from HuggingFace, standardize into
                         unified CSV (conversation_id, dataset, turn_number,
                         speaker, text, model)
        ↓
02_emotion_scoring.py    Score every turn with SamLowe/roberta-base-go_emotions
                         (27 GoEmotions labels → Ekman 6 + neutral, valence scores)
        ↓
03_linguistic_features.py  Extract self-disclosure, hedging, empathy markers,
                            cognitive analytical style, question rate, sentiment arc
        ↓
04_analysis.py           Statistical comparisons across datasets and speaker roles
        ↓
05_visualizations.py     Figures and tables for the paper
```

---

## Emotion Model

**[SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)**

- Fine-tuned RoBERTa-base on Google GoEmotions (Reddit comments)
- 27 emotion labels collapsed to Ekman's 6 basic emotions + neutral
- All turns scored independently; analysis split by speaker role
- MPS-accelerated inference on Apple Silicon

GoEmotions → Ekman mapping used:

| Ekman | GoEmotions labels |
|---|---|
| joy | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| sadness | disappointment, embarrassment, grief, remorse, sadness |
| anger | anger, annoyance, disapproval |
| fear | fear, nervousness |
| disgust | disgust |
| surprise | confusion, curiosity, realization, surprise |
| neutral | neutral |

---

## Linguistic Features (Script 03)

| Feature | Operationalization |
|---|---|
| Self-disclosure intensity | 1st-person pronoun density (I/me/myself / word count) |
| Hedging / uncertainty | Epistemic hedge keyword density |
| Empathy markers | ESConv strategy labels + keyword patterns |
| Cognitive analytical style | Function word ratio |
| Emotional variability | Per-session GoEmotions entropy |
| Sentiment arc slope | Linear trend of valence across turns |
| Question rate | `?` count / turn count per session |

---

## Output Schema

**`data/processed/unified_conversations.csv`**

| Column | Description |
|---|---|
| conversation_id | SHA-1 hash of dataset + raw ID (stable across runs) |
| dataset | esconv / wildchat / lmsys / talk2ai |
| turn_number | 0-indexed position in conversation |
| speaker | user / assistant / supporter |
| text | Raw turn text |
| model | Model name or "human" for ESConv |

**`data/processed/emotion_scores.csv`** — above + 27 GoEmotions scores + `top_emotion`, `ekman_emotion`, `valence_pos`, `valence_neg`

---

## Setup

```bash
# Clone and activate environment
git clone https://github.com/SabbySingh1/Psych-AI.git
cd Psych-AI
python -m venv robopsych_env
source robopsych_env/bin/activate
pip install datasets transformers torch pandas tqdm huggingface_hub

# LMSYS requires HuggingFace authentication (gated dataset)
hf auth login

# Run pipeline in order
python scripts/01_load_data.py
python scripts/02_emotion_scoring.py
python scripts/03_linguistic_features.py
python scripts/04_analysis.py
python scripts/05_visualizations.py
```

> **Note:** Raw data and processed CSVs are excluded from the repo (`.gitignore`) — they are re-generated from HuggingFace on first run. Expect ~1 hour for full data load and ~1 hour for emotion scoring on Apple Silicon M-series.

---

## Project Structure

```
Psych-AI/
├── scripts/
│   ├── 01_load_data.py
│   ├── 02_emotion_scoring.py
│   ├── 03_linguistic_features.py   (coming)
│   ├── 04_analysis.py              (coming)
│   └── 05_visualizations.py        (coming)
├── notebooks/
│   └── exploration.ipynb           (coming)
├── outputs/
│   ├── figures/
│   └── tables/
├── data/                           (gitignored)
│   ├── raw/
│   └── processed/
└── README.md
```

---

## References

- Zahiri & Choi (2018). Emotion Detection on TV Show Transcripts with Sequence-Based Convolutional Neural Networks.
- Liu et al. (2021). [ESConv: Towards Emotional Support Conversation Systems](https://arxiv.org/abs/2106.01144).
- Zhao et al. (2023). [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470).
- Zheng et al. (2023). [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://arxiv.org/abs/2309.11998).
- Demichelis et al. (2025). [Talk2AI: A Longitudinal Dataset of Human-AI Persuasive Conversations](https://arxiv.org/abs/2604.04354).
- Lowe et al. (2022). [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).
