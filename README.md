# Psych-AI

## Overview

Human interaction with AI systems has moved far beyond task completion. People increasingly turn to AI chatbots for emotional support, personal disclosure, and companionship — yet the psychological dynamics of these interactions remain poorly understood. This project applies computational methods to characterize how emotional expression, empathy, and self-disclosure unfold differently in human-to-human counseling versus human-to-AI conversation, across four large-scale public datasets totaling over 100,000 conversations.

This sits within the emerging field of **robopsychology** — the psychological study of how humans relate to, are shaped by, and form patterns of interaction with AI systems. To our knowledge, this is the first study to conduct a cross-dataset emotional analysis using human counseling conversations (ESConv) as a psychological baseline against which naturalistic and structured AI conversations are systematically compared.

---

## Research Questions

- How do emotion distributions differ between human counselors and AI assistants?
- Do AI models (GPT-4, Claude, Mistral, etc.) mirror user emotional states or respond with consistent affect regardless of user emotion?
- How does self-disclosure and hedging language vary across datasets and speaker roles?
- What does the emotional arc of a conversation look like — and does it differ between ESConv support sessions and open-domain AI chat?

---

## Hypotheses

Hypotheses are grounded in the existing robopsychology and human-computer interaction literature and are stated directionally prior to analysis.

**H1 — Emotion distribution:** AI assistants will show significantly higher positive affect (joy, optimism) and lower negative affect (sadness, fear) than human counselors in ESConv, regardless of the user's emotional state. This is predicted by research showing LLMs are biased toward positive emotional tone and by sycophancy literature demonstrating that AI systems systematically affirm and validate users.

**H2 — Emotional mirroring:** Human counselors in ESConv will show greater turn-by-turn emotional alignment with the user than AI assistants across WildChat and LMSYS. AI assistants are expected to maintain more stable, context-independent affect — consistent with findings that LLMs lock into repetitive support tactics at nearly double the rate of human supporters.

**H3 — Self-disclosure asymmetry:** Users in WildChat (naturalistic, unsolicited use) will show significantly higher rates of emotional self-disclosure than users in LMSYS (evaluation context), reflecting the dampening effect of an explicitly evaluative interface on personal expression. ESConv seekers will show the highest self-disclosure rates of all, consistent with the intentional help-seeking context of that dataset.

**H4 — Emotional arc:** ESConv conversations will show a measurable negative-to-positive emotional arc across turns (distress → relief), consistent with Hill's Helping Skills Theory which underpins the dataset's design. WildChat conversations containing emotional content will show a flatter or more variable arc, reflecting the absence of structured therapeutic intent in AI responses.

**H5 — Model differences:** Across LMSYS, models will differ significantly in their emotional response profiles. Models with stronger RLHF alignment (GPT-4) will show higher positive affect and lower emotional variability than open-source models (Mistral, LLaMA variants), consistent with research showing alignment training produces more uniformly warm and validating responses.

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

### What Makes This Novel

Prior computational work on these datasets has focused on task completion, toxicity detection, and model benchmarking. No published study has applied a unified emotional analysis framework across all three datasets simultaneously, or used human emotional support conversations as a psychological baseline for comparison. The cross-dataset design allows us to isolate whether emotional dynamics in AI conversations are a property of the technology, the user's intent, or the conversational context — a question with direct implications for AI companion design and mental health applications.

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

## Authors

- **Sabadnoor Singh** — Independent Researcher

---

## Target Publication Venue

Primary: *Computers in Human Behavior* (Elsevier) — the leading journal for psychological and behavioral research on technology interactions. Impact factor ~9.

Backup: *PLOS ONE* — open access, broad methodological scope, ensures public availability of findings.

---

## References

- Zahiri & Choi (2018). Emotion Detection on TV Show Transcripts with Sequence-Based Convolutional Neural Networks.
- Liu et al. (2021). [ESConv: Towards Emotional Support Conversation Systems](https://arxiv.org/abs/2106.01144).
- Zhao et al. (2023). [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://arxiv.org/abs/2405.01470).
- Zheng et al. (2023). [LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset](https://arxiv.org/abs/2309.11998).
- Demichelis et al. (2025). [Talk2AI: A Longitudinal Dataset of Human-AI Persuasive Conversations](https://arxiv.org/abs/2604.04354).
- Lowe et al. (2022). [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions).
