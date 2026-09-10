# Psych-AI

Computational comparison of emotional expression, empathy, and self-disclosure in human counseling versus AI-mediated conversation.

**[Read the full paper →](paper/full_draft.md)**

## Overview

People increasingly bring emotional disclosure and requests for support into conversations with AI systems that have no human counterpart in the loop. This project applies a single unified emotion-classification and linguistic-feature pipeline to four large-scale conversation corpora — human emotional-support counseling (ESConv), naturalistic AI chat (WildChat), a multi-model comparison arena (LMSYS-Chat-1M), and publicly shared conversations across five commercial AI platforms (ShareChat) — to ask whether AI systems track user distress the way a trained human supporter does, and whether that varies by platform.

This sits within **robopsychology** — the study of how humans relate to, and are shaped by, AI systems.

## Key Findings

- Users disclose personal content to AI at roughly **one-tenth the rate** observed with human counselors — a gap that persists even in publicly shared conversations.
- Emotional range within a single AI conversation is **four to five standard deviations narrower** than with a human counselor.
- Turn-pair analysis of 177,784 user-to-AI exchanges shows human supporters **reduce** their own positivity as user distress rises (r = −0.225); every AI platform tested shows the **opposite** direction (r = 0.010–0.106) — a sign inversion, not just a weaker effect.
- This varies systematically by platform: Gemini's response most closely approximates human tracking; Perplexity's is flattest; Claude and Grok fall in between.
- AI-mediated distressed conversations are **shorter**, not longer, than non-distressed ones — a pattern that complicates simple engagement-optimization explanations of this behavior.

Full statistics, methodology, limitations, and discussion are in [`paper/full_draft.md`](paper/full_draft.md). Figures are in [`outputs/figures/`](outputs/figures/); result tables are in [`outputs/tables/`](outputs/tables/).

## Research Questions

- Do AI assistants show a different balance of positive and negative affect than human counselors, independent of the user's own emotional state?
- Do human counselors show greater turn-by-turn emotional alignment with user distress than AI assistants?
- Does self-disclosure differ systematically between human-counseling and AI-mediated contexts, and with platform?
- Does a human-counseling conversation show a measurable emotional arc across turns that AI-mediated conversation lacks?

These questions motivated the initial study design. As documented in the paper's Methods section, several additional analyses (the platform breakdown, the turn-pair mirroring measure, and two robustness checks) were added after an early finding prompted a closer look at mechanism — this is disclosed explicitly rather than presented as pre-registered.

## Datasets

| Dataset | Conversations | Turns | Language | Focus |
|---|---|---|---|---|
| [ESConv](https://huggingface.co/datasets/thu-coai/esconv) | 910 | 26,648 | English | Human-to-human emotional support counseling |
| [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | 50,000 | 290,284 | English | Real-world ChatGPT interaction logs |
| [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) | 50,000 | 202,030 | English | Multi-model chat arena (gated; assistant turns only) |
| [ShareChat](https://huggingface.co/datasets/tucnguyen/ShareChat) | 7,936 | 46,842 | English | Publicly shared conversations across 5 commercial AI platforms (ChatGPT, Claude, Gemini, Grok, Perplexity) |

**Combined corpus:** 108,846 conversations / 565,804 turns.

## Pipeline

```
01_load_data.py                       Load ESConv, WildChat, LMSYS; standardize into
                                       unified CSV (conversation_id, dataset, turn_number,
                                       speaker, text, model)
01b_load_sharechat.py                 Load ShareChat (5 platforms, English-only)
        ↓
02_emotion_scoring.py                 Score every turn with SamLowe/roberta-base-go_emotions
                                       (27 GoEmotions labels → Ekman 6 + neutral, valence scores)
        ↓
03_linguistic_features.py             Extract self-disclosure, hedging, empathy markers,
03b_linguistic_features_sharechat.py  cognitive analytical style, question rate, valence arc slope
        ↓
04_analysis.py                        Cross-dataset and ShareChat platform-level statistical comparisons
        ↓
05_visualizations.py                  Core figures and tables
06_mirroring_analysis.py              Turn-pair emotional mirroring across all datasets and platforms
07_paper_figures.py                   Paper-specific figures (self-disclosure, mirroring, platform profile, arc)
08_crisis_adjacent_analysis.py        Crisis-adjacent language subset analysis
```

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

## Linguistic Features (Scripts 03 / 03b)

| Feature | Operationalization |
|---|---|
| Self-disclosure | 1st-person pronoun density (I/me/myself / word count) |
| Hedging / uncertainty | Epistemic hedge keyword density |
| Empathy markers | Supportive/validating keyword density |
| Cognitive analytical style | Analytical-connective keyword density |
| Emotion entropy | Shannon entropy over the 27-label GoEmotions distribution |
| Valence arc slope | Linear trend of valence across turns, per conversation |
| Question rate | `?` count / sentence count per turn |

Each feature is a keyword- or token-density proxy, not a validated psychometric instrument — see `paper/methods.md` for full detail and limitations.

## Output Schema

**`data/processed/unified_conversations.csv`**

| Column | Description |
|---|---|
| conversation_id | SHA-1 hash of dataset + raw ID (stable across runs) |
| dataset | esconv / wildchat / lmsys / sharechat |
| turn_number | 0-indexed position in conversation |
| speaker | user / assistant / supporter |
| text | Raw turn text |
| model | Model name or "human" for ESConv |

ShareChat additionally carries a `platform` column (chatgpt / claude / gemini / grok / perplexity).

**`data/processed/emotion_scores.csv`** — above + 27 GoEmotions scores + `top_emotion`, `ekman_emotion`, `valence_pos`, `valence_neg`

## Setup

```bash
git clone https://github.com/SabbySingh1/Psych-AI.git
cd Psych-AI
python -m venv robopsych_env
source robopsych_env/bin/activate
pip install datasets transformers torch pandas scipy matplotlib tqdm huggingface_hub

# LMSYS and ShareChat are gated on HuggingFace — request access, then:
hf auth login

# Run pipeline in order
python scripts/01_load_data.py
python scripts/01b_load_sharechat.py
python scripts/02_emotion_scoring.py
python scripts/03_linguistic_features.py
python scripts/03b_linguistic_features_sharechat.py
python scripts/04_analysis.py
python scripts/05_visualizations.py
python scripts/06_mirroring_analysis.py
python scripts/07_paper_figures.py
python scripts/08_crisis_adjacent_analysis.py
```

> Raw data and processed CSVs are excluded from the repo (`.gitignore`) — they are regenerated from HuggingFace on first run. Expect roughly an hour each for data loading and emotion scoring on Apple Silicon.

## Project Structure

```
Psych-AI/
├── scripts/            Pipeline (01-08, see above)
├── paper/               Full paper draft (abstract, introduction, methods, results, discussion)
├── outputs/
│   ├── figures/         All figures, 300 DPI PNG (+ PDF for paper figures)
│   └── tables/          All result tables (CSV)
├── data/                 Gitignored — regenerated from HuggingFace
│   ├── raw/
│   └── processed/
└── README.md
```

## Author

- **Sabadnoor Singh** — Independent Researcher

## References

- Chu, M. D., Gerard, P., Pawar, K., Bickham, C., & Lerman, K. (2025). Illusions of Intimacy: How Emotional Dynamics Shape Human-AI Relationships. arXiv:2505.11649.
- Chu, M. D., Wu, Y., Chen, Z., Hwang, A. H., & Luceri, L. (2026). When Chatbots Accommodate: What AI Companions Optimize for in Vulnerable Conversations. arXiv:2606.04431.
- Liu, S., Zheng, C., Demasi, O., Sabour, S., Li, Y., Yu, Z., Jiang, Y., & Huang, M. (2021). Towards Emotional Support Dialog Systems. ACL 2021. arXiv:2106.01144.
- Lowe, S. (2022). SamLowe/roberta-base-go_emotions.
- McBain, R. K., et al. (2026). AI Chatbot Use and Disclosure for Mental Health Among US Adolescents and Young Adults. *JAMA Pediatrics*, 180(8), 884–890.
- Yan, Y., Nguyen, T., Su, B., Lieffers, M., & Le, T. (2026). ShareChat: A Dataset of Chatbot Conversations in the Wild. arXiv:2512.17843.
- Zhao, W., Ren, X., Hessel, J., Cardie, C., Choi, Y., & Deng, Y. (2024). WildChat: 1M ChatGPT Interaction Logs in the Wild. ICLR 2024. arXiv:2405.01470.
- Zheng, L., et al. (2024). LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset. ICLR 2024. arXiv:2309.11998.
- Zhu, J., Coifman, K. G., & Jin, R. (2026). Understanding Risk and Dependency in AI Chatbot Use from User Discourse. arXiv:2602.09339.

Full reference list with complete author lists: [`paper/full_draft.md`](paper/full_draft.md#references).
