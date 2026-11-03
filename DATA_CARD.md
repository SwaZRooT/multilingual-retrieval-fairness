# Data Card: Belebele (EN / HI / ML subset)

**Used for:** Topic 39, Multilingual Retrieval Fairness.
**Substitution note:** Replaces the suggested FLORES-200 / MIRACL / IndicQA / Samanantar sets because none of them alone
provides parallel passages, questions and relevance labels. Belebele is built on FLORES-200 passages.

| Item | Value |
|---|---|
| Source | Hugging Face `facebook/belebele` (Meta) |
| Languages used | English (`eng_Latn`), Hindi (`hin_Deva`), Malayalam (`mal_Mlym`) |
| Size per language | 488 unique passages, 900 multiple-choice questions (4 options each) |
| Parallel | Yes: same passages and questions in every language |
| Relevance labels | Each question maps to its source passage (built in `src/build_dataset.py`) |
| Licence | Check the dataset card before redistribution (believed CC BY-SA 4.0; FLORES-200 passages likewise). Raw data is NOT committed to this repo. |
| Personal data | None expected (translated encyclopedic and educational text). No attempt was made to identify anyone. |

## Known limitations
- Reading-comprehension questions written for evaluation, cleaner than real citizen queries.
- Passages are translations of English text, so they may sound unlike native Hindi or Malayalam writing.
- Raw rows are not aligned across languages; we join on (`link`, `question_number`).
- Machine translations (NLLB-200 distilled 600M) derived from this data are stored locally and are not redistributed.

## Models used (model-card pointers)
BAAI/bge-m3, intfloat/multilingual-e5-small, sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2,
facebook/nllb-200-distilled-600M, Qwen2.5 3b and 7b via Ollama. Check each model's licence before reuse.
