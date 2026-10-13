# Biomedical Retrieval for RAG: BM25 vs Dense vs Hybrid

Which retrieval method should a RAG system use for biomedical questions?
This project benchmarks five retrieval approaches on [PubMedQA](https://pubmedqa.github.io/), from a naive word-overlap baseline to a cross-encoder reranker, with BM25 implemented from scratch.

It is the retrieval half of a RAG pipeline for small language models, and part of my BSc dissertation work on RAG vs GraphRAG for SLMs (Heriot-Watt University Dubai). It covers both halves of a RAG pipeline for small language models: retrieval, and answer generation with Qwen3.5-2B. Part of my BSc dissertation work on RAG vs GraphRAG for SLMs (Heriot-Watt University Dubai).

## Task

- **Questions:** 1,000 expert-labeled PubMedQA questions (`pqa_labeled`).
- **Corpus:** 6,000 PubMed abstracts: the 1,000 gold abstracts plus 5,000 distractors from `pqa_artificial`.
- **Goal:** for each question, retrieve the abstract it was written from.
- **Metrics:** Recall@1 (gold abstract ranked first) and Recall@5 (gold abstract in the top 5).

## Results

Evaluated on the first 300 questions.

| Method | Recall@1 | Recall@5 |
|---|---|---|
| Word overlap (naive) | 0.793 | 0.883 |
| Word overlap + IDF weighting | 0.890 | 0.953 |
| BM25 (from scratch) | 0.907 | 0.967 |
| **Dense** (`bge-small-en-v1.5`) | **0.953** | **0.977** |
| Hybrid: BM25 + Dense (RRF) | 0.953 | 0.973 |
| Hybrid + cross-encoder reranker | 0.950 | 0.977 |

## Findings

1. **IDF does most of the work in lexical search.** Weighting words by rarity lifted Recall@1 from 0.79 to 0.89. BM25's term-frequency saturation and length normalisation added about 2 more points.
2. **Dense retrieval beat BM25** (0.953 vs 0.907 Recall@1), even though PubMedQA questions are derived from article titles, which favours lexical matching.
3. **Hybrid search did not help.** Error analysis shows why: compared with dense alone, RRF fixed 5 questions and broke 5 others. BM25 is clearly weaker here (dense alone solved 20 questions BM25 missed; BM25 alone solved 6), so equal-weight fusion just trades errors.
4. **The reranker did not help either.** `ms-marco-MiniLM-L-6-v2` is trained on web search queries, not biomedical text, and dense retrieval is already near the ceiling: only 8 of 300 questions were missed by both BM25 and dense.

**Takeaway:** the simplest strong method, a small dense model, was the best choice. Extra pipeline stages added latency without improving recall, so each component should earn its place with a measurement.

## Where the methods disagree

BM25 wins on questions with rare, exact technical terms (*predilatation*, *unicompartmental*, *tibial component*), where a small embedding model retrieves an article on the right topic but not the right article. Dense wins on vocabulary mismatch: one question asks about *double balloon enteroscopy* while its abstract only uses the abbreviation *DBE*, so word matching cannot connect them.

## Answer generation with a small language model

**Reader:** `Qwen/Qwen3.5-2B`. For each question the model gets the top-3 abstracts from dense retrieval and answers yes / no / maybe. Answers are read from the model's next-token probabilities for the three labels (one forward pass, no free-text parsing).

**Modes:** no retrieval; RAG (top-3 dense results); oracle (the gold abstract, an upper bound for perfect retrieval).

Evaluated on 300 held-out questions (questions 50–349; label split 161 yes / 86 no / 53 maybe).

| Mode | Accuracy | Macro-F1 | Accuracy (calibrated) | Macro-F1 (calibrated) |
|---|---|---|---|---|
| Majority class ("yes") | 0.537 | 0.233 | — | — |
| No retrieval | 0.560 | 0.319 | 0.480 | 0.363 |
| **RAG (dense, top-3)** | **0.650** | 0.448 | 0.613 | **0.530** |
| Oracle (gold abstract) | 0.690 | 0.498 | 0.620 | 0.551 |

### Findings

1. **Without retrieval the 2B model barely beats always answering "yes"** (0.560 vs 0.537): it answered "yes" to 271 of 300 questions.
2. **Retrieval adds 9 points of accuracy** (0.560 → 0.650) and closes about 70% of the gap to the oracle. The bottleneck is now the reader, not the retriever.
3. **The model has a strong yes-bias.** On a content-free question ("N/A") it answers "yes" with probability 0.80, and it almost never says "maybe".
4. **Contextual calibration fixes the label distribution.** Dividing the label probabilities by the model's content-free bias brought RAG predictions to 164 / 87 / 49 against the true 161 / 86 / 53, and raised macro-F1 from 0.448 to 0.530. The trade-off is 3–8 points of accuracy, since the uncalibrated model scored partly by defaulting to the majority class.

### Method note

The calibration variant was chosen on the first 50 questions. Calibrating with a bias computed on the same retrieved abstracts over-predicted "maybe" there, so the final version uses a single bias computed without context. All numbers above are on 300 different questions, so the choice was not tuned on the reported results.

## How it works

- **BM25** (`src/bm25.py`): inverted-index style term statistics, IDF, term-frequency saturation (`k1 = 1.5`) and document length normalisation (`b = 0.75`). No search libraries.
- **Dense:** abstracts embedded once with `BAAI/bge-small-en-v1.5` (384 dimensions, normalised), queries prefixed with the bge retrieval instruction, cosine similarity via a dot product.
- **Hybrid:** Reciprocal Rank Fusion over the top 50 of each retriever. RRF combines ranks rather than scores, so BM25 scores and cosine similarities never need to be put on the same scale.
- **Reranker:** cross-encoder re-scores the top 20 hybrid candidates by reading question and abstract together.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then run the notebooks in order: `data.ipynb` (downloads PubMedQA and builds the corpus), `bm25.ipynb`, `dense.ipynb`.

## Limitations

- Results are on 300 questions; differences under about 1 point are within noise.
- PubMedQA questions are derived from article titles, which is easier than real user queries.
- The corpus is 6,000 abstracts; a larger corpus with more distractors would be harder.

## Next steps

- [ ] Evaluate on all 1,000 questions
- [ ] Compare reader sizes and families from the dissertation
- [ ] Gradio demo
- [ ] GraphRAG comparison (dissertation)