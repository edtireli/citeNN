# citeNN

A citation recommender that **cannot hallucinate**. Given a paragraph of your draft, it suggests real DOIs and real BibTeX entries grounded in actually indexed papers from **arXiv**, **bioRxiv**, and **PubMed**.

## Why a retriever, not a generator

A fine-tuned language model asked to "add citations" will happily invent `\cite{Smith2019}` keys, author names, and even DOIs that do not exist. That is the single biggest failure mode of LLM citation tools in the wild.

citeNN uses a **bi-encoder retriever**:
- **f_ctx(sentence)** projects your writing into an embedding space.
- **f_paper(title + abstract)** projects each paper in the index into the same space.
- Inference takes the top-k nearest papers and inserts a `\cite{...}` marker using the exact BibTeX key Crossref returns for that DOI.

The model physically cannot return a DOI that is not in the index, and the index contains only DOIs that resolved against Crossref. Every suggested key has a real paper behind it.

## Pipeline

1. **Mine** — extract `(context, DOI)` pairs from arXiv LaTeX source tarballs. Each `\cite{key}` is resolved via the sibling `.bbl` / `.bib` file and the surrounding sentence is kept as supervision. The `\cite` command itself is stripped so the model cannot cheat.
2. **Train** — InfoNCE contrastive training. Each batch pulls `B` `(context, paper_text)` positives and uses the other `B-1` items as in-batch negatives. Loss is symmetric (context → paper and paper → context). Progress bar + `loss.csv` with step, loss, recall@1, recall@5, learning rate.
3. **Index** — embed every paper whose DOI was seen, save a cosine-normalised numpy matrix (or FAISS index if `faiss-cpu` is installed).
4. **Cite** — split your text into sentences, embed each, retrieve top-k, emit `\cite{...}` + companion BibTeX file fetched from Crossref.

## Install

```bash
cd /Users/edt/citeNN
pip install -e .
# optional:
pip install -e ".[faiss]"
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Mine training pairs from some arXiv tarballs
citenn mine path/to/*.tar.gz --out pairs.jsonl

# 2. Train
citenn train --pairs pairs.jsonl --output-dir ./checkpoints --max-steps 2000

# 3. Build index from same pool
citenn index --pairs pairs.jsonl --checkpoint ./checkpoints/step-2000 --out ./index

# 4. Suggest citations for a draft
citenn cite \
    --in draft.txt \
    --checkpoint ./checkpoints/step-2000 \
    --index ./index \
    --out-tex draft.cited.tex \
    --out-bib refs.bib
```

## Progress visibility

Training prints a `tqdm` progress bar with `loss`, `r@1`, `r@5`, and `lr`. A full history is written to `<output_dir>/loss.csv` with columns:

```
step,loss,r@1,r@5,lr,elapsed_sec
```

## Configuration defaults

- Backbone: `sentence-transformers/all-MiniLM-L6-v2` (small, CPU-runnable).
- Projection dim: 256.
- Max sequence length: 256 tokens.
- Temperature: 0.05.
- Batch size: 32 (in-batch negatives, so effective contrastive pool is 32).
- Optimiser: AdamW, lr 2e-5, cosine schedule with 50 warmup steps.

## Source of data

- **arXiv** — LaTeX tarballs give us `\cite{key}` + the `.bbl` with the DOI.
- **bioRxiv** — API returns metadata with DOI; PDF mining is harder but the same `(context, DOI)` pattern applies to XML variants.
- **PubMed** — full-text XML (PMC) includes structured reference lists with DOIs. Can be ingested using the same `{context, dois}` JSONL format, so no new code path is needed.

## Safety

- Retrieval-only: no generative model, no hallucinated keys.
- `min_score` threshold (default 0.45) drops weak matches rather than forcing a citation.
- Crossref fetches are cached locally under `~/.cache/citenn/crossref/`, rate-limited, and only use the public polite endpoint.
