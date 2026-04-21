"""
Mine (context, cited_DOI) training pairs from real papers.

Strategy
--------
For arXiv source tarballs:

  1. Find the main `.tex` and its `.bbl` (or `.bib`) siblings.
  2. Parse the bibliography to build a map cite_key -> DOI.
     - From .bbl: regex for \\bibitem{key} ... doi{...} / 10.xxxx/... patterns.
     - From .bib: @article{key, ..., doi = {...}}.
  3. Walk the body, find every \\cite{keyA,keyB} and keep the surrounding
     sentence as `context`. Emit one (context, doi) pair per resolved key.

The output record:
    {"context": "...", "dois": ["10.1038/...","10.1103/..."], "source_id": "..."}

This gives the retriever real-world training examples where a sentence's
meaning is supervised against the papers a domain expert actually cited.

Other sources (PubMed, bioRxiv) can be mined similarly by extracting the
references section and keeping each sentence as context with the DOIs
found within the ~1-3 sentences immediately around a citation marker.
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable, Iterator

from citenn.crossref import DOI_RE, normalize_doi


# ---------------------------------------------------------------------------
# Bibliography parsers
# ---------------------------------------------------------------------------

# .bbl entries: \bibitem[opt]{key} ... <stuff possibly containing a DOI>
_BBL_ITEM = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\s*\{([^{}]+)\}"
    r"(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)",
    re.DOTALL,
)

# .bib entries: @article{key, ... doi = {...} ... }
_BIB_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}\s*", re.DOTALL)
_BIB_DOI = re.compile(r"doi\s*=\s*[\"{]\s*([^\"{}]+)\s*[\"}]", re.IGNORECASE)


def parse_bbl(bbl_text: str) -> dict[str, str]:
    """Return cite_key -> normalised DOI for all items whose DOI is discoverable."""
    out: dict[str, str] = {}
    for m in _BBL_ITEM.finditer(bbl_text):
        key, body = m.group(1).strip(), m.group(2)
        dm = DOI_RE.search(body)
        if dm:
            out[key] = normalize_doi(dm.group(0))
    return out


def parse_bib(bib_text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _BIB_ENTRY.finditer(bib_text):
        key, body = m.group(1).strip(), m.group(2)
        dm = _BIB_DOI.search(body) or DOI_RE.search(body)
        if dm:
            out[key] = normalize_doi(dm.group(1) if dm.lastindex else dm.group(0))
    return out


# ---------------------------------------------------------------------------
# Body parser: pull (sentence-context, [keys]) pairs
# ---------------------------------------------------------------------------

# Match any of the cite-family commands. Captures the braced key list.
_CITE_CMD = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|citeyear|citeauthor|citenum|"
    r"footcite|autocite|parencite|textcite|fullcite|Cite)\*?"
    r"(?:\s*\[[^\[\]]*\])*"
    r"\s*\{([^{}]+)\}"
)

# Sentence splitter: naive but good enough for scientific prose.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _strip_latex_noise(text: str) -> str:
    """Remove \\label, \\ref etc. from a context window without touching \\cite
    (we rely on those being already removed when building the context)."""
    text = re.sub(r"\\(?:label|ref|eqref|autoref|cref|Cref|pageref)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\(?:emph|textbf|textit|texttt|textsc)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{[^{}]*\}|\\href\{[^{}]*\}\{[^{}]*\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mine_tex_body(
    body: str,
    key_to_doi: dict[str, str],
    context_window: int = 400,
) -> Iterator[tuple[str, list[str]]]:
    """Yield (context, list_of_dois) pairs from a cleaned LaTeX body string.

    `context` is the surrounding text up to `context_window` chars on each
    side, with the \\cite command itself removed. We then keep only the
    sentence(s) that actually surround the citation to keep the signal
    localised.
    """
    positions = [(m.start(), m.end(), m.group(1)) for m in _CITE_CMD.finditer(body)]
    if not positions:
        return

    for start, end, key_str in positions:
        keys = [k.strip() for k in key_str.split(",") if k.strip()]
        dois = [key_to_doi[k] for k in keys if k in key_to_doi]
        if not dois:
            continue
        # Grab a window around the citation and kill the cite command itself.
        left = body[max(0, start - context_window) : start]
        right = body[end : end + context_window]
        raw = left + " " + right  # drop the \cite{...}
        # Remove any other \cite commands in the window so the context
        # cannot leak cite markers into the training signal.
        raw = _CITE_CMD.sub("", raw)
        raw = _strip_latex_noise(raw)
        # Pick the sentence(s) closest to where the citation was.
        sents = _SENT_SPLIT.split(raw)
        if not sents:
            continue
        # Heuristic: the citation sat near the boundary of two halves we
        # concatenated above, so the middle sentence is the most relevant.
        mid = len(sents) // 2
        ctx = " ".join(sents[max(0, mid - 1) : mid + 1]).strip()
        if len(ctx) < 40:
            continue
        yield ctx, dois


# ---------------------------------------------------------------------------
# Tarball entry point
# ---------------------------------------------------------------------------

def mine_arxiv_tarball(tar_path: os.PathLike) -> list[dict]:
    """Return a list of mined records from one arXiv source archive."""
    tar_path = str(tar_path)
    records: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        try:
            with tarfile.open(tar_path, "r:*") as tf:
                members = [
                    m for m in tf.getmembers()
                    if m.isreg()
                    and not m.name.startswith("/")
                    and ".." not in m.name.split("/")
                ]
                tf.extractall(tdp, members=members)
        except (tarfile.ReadError, OSError):
            # Single-file source: nothing to mine (no .bbl)
            return records

        # Build cite_key -> DOI map from all bbl/bib files found.
        key_to_doi: dict[str, str] = {}
        for p in list(tdp.rglob("*.bbl")) + list(tdp.rglob("*.bib")):
            try:
                txt = p.read_text(errors="ignore")
            except OSError:
                continue
            if p.suffix == ".bbl":
                key_to_doi.update(parse_bbl(txt))
            else:
                key_to_doi.update(parse_bib(txt))

        if not key_to_doi:
            return records

        # Find body from the main .tex
        main = None
        for p in tdp.rglob("*.tex"):
            try:
                content = p.read_text(errors="ignore")
            except OSError:
                continue
            if "\\begin{document}" in content:
                main = (p, content)
                break
        if main is None:
            return records

        _, body = main
        # Strip comments first — matches the cleaner's rule.
        body = re.sub(r"(?<!\\)%.*?(?=\n|$)", "", body)

        for ctx, dois in mine_tex_body(body, key_to_doi):
            records.append({"context": ctx, "dois": dois})

    return records


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterable[dict], path: os.PathLike) -> int:
    n = 0
    with Path(path).open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n
