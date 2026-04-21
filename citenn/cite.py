"""
Citation suggester. Given a checkpoint, an index, and user text, splits the
text into sentences, embeds each, retrieves the top-k DOIs and inserts
\\cite{key} markers. A companion .bib file is emitted with real entries
fetched from Crossref, so every key the model inserts is guaranteed to
resolve to a real paper.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\\])")


def _bibkey_from_bibtex(bib: str) -> Optional[str]:
    m = re.match(r"@\w+\s*\{\s*([^,\s}]+)\s*,", bib)
    return m.group(1) if m else None


def suggest_citations(
    text: str,
    checkpoint_dir: str,
    index_dir: str,
    k: int = 3,
    min_score: float = 0.45,
    crossref_email: Optional[str] = None,
) -> tuple[str, str]:
    """
    Return (annotated_text, bibtex_document).

    Only citations whose retrieval score exceeds `min_score` are emitted,
    which is a hard filter against low-confidence (and therefore likely
    off-topic) suggestions.
    """
    import torch
    from citenn.model import BiEncoder, BiEncoderConfig
    from citenn.index import CitationIndex
    from citenn.crossref import CrossrefClient

    ckpt_dir = Path(checkpoint_dir)
    state = torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg_dict = state["cfg"]
    cfg = BiEncoderConfig(
        backbone=cfg_dict["backbone"],
        proj_dim=cfg_dict["proj_dim"],
        max_len=cfg_dict["max_len"],
        temperature=cfg_dict["temperature"],
    )
    model = BiEncoder(cfg)
    model.load_state_dict(state["model"])
    model.eval()

    index = CitationIndex.load(index_dir)
    cr = CrossrefClient(email=crossref_email)

    sents = [s.strip() for s in _SENT.split(text) if s.strip()]
    if not sents:
        return text, ""

    with torch.no_grad():
        z = model.encode_contexts(sents).cpu().numpy()

    hits_per_sent = index.search(z, k=k)

    used_dois: dict[str, str] = {}  # doi -> bibkey
    bib_entries: list[str] = []
    annotated_sents: list[str] = []

    for sent, hits in zip(sents, hits_per_sent):
        good = [(doi, title, score) for doi, title, score in hits if score >= min_score]
        if not good:
            annotated_sents.append(sent)
            continue
        keys: list[str] = []
        for doi, _title, _score in good:
            if doi in used_dois:
                keys.append(used_dois[doi])
                continue
            bib = cr.bibtex(doi)
            if not bib:
                continue
            key = _bibkey_from_bibtex(bib)
            if not key:
                continue
            # Make keys unique across the document even across DOIs.
            base = key
            i = 2
            existing = set(used_dois.values())
            while key in existing:
                key = f"{base}_{i}"
                i += 1
            used_dois[doi] = key
            # Patch the bibkey in the bibtex text itself so it matches.
            bib = re.sub(r"(@\w+\s*\{\s*)[^,\s}]+", r"\g<1>" + key, bib, count=1)
            bib_entries.append(bib.strip())
            keys.append(key)
        if keys:
            # Trim trailing '.' off sentence, append \cite, then restore '.'.
            trimmed = sent.rstrip()
            tail = ""
            while trimmed and trimmed[-1] in ".!?":
                tail = trimmed[-1] + tail
                trimmed = trimmed[:-1]
            annotated_sents.append(f"{trimmed} \\cite{{{','.join(keys)}}}{tail}")
        else:
            annotated_sents.append(sent)

    return " ".join(annotated_sents), "\n\n".join(bib_entries) + ("\n" if bib_entries else "")


# ---------------------------------------------------------------------------
# Helper: build a paper index from a pairs JSONL file
# ---------------------------------------------------------------------------

def build_index_from_pairs(
    pairs_file: str,
    checkpoint_dir: str,
    index_dir: str,
    crossref_email: Optional[str] = None,
) -> int:
    """Build a CitationIndex from every DOI referenced by the mined pairs."""
    import json
    import torch
    from tqdm.auto import tqdm

    from citenn.model import BiEncoder, BiEncoderConfig
    from citenn.index import CitationIndex
    from citenn.crossref import CrossrefClient

    ckpt = Path(checkpoint_dir)
    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
    cfg_dict = state["cfg"]
    cfg = BiEncoderConfig(
        backbone=cfg_dict["backbone"],
        proj_dim=cfg_dict["proj_dim"],
        max_len=cfg_dict["max_len"],
        temperature=cfg_dict["temperature"],
    )
    model = BiEncoder(cfg)
    model.load_state_dict(state["model"])
    model.eval()

    cr = CrossrefClient(email=crossref_email)
    dois_seen: set[str] = set()
    dois: list[str] = []
    titles: list[str] = []
    texts: list[str] = []

    for line in Path(pairs_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for doi in r.get("dois", []):
            if doi in dois_seen:
                continue
            meta = cr.metadata(doi)
            if not meta:
                continue
            title = ""
            if isinstance(meta.get("title"), list) and meta["title"]:
                title = meta["title"][0]
            elif isinstance(meta.get("title"), str):
                title = meta["title"]
            abstract = re.sub(r"<[^>]+>", " ", meta.get("abstract", "") or "")
            text = re.sub(r"\s+", " ", (title + ". " + abstract).strip())
            if len(text) < 20:
                continue
            dois_seen.add(doi)
            dois.append(doi)
            titles.append(title)
            texts.append(text)

    if not texts:
        raise RuntimeError("no papers resolved; nothing to index")

    embs: list[np.ndarray] = []
    import torch as _torch
    with _torch.no_grad():
        for i in tqdm(range(0, len(texts), 32), desc="embed", unit="batch"):
            chunk = texts[i : i + 32]
            z = model.encode_papers(chunk).cpu().numpy()
            embs.append(z)
    matrix = np.vstack(embs)
    idx = CitationIndex(dim=matrix.shape[1])
    idx.build(dois, titles, matrix)
    idx.save(index_dir)
    return len(dois)
