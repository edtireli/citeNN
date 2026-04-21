"""
Crossref client. Exchange a DOI for a real BibTeX entry.

Uses the public `transform/application/x-bibtex` endpoint, no auth required:
    https://api.crossref.org/works/{doi}/transform/application/x-bibtex
Polite header (`User-Agent: citenn/0.1 (mailto:...)`) as per Crossref's API etiquette.

All entries returned are by definition real DOIs, so the retriever can
never emit a hallucinated key: if a DOI resolves, it is a real paper.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


CROSSREF_BIBTEX = "https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
CROSSREF_JSON = "https://api.crossref.org/works/{doi}"


# DOIs look like 10.XXXX/... but can contain almost anything after the slash.
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s'\"<>\]\\]+", re.IGNORECASE)


def normalize_doi(doi: str) -> str:
    """Strip common prefixes and normalise case of the registrant."""
    s = str(doi).strip()
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^doi:\s*", "", s, flags=re.IGNORECASE)
    s = s.strip().rstrip(".,;)")
    return s.lower()


class CrossrefClient:
    """Rate-limited Crossref fetcher with a persistent on-disk cache."""

    def __init__(
        self,
        email: Optional[str] = None,
        delay: float = 0.05,
        cache_dir: Optional[Path] = None,
    ):
        self.email = email
        self.delay = delay
        self._last = 0.0
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "citenn" / "crossref")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ua(self) -> str:
        if self.email:
            return f"citenn/0.1 (mailto:{self.email})"
        return "citenn/0.1"

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last = time.time()

    def _cache_path(self, doi: str, kind: str) -> Path:
        safe = doi.replace("/", "_")
        return self.cache_dir / f"{safe}.{kind}"

    def bibtex(self, doi: str) -> Optional[str]:
        doi = normalize_doi(doi)
        if not doi:
            return None
        cp = self._cache_path(doi, "bib")
        if cp.exists():
            return cp.read_text()
        self._rate_limit()
        url = CROSSREF_BIBTEX.format(doi=quote(doi, safe="/"))
        req = Request(url, headers={
            "Accept": "application/x-bibtex",
            "User-Agent": self._ua(),
        })
        try:
            with urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError):
            return None
        if not data.strip().startswith("@"):
            return None
        cp.write_text(data)
        return data

    def metadata(self, doi: str) -> Optional[dict]:
        doi = normalize_doi(doi)
        if not doi:
            return None
        cp = self._cache_path(doi, "json")
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except Exception:
                cp.unlink(missing_ok=True)
        self._rate_limit()
        url = CROSSREF_JSON.format(doi=quote(doi, safe="/"))
        req = Request(url, headers={
            "Accept": "application/json",
            "User-Agent": self._ua(),
        })
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            return None
        cp.write_text(json.dumps(data))
        return data.get("message", data)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def doi_to_bibtex(doi: str, email: Optional[str] = None) -> Optional[str]:
    return CrossrefClient(email=email).bibtex(doi)


def extract_dois(text: str) -> list[str]:
    """Find all DOIs that appear verbatim in a chunk of text."""
    return list({normalize_doi(m.group(0)) for m in DOI_RE.finditer(text or "")})


def title_year_from_metadata(meta: dict) -> tuple[str, str]:
    """Best-effort (title, year) tuple for search-based matching."""
    title = ""
    if isinstance(meta.get("title"), list) and meta["title"]:
        title = meta["title"][0]
    elif isinstance(meta.get("title"), str):
        title = meta["title"]
    year = ""
    for k in ("issued", "published-print", "published-online", "published"):
        v = meta.get(k)
        if isinstance(v, dict):
            parts = v.get("date-parts") or []
            if parts and parts[0]:
                year = str(parts[0][0])
                break
    return title.strip(), year
