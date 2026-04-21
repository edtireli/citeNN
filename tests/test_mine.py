from citenn.mine_arxiv import parse_bbl, parse_bib, mine_tex_body


def test_parse_bbl_extracts_dois():
    bbl = r"""
    \begin{thebibliography}{99}
    \bibitem{foo2020} F. Oo, "Title", Journal, 2020. doi:10.1234/foo.2020.1
    \bibitem{bar2019} B. Ar, doi 10.5678/bar.2019.2
    \bibitem{nope} Y. Pe, no doi here.
    \end{thebibliography}
    """
    m = parse_bbl(bbl)
    assert m["foo2020"] == "10.1234/foo.2020.1"
    assert m["bar2019"] == "10.5678/bar.2019.2"
    assert "nope" not in m


def test_parse_bib_extracts_dois():
    bib = """
@article{foo2020,
  title = {Something},
  author = {Foo},
  doi = {10.1234/FOO.2020.1},
  year = {2020}
}
@article{nope,
  title = {No doi},
  author = {Who}
}
"""
    m = parse_bib(bib)
    assert m["foo2020"] == "10.1234/foo.2020.1"
    assert "nope" not in m


def test_mine_tex_body_pairs_citations_with_dois():
    body = (
        r"We extend the Standard Model. "
        r"Gauge invariance was shown to be fundamental \cite{foo2020}. "
        r"Later, renormalisability followed from \citep{bar2019, baz}. "
        r"An uncited sentence. "
        r"End of body."
    )
    key_to_doi = {"foo2020": "10.1/a", "bar2019": "10.2/b"}  # baz unresolved
    pairs = list(mine_tex_body(body, key_to_doi))
    assert len(pairs) == 2
    contexts = [p[0] for p in pairs]
    dois = [p[1] for p in pairs]
    # Citations must never leak into the context (otherwise the model trains
    # on a shortcut feature).
    for c in contexts:
        assert "\\cite" not in c and "\\citep" not in c
        assert "foo2020" not in c and "bar2019" not in c
    assert ["10.1/a"] in dois
    assert any(set(d) == {"10.2/b"} for d in dois)
