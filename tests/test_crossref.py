from citenn.crossref import extract_dois, normalize_doi


def test_normalize_doi_strips_prefixes():
    assert normalize_doi("https://doi.org/10.1038/NATURE12373") == "10.1038/nature12373"
    assert normalize_doi("doi: 10.1103/PhysRevD.98.030001") == "10.1103/physrevd.98.030001"
    assert normalize_doi("10.1103/PhysRevLett.123.10.") == "10.1103/physrevlett.123.10"


def test_extract_dois_finds_all():
    text = (
        "See 10.1038/nature12373 and also https://doi.org/10.1103/PhysRevD.98.030001 "
        "plus a junk string 10.xxxx not a doi."
    )
    got = set(extract_dois(text))
    assert "10.1038/nature12373" in got
    assert "10.1103/physrevd.98.030001" in got
