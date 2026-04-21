import numpy as np

from citenn.index import CitationIndex


def test_index_roundtrip_and_search(tmp_path):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(8, 16)).astype(np.float32)
    dois = [f"10.1/{i}" for i in range(8)]
    titles = [f"Paper {i}" for i in range(8)]
    idx = CitationIndex(dim=16)
    idx.build(dois, titles, vecs)

    # Querying with the first vector should return index 0 first.
    hits = idx.search(vecs[:1], k=3)[0]
    assert hits[0][0] == "10.1/0"
    assert hits[0][2] > 0.99  # cosine with itself ≈ 1

    # Persistence
    idx.save(tmp_path / "idx")
    loaded = CitationIndex.load(tmp_path / "idx")
    hits2 = loaded.search(vecs[:1], k=3)[0]
    assert [h[0] for h in hits] == [h[0] for h in hits2]
