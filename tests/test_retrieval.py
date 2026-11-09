from app.retrieval import rrf_merge


def _chunk(i):
    return {"id": i, "page": i, "content": f"chunk {i}", "filename": "f.pdf", "score": 0.0}


def test_rrf_prefers_chunks_found_by_both_searches():
    vector_results = [_chunk(1), _chunk(2), _chunk(3)]
    keyword_results = [_chunk(3), _chunk(4), _chunk(2)]
    merged = rrf_merge(vector_results, keyword_results)
    assert [c["id"] for c in merged] == [3, 2, 1, 4]


def test_rrf_has_no_duplicates_and_keeps_single_list_order():
    only = [_chunk(5), _chunk(6)]
    merged = rrf_merge(only, [])
    assert [c["id"] for c in merged] == [5, 6]