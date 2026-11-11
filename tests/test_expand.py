from app.retrieval import merge_page_group


def ch(i, page):
    return {"id": i, "page": page, "content": f"c{i}", "filename": "f.pdf", "score": 1.0}


def test_page_group_is_in_reading_order_and_first():
    hits = [ch(30, 12), ch(5, 3), ch(31, 12)]
    page = [ch(i, 12) for i in (28, 29, 30, 31, 32, 33)]
    out = merge_page_group(hits, page)
    assert [c["id"] for c in out] == [28, 29, 30, 31, 32, 33, 5]
    assert [c["id"] for c in out if c.get("expanded")] == [28, 29, 32, 33]
    assert out[2]["score"] == 1.0


def test_cap_on_extra_chunks():
    out = merge_page_group([ch(1, 1)], [ch(i, 1) for i in range(1, 20)], max_extra=3)
    assert len(out) == 4


def test_empty_inputs_unchanged():
    assert merge_page_group([], [ch(1, 1)]) == []
    hits = [ch(1, 1)]
    assert merge_page_group(hits, []) == hits