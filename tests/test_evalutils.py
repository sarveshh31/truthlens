from app.evalutils import first_relevant_rank, is_relevant, load_golden, normalize, summarize


def test_normalize_ignores_separators_and_case():
    assert normalize("EUR 22,465 million") == "eur22465million"


def test_relevant_by_page_or_keyword():
    item = {"pages": [32], "keywords": ["22,465"]}
    assert is_relevant({"page": 32, "content": "x"}, item)
    assert is_relevant({"page": 9, "content": "revenues were 22 465 million"}, item)
    assert not is_relevant({"page": 9, "content": "nothing here"}, item)


def test_rank_and_summary():
    item = {"pages": [2], "keywords": []}
    chunks = [{"page": 1, "content": "a"}, {"page": 2, "content": "b"}]
    assert first_relevant_rank(chunks, item) == 2
    s = summarize([1, 2, None, 5], 5)
    assert s["hit@1"] == 0.25
    assert s["hit@3"] == 0.5
    assert s["hit@5"] == 0.75
    assert abs(s["mrr"] - (1 + 0.5 + 0.2) / 4) < 1e-9


def test_load_golden_reports_missing_pages(tmp_path):
    f = tmp_path / "g.json"
    f.write_text('[{"id": 1, "question": "Q?", "answerable": true}]', encoding="utf-8")
    items, problems = load_golden(f)
    assert len(items) == 1
    assert any("pages" in p for p in problems)