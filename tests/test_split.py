from app.verify import split_sentences


def test_numbered_list_is_not_split_into_fragments():
    text = (
        "The priorities are:\n"
        "1. Growth and technology: acquire WNS [1].\n"
        "2. Market monitoring: review changes [1]."
    )
    assert split_sentences(text) == [
        "Growth and technology: acquire WNS [1].",
        "Market monitoring: review changes [1].",
    ]


def test_two_sentences_on_one_line():
    assert split_sentences("Revenue was 5 million [1]. Profit rose [2].") == [
        "Revenue was 5 million [1].",
        "Profit rose [2].",
    ]


def test_bullets_and_bold_are_cleaned():
    assert split_sentences("- **Net zero** by 2040 [4].") == ["Net zero by 2040 [4]."]


def test_decimal_at_line_start_is_kept():
    assert split_sentences("5.2 billion euros were invested [1].") == [
        "5.2 billion euros were invested [1]."
    ]