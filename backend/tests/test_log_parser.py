
import pytest
from ingestion.log_parser import parse_log


@pytest.mark.parametrize(
    "log",
    [
        "AssertionError: expected 200 but got 500",
        "ModuleNotFoundError: No module named 'requests'",
        "TimeoutError: operation timed out",
    ],
)
def test_parse_log_returns_searchable_text(log):
    result = parse_log(log)

    assert result is not None
    assert callable(getattr(result, "to_search_text", None))

    text = result.to_search_text()
    assert isinstance(text, str)
    assert text.strip()


def test_parse_log_handles_empty_input():
    result = parse_log("")
    assert result is not None
    assert isinstance(result.to_search_text(), str)


def test_parse_log_is_deterministic():
    log = "AssertionError: expected 10 but got 5"

    first = parse_log(log).to_search_text()
    second = parse_log(log).to_search_text()

    assert first == second
