import pytest

from evaluation.stats import sign_test_pvalue, wilson_interval


def test_wilson_empty():
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_wilson_interval_around_half():
    lo, hi = wilson_interval(50, 100)
    assert 0.39 < lo < 0.42
    assert 0.58 < hi < 0.61


def test_sign_test_extremes():
    assert sign_test_pvalue(0, 0) == 1.0
    assert sign_test_pvalue(5, 5) == 1.0
    assert sign_test_pvalue(10, 0) == pytest.approx(2 / 1024)
    assert sign_test_pvalue(0, 10) == pytest.approx(2 / 1024)