from ingestion.log_parser import parse_log

PYTEST_VERBOSE = "tests/test_cart.py::TestCart::test_total FAILED                          [ 50%]\n"
PYTEST_XDIST = "[gw0] [ 50%] FAILED tests/test_cart.py::test_total\n"
SUREFIRE_PER_TEST = "[ERROR] totalIsCorrect(com.example.CartTest)  Time elapsed: 0.01 s  <<< FAILURE!\n"
SUREFIRE_SUMMARY = (
    "[ERROR] Failures: \n"
    "[ERROR]   CartTest.totalIsCorrect:42 expected: <11> but was: <10>\n"
)
JEST_SUITE_FAILED = """\
FAIL src/utils/format.test.ts
  ● Test suite failed to run

    Cannot find module 'lodash' from 'src/utils/format.ts'
"""
PYTHON_TRACEBACK = """\
Traceback (most recent call last):
  File "/home/runner/work/app/app/services/billing.py", line 10, in charge
    raise ValueError("bad amount")
ValueError: bad amount
"""
PYTEST_LOCATION = """\
    def test_total():
>       assert 1 == 2
E       assert 1 == 2

tests/test_billing.py:25: AssertionError
"""


def test_pytest_verbose_line():
    p = parse_log(PYTEST_VERBOSE)
    assert p.test_name == "TestCart::test_total"
    assert p.module == "cart"


def test_pytest_xdist_line():
    p = parse_log(PYTEST_XDIST)
    assert p.test_name == "test_total"
    assert p.module == "cart"


def test_surefire_per_test_line():
    assert parse_log(SUREFIRE_PER_TEST).test_name == "CartTest.totalIsCorrect"


def test_surefire_failure_summary():
    assert parse_log(SUREFIRE_SUMMARY).test_name == "CartTest.totalIsCorrect"


def test_jest_suite_failure_uses_file_name_and_detects_missing_module():
    p = parse_log(JEST_SUITE_FAILED)
    assert p.test_name == "format.test.ts"
    assert p.module == "utils"
    assert p.error_type == "ModuleNotFoundError"
    assert "lodash" in p.error_message


def test_python_traceback_gives_error_and_module():
    p = parse_log(PYTHON_TRACEBACK)
    assert p.error_type == "ValueError"
    assert p.error_message == "bad amount"
    assert p.module == "services"


def test_pytest_traceback_location_gives_test_file_and_module():
    p = parse_log(PYTEST_LOCATION)
    assert "test_billing" in p.test_name
    assert p.module == "billing"