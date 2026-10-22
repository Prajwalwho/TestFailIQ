from ingestion.log_parser import VARIANTS, normalize, parse_log

PYTEST_ASSERT = """\
=================================== FAILURES ===================================
______________________________ TestCart.test_total ______________________________

self = <test_cart.TestCart object at 0x7f3a1c>

    def test_total(self):
>       assert cart.total() == 11
E       assert 10 == 11
E        +  where 10 = <bound method total>()

tests/test_cart.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_cart.py::TestCart::test_total - assert 10 == 11
"""

PYTEST_VALUE_ERROR = """\
____________________________ test_parse _____________________________

    def test_parse():
>       parse("x")
E       ValueError: Array indexing is only allowed when arrays match

tests/test_x.py:5: ValueError
FAILED tests/test_x.py::test_parse - ValueError: Array indexing is only allowed
"""

PYTEST_IMPORT_ERROR = """\
==================================== ERRORS ====================================
_____________ ERROR collecting tests/components/dali/test_config_flow.py _____________
ImportError while importing test module '/home/runner/work/core/core/tests/components/dali/test_config_flow.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
E   ModuleNotFoundError: No module named 'dali_ctl'
=========================== short test summary info ============================
ERROR tests/components/dali/test_config_flow.py
"""

VITEST = """\
 FAIL  playground/hmr/__tests__/hmr.spec.ts > import.meta.hot?.accept
AssertionError: expected 'before' to be 'after' // Object.is equality

- Expected: "after"
+ Received: "before"

 ❯ playground/hmr/__tests__/hmr.spec.ts:12:5
"""

JEST_MATCHER = """\
FAIL __tests__/sum.test.js
  ● sum › adds numbers

    expect(received).toBe(expected) // Object.is equality

    Expected: 3
    Received: 2

      at Object.<anonymous> (__tests__/sum.test.js:5:19)
"""

GRADLE_JUNIT = """\
> Task :platform-tests:test

CloseablePathTests > resolvesSymlinkedPaths(Path) FAILED
    org.opentest4j.AssertionFailedError:
    expected: "/var/folders/a/original.jar"
     but was: "/private/var/folders/a/original.jar"
        at app//org.junit.platform.commons.util.CloseablePathTests.resolvesSymlinkedPaths(CloseablePathTests.java:162)

> Task :platform-tests:test FAILED
"""

GRADLE_COMPILE = """\
> Task :documentation:compileToolsJava FAILED
/home/runner/work/junit/junit/documentation/src/tools/java/ApiReportGenerator.java:160: error: cannot find symbol
                        .filter(it -> !it.getAnnotationInfo(API.class).isInherited())
                                          ^
  symbol:   method getAnnotationInfo(Class<API>)
"""

GO_TEST = (
    "=== RUN   TestRender\n"
    "--- FAIL: TestRender (0.00s)\n"
    "    render_test.go:42: expected 200, got 404\n"
    "FAIL\n"
    "FAIL\tgithub.com/gin-gonic/gin/render\t0.012s\n"
)

MOCHA = """\
  1) GET /
       should respond with a cookie:
     Error: expected "Set-Cookie" of "sid=; Max-Age=10", got "sid=; Max-Age=11"
      at error (test/support/utils.js:12:10)
      at Context.<anonymous> (test/res.cookie.js:50:10)
"""

JAVA_DB_TIMEOUT = """\
[INFO]  Running DatabaseIntegrationTest
[INFO]  Initializing Spring TestContext Framework
[ERROR] testSaveAndRetrieveUser -- Time elapsed: 30.001 s <<< ERROR!
com.zaxxer.hikari.pool.HikariPool$PoolInitializationException: Failed to initialize pool: Connection timeout
    at com.zaxxer.hikari.pool.HikariPool.throwPoolInitializationException(HikariPool.java:596)
    at com.zaxxer.hikari.pool.HikariPool.checkFailFast(HikariPool.java:580)
Caused by: java.sql.SQLException: Timeout waiting for connection from pool (connectionTimeout=30000ms)
    at com.example.db.DatabaseIntegrationTest.testSaveAndRetrieveUser(DatabaseIntegrationTest.java:55)
[ERROR] Tests run: 5, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE
"""

JAVA_NPE = """\
[INFO]  Running UserServiceTest
[ERROR] testFindByIdReturnsUser -- Time elapsed: 0.034 s <<< FAILURE!
java.lang.NullPointerException: Cannot invoke method getEmail() on null object
    at com.example.service.UserService.findById(UserService.java:48)
    at com.example.service.UserServiceTest.testFindByIdReturnsUser(UserServiceTest.java:32)
[INFO]  BUILD FAILURE
"""

JAVA_NO_MESSAGE = """\
[INFO]  Running CacheManagerTest
[ERROR] testEvictOnCapacityExceeded -- Time elapsed: 0.221 s <<< ERROR!
java.util.ConcurrentModificationException
    at java.base/java.util.HashMap$HashIterator.nextNode(HashMap.java:1597)
    at com.example.cache.CacheManager.evictLRU(CacheManager.java:78)
"""


def test_pytest_assert():
    p = parse_log(PYTEST_ASSERT)
    assert p.framework == "pytest"
    assert p.error_type == "AssertionError"
    assert "assert 10 == 11" in p.error_message
    assert "test_total" in p.test_name
    assert p.module == "cart"


def test_pytest_value_error():
    p = parse_log(PYTEST_VALUE_ERROR)
    assert p.error_type == "ValueError"
    assert p.error_message.startswith("Array indexing")
    assert p.test_name == "test_parse"


def test_pytest_import_error():
    p = parse_log(PYTEST_IMPORT_ERROR)
    assert p.framework == "pytest"
    assert p.error_type == "ModuleNotFoundError"
    assert "No module named" in p.error_message
    assert p.module == "dali"


def test_vitest():
    p = parse_log(VITEST)
    assert p.framework == "jest"
    assert p.error_type == "AssertionError"
    assert "expected 'before'" in p.error_message
    assert p.test_name == "import.meta.hot?.accept"
    assert p.module == "hmr"


def test_jest_matcher():
    p = parse_log(JEST_MATCHER)
    assert p.framework == "jest"
    assert p.error_type == "AssertionError"
    assert "toBe" in p.error_message
    assert "adds numbers" in p.test_name


def test_gradle_junit_failure():
    p = parse_log(GRADLE_JUNIT)
    assert p.framework == "gradle"
    assert p.error_type == "AssertionFailedError"
    assert "expected" in p.error_message
    assert p.test_name == "CloseablePathTests > resolvesSymlinkedPaths(Path)"
    assert p.module == "platform-tests"


def test_gradle_compile_error():
    p = parse_log(GRADLE_COMPILE)
    assert p.error_type == "CompilationError"
    assert p.error_message == "cannot find symbol"
    assert p.module == "documentation"


def test_go_test_failure():
    p = parse_log(GO_TEST)
    assert p.framework == "go"
    assert p.error_type == "TestFailure"
    assert "expected 200" in p.error_message
    assert p.test_name == "TestRender"
    assert p.module == "render"


def test_mocha_error():
    p = parse_log(MOCHA)
    assert p.error_type == "Error"
    assert p.error_message.startswith("expected")
    assert p.test_name == "GET /"


def test_java_db_timeout_matches_original_behaviour():
    p = parse_log(JAVA_DB_TIMEOUT)
    assert p.framework == "maven"
    assert p.error_type == "PoolInitializationException"
    assert p.error_message == "Failed to initialize pool: Connection timeout"
    assert p.test_name == "testSaveAndRetrieveUser"
    assert p.module == "DatabaseIntegration"


def test_java_npe_module_is_first_non_test_class():
    p = parse_log(JAVA_NPE)
    assert p.error_type == "NullPointerException"
    assert p.module == "UserService"
    assert p.test_name == "testFindByIdReturnsUser"


def test_java_exception_without_message():
    p = parse_log(JAVA_NO_MESSAGE)
    assert p.error_type == "ConcurrentModificationException"
    assert p.error_message == "No message extracted"


def test_unparseable_logs_do_not_crash():
    for text in ("hello world", ""):
        p = parse_log(text)
        assert p.error_type == "UnknownError"
        assert p.error_message == "No message extracted"
        assert p.test_name == "UnknownTest"
        assert p.module == "Unknown"


def test_overrides_win():
    p = parse_log(JAVA_NPE, test_name="MyTest", module="MyModule")
    assert p.test_name == "MyTest"
    assert p.module == "MyModule"


def test_search_text_variants():
    p = parse_log(JAVA_NPE)
    assert set(VARIANTS) == {"full", "no_names", "no_names_norm", "error_only", "raw_head"}
    assert p.to_search_text() == p.to_search_text("error_only")
    assert p.to_search_text("not-a-variant") == p.to_search_text("full")
    assert "Test:" in p.to_search_text("full") and "Module:" in p.to_search_text("full")
    no_names = p.to_search_text("no_names")
    assert "Test:" not in no_names and "Module:" not in no_names and "Error:" in no_names
    assert p.to_search_text("error_only") == (
        "Error: NullPointerException: Cannot invoke method getEmail() on null object")
    assert "48" not in p.to_search_text("no_names_norm")
    assert p.to_search_text("raw_head") == p.raw_log[:1000]


def test_normalize():
    out = normalize("line 42 at /home/a/b.py 0x7f12 id 123e4567-e89b-12d3-a456-426614174000")
    assert out == "line N at PATH ADDR id UUID"