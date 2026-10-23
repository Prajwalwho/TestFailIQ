"""
Log parser: extracts structured fields from raw CI / test-failure logs.

Handles Java (JUnit / Maven / Gradle), Python (pytest), JavaScript (Jest, Vitest, Mocha)
and Go test output, with generic fallbacks. parse_log() tries the pattern groups in a
fixed order and returns the first hit.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# Which text is embedded for similarity search. "error_only" won the real-data evaluation
# (evaluation/run_real_eval.py). After changing this, run ingest_real_logs.py to re-embed the index.
DEFAULT_VARIANT = "error_only"
VARIANTS = ("full", "no_names", "no_names_norm", "error_only", "raw_head")

_NORMALIZERS = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "UUID"),
    (re.compile(r"0x[0-9a-fA-F]+"), "ADDR"),
    (re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.\-$]+){2,}"), "PATH"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "N"),
]


def normalize(text: str) -> str:
    """Replace volatile tokens (ids, addresses, paths, numbers) so similar failures look alike."""
    for pattern, token in _NORMALIZERS:
        text = pattern.sub(token, text)
    return text


@dataclass
class ParsedLog:
    test_name: str
    module: str
    error_type: str
    error_message: str
    stacktrace: str
    raw_log: str
    chunks: list[str] = field(default_factory=list)
    framework: str = "unknown"

    def to_search_text(self, variant: Optional[str] = None) -> str:
        """Compact representation used for embedding."""
        variant = variant or DEFAULT_VARIANT
        error = f"Error: {self.error_type}: {self.error_message}"
        if variant == "error_only":
            return error
        if variant == "raw_head":
            return self.raw_log[:1000]
        if variant in ("no_names", "no_names_norm"):
            text = f"{error}\nStacktrace:\n{self.stacktrace}"
            return normalize(text) if variant == "no_names_norm" else text
        return (
            f"Test: {self.test_name}\n"
            f"Module: {self.module}\n"
            f"{error}\n"
            f"Stacktrace:\n{self.stacktrace}"
        )


# ── Framework detection ──────────────────────────────────────────────────────
def detect_framework(text: str) -> str:
    if re.search(r"> Task :\S+|BUILD FAILED|Gradle Test Run", text):
        return "gradle"
    if re.search(r"\[ERROR\] Failed to execute goal|Tests run:.*Failures|<<< (?:FAILURE|ERROR)!", text):
        return "maven"
    if re.search(r"short test summary info|=+ FAILURES =+|^E[ \t]{2,}\S|Traceback \(most recent", text, re.M):
        return "pytest"
    if re.search(r"^--- FAIL|^FAIL\t|\bgo test\b|^panic: ", text, re.M):
        return "go"
    if re.search(r"^[ \t]*FAIL[ \t]+\S+\.(?:[jt]sx?|mjs|cjs)|Test Suites:|Test Files|\bjest\b|\bvitest\b|● .+ ›",
                 text, re.M):
        return "jest"
    if re.search(r"\bmocha\b|\b\d+ passing\b|\b\d+ failing\b", text, re.I):
        return "mocha"
    return "unknown"


# ── Error type and message ───────────────────────────────────────────────────
_PY_E_LINE = re.compile(
    r"^E[ \t]+((?:[\w.]+\.)?\w*(?:Error|Exception|Exit|Interrupt|Failed|Warning|TIMEOUT))"
    r"(?:[ \t]*:[ \t]*(.*))?$", re.M)
_PY_E_ASSERT = re.compile(r"^E[ \t]+(assert .+)$", re.M)
_PY_SUMMARY = re.compile(
    r"^(?:FAILED|ERROR)[ \t]+\S+[ \t]+-[ \t]+"
    r"(?:((?:[\w.]+\.)?\w*(?:Error|Exception|Failed|TIMEOUT))[ \t]*:[ \t]*)?(.*)$", re.M)
_JEST_EXPECT = re.compile(r"^\s*(expect\(.+?\)\.[\w.]+\(.*?\))", re.M)
_JS_MISSING = re.compile(r"Cannot find module '([^']+)'")
_JAVAC = re.compile(r"\.(?:java|kt|groovy|scala):\d+:\s*error:\s*(.+)$", re.M)
_TSC = re.compile(r"error (TS\d+):\s*(.+)$", re.M)
_HEADLINE = re.compile(
    r"^[ \t]*(?:Exception in thread \"[^\"]*\"[ \t]+)?(?:Caused by:[ \t]*)?"
    r"((?:[\w$]+\.)*[\w$]*(?:Exception|Error))"
    r"(?:[ \t]*\[\w+\])?(?::[ \t]*(.*))?$", re.M)
_GO_PANIC = re.compile(r"^panic:[ \t]*(.+)$", re.M)


def _clean_msg(text) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:300]


def _short_type(name: str) -> str:
    return re.split(r"[.$]", name)[-1] or name


def _next_line(text: str, pos: int) -> str:
    """First meaningful line after pos that is not a stack frame (used when a headline has no message)."""
    for line in text[pos:].splitlines()[1:8]:
        s = line.strip()
        if not s:
            continue
        if re.match(r"at\s", s) or s.startswith("..."):
            return ""
        return _clean_msg(s)
    return ""


def _go_failure(text: str):
    m = _GO_PANIC.search(text)
    if m:
        return "Panic", _clean_msg(m.group(1))
    m = re.search(r"^--- FAIL:[^\n]*\n", text, re.M)
    if m:
        for line in text[m.end():].splitlines()[:8]:
            g = re.search(r"\.go:\d+:\s*(.+)$", line)
            if g:
                return "TestFailure", _clean_msg(g.group(1))
        return "TestFailure", _clean_msg(m.group(0))
    return None


def _generic_failure(text: str):
    m = re.search(r"(\w+(?:Exception|Error|Failure))\b[ \t:]*(.*)", text)
    if m:
        return m.group(1), _clean_msg(m.group(2))
    m = re.search(r"^.*\b(?:FAILED|FAIL|failed|failing)\b.*$", text, re.M)
    if m:
        return "TestFailure", _clean_msg(m.group(0))
    return "UnknownError", ""


def extract_error(text: str, framework: str = "unknown"):
    """Return (error_type, message). Message may be empty."""
    if framework == "go":
        found = _go_failure(text)
        if found:
            return found

    m1, m2 = _PY_E_LINE.search(text), _PY_E_ASSERT.search(text)
    if m1 and (not m2 or m1.start() < m2.start()):
        return _short_type(m1.group(1)), _clean_msg(m1.group(2)) or _next_line(text, m1.end())
    if m2:
        return "AssertionError", _clean_msg(m2.group(1))

    m = _JEST_EXPECT.search(text)
    if m:
        return "AssertionError", _clean_msg(m.group(1))

    m = _JAVAC.search(text)
    if m:
        return "CompilationError", _clean_msg(m.group(1))
    m = _TSC.search(text)
    if m:
        return m.group(1), _clean_msg(m.group(2))

    m = _JS_MISSING.search(text)
    if m:
        return "ModuleNotFoundError", f"Cannot find module '{m.group(1)}'"

    m = _HEADLINE.search(text)
    if m:
        return _short_type(m.group(1)), _clean_msg(m.group(2)) or _next_line(text, m.end())

    m = _PY_SUMMARY.search(text)
    if m:
        message = _clean_msg(m.group(2))
        if m.group(1):
            return _short_type(m.group(1)), message
        return ("AssertionError" if message.startswith("assert") else "TestFailure"), message

    if framework != "go":
        found = _go_failure(text)
        if found:
            return found
    return _generic_failure(text)


def extract_error_type(log_text: str) -> str:
    return extract_error(log_text)[0] or "UnknownError"


def extract_error_message(log_text: str, error_type: str = "") -> str:
    return extract_error(log_text)[1] or "No message extracted"


# ── Test name ────────────────────────────────────────────────────────────────
def _short_node_id(node_id: str) -> str:
    parts = node_id.split("::")
    return "::".join(parts[1:]) if len(parts) > 1 else node_id


def _simple(name: str) -> str:
    return re.split(r"[.$]", name)[-1]


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path)[-1]


def extract_test_name(text: str) -> str:
    m = re.search(r"^FAILED[ \t]+(\S+?::\S+?)(?:[ \t]+-[ \t]|[ \t]*$)", text, re.M)  # pytest summary
    if m:
        return _short_node_id(m.group(1))
    m = re.search(r"^(\S+\.py::\S+)[ \t]+(?:FAILED|ERROR)\b", text, re.M)  # pytest -v
    if m:
        return _short_node_id(m.group(1))
    m = re.search(r"\[gw\d+\][^\n]*?(?:FAILED|ERROR)[ \t]+(\S+::\S+)", text)  # pytest-xdist
    if m:
        return _short_node_id(m.group(1))
    m = re.search(r"^[ \t]*(\w+(?:Tests?|Spec|IT))[ \t]+>[ \t]+(.+?)[ \t]+FAILED[ \t]*$", text, re.M)  # Gradle
    if m:
        return f"{m.group(1)} > {m.group(2)}"
    m = re.search(r"^[ \t]*FAIL[ \t]+\S+[ \t]+>[ \t]+(.+)$", text, re.M)  # Vitest
    if m:
        return m.group(1).strip()
    m = re.search(r"^[ \t]*●[ \t]+(.+?)[ \t]*$", text, re.M)  # Jest
    if m and not m.group(1).lower().startswith("test suite failed"):
        return m.group(1).strip()
    m = re.search(r"^--- FAIL:[ \t]+(\S+)", text, re.M)  # Go
    if m:
        return m.group(1)
    m = re.search(r"^_{3,}[ \t]*(.+?)[ \t]*_{3,}[ \t]*$", text, re.M)  # pytest failure header
    if m:
        return m.group(1).strip()
    m = re.search(r"\[ERROR\][ \t]+(\w+)\(([\w.$]+)\)[ \t]+Time elapsed", text)  # Surefire per test
    if m:
        return f"{_simple(m.group(2))}.{m.group(1)}"
    m = re.search(r"^\[ERROR\][ \t]+([\w.$]+)\.(\w+):\d+", text, re.M)  # Surefire failure summary
    if m:
        return f"{_simple(m.group(1))}.{m.group(2)}"
    m = re.search(r"(test\w+|Test\w+)\s*--", text)
    if m:
        return m.group(1)
    m = re.search(r"Running (\w+Test)", text)
    if m:
        return m.group(1)
    m = re.search(r"^[ \t]*FAIL[ \t]+(\S+\.(?:[jt]sx?|mjs|cjs))", text, re.M)  # Jest: file only
    if m:
        return _basename(m.group(1))
    m = re.search(r"^[ \t]*\d+\)[ \t]+(.+)$", text, re.M)  # Mocha
    if m:
        return m.group(1).strip()
    m = re.search(r"^(\S*test\S*\.py):\d+:", text, re.M)  # pytest traceback location
    if m:
        return _basename(m.group(1))
    m = re.search(r"([\w./\\-]*(?:test|spec)[\w.\-]*\.(?:py|js|jsx|ts|tsx|mjs|java|kt|go|rb))", text)
    if m:
        return _basename(m.group(1))
    return "UnknownTest"


# ── Module ───────────────────────────────────────────────────────────────────
_GENERIC_DIRS = {"tests", "test", "__tests__", "spec", "specs", "src", "unit", "integration",
                 "testing", "lib", "pkg", "."}
_FRAMEWORK_FRAME_PREFIXES = ("java.", "javax.", "jdk.", "sun.", "kotlin.", "org.junit", "org.gradle",
                             "org.apache.maven", "org.opentest4j", "org.testng")
_JAVA_FRAME = re.compile(r"^\s*at\s+(?:app//)?([\w.$]+)\.([\w$<>]+)\(", re.M)
_PATH_PATTERNS = [
    r"^(?:FAILED|ERROR)[ \t]+(\S+?\.py)",
    r"^(\S+?\.py)::\S+[ \t]+(?:FAILED|ERROR)\b",
    r"\[gw\d+\][^\n]*?(?:FAILED|ERROR)[ \t]+(\S+?\.py)",
    r"test module '([^']+\.py)'",
    r"^[ \t]*FAIL[ \t]+(\S+\.(?:[jt]sx?|mjs|cjs))",
]
_THIRD_PARTY_PATH = re.compile(r"site-packages|node_modules|node:|/lib/python|\\Lib\\|dist-packages")


def _module_from_path(path: str) -> str:
    parts = [p for p in re.split(r"[\\/]", path) if p and p != "."]
    if not parts:
        return "Unknown"
    for directory in reversed(parts[:-1]):
        if directory.lower() not in _GENERIC_DIRS:
            return directory
    stem = re.sub(r"\.\w+$", "", parts[-1])
    stem = re.sub(r"^test_|_test$|\.test$|\.spec$", "", stem)
    return stem or "Unknown"


def _first_project_path(text: str) -> Optional[str]:
    """First project (not library) file named in a traceback or stack frame."""
    patterns = [
        r"^(\S+\.py):\d+:",                                  # pytest traceback location
        r'^\s*File "([^"]+\.py)", line \d+',                  # Python traceback frame
        r"\(([^()\s]+\.(?:[jt]sx?|mjs|cjs)):\d+:\d+\)",       # JavaScript stack frame
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.M):
            if not _THIRD_PARTY_PATH.search(m.group(1)):
                return m.group(1)
    return None


def extract_module(text: str) -> str:
    # 1. com.example convention
    classes = re.findall(r"at com\.example(?:\.\w+)*\.(\w+)\.\w+\(", text)
    app = [c for c in classes if not c.endswith(("Test", "Tests"))]
    if app:
        return app[0]
    if classes:
        return re.sub(r"Tests?$", "", classes[0]) or classes[0]

    # 2. Gradle task path, e.g. "> Task :module:spring-boot-actuator:test FAILED"
    m = re.search(r"> Task :([\w:\-]+) FAILED", text)
    if m:
        parts = [p for p in m.group(1).split(":") if p]
        if parts:
            return parts[-2] if len(parts) >= 2 else parts[0]

    # 3. File paths named in failure lines
    for pattern in _PATH_PATTERNS:
        m = re.search(pattern, text, re.M)
        if m:
            return _module_from_path(m.group(1))

    # 4. Go package
    m = re.search(r"^FAIL[ \t]+(\S+)[ \t]+[\d.]+s", text, re.M)
    if m:
        return m.group(1).rstrip("/").split("/")[-1]

    # 5. First project file in a traceback or stack frame
    path = _first_project_path(text)
    if path:
        return _module_from_path(path)

    # 6. First non-framework Java frame
    for m in _JAVA_FRAME.finditer(text):
        qualified = m.group(1)
        if qualified.startswith(_FRAMEWORK_FRAME_PREFIXES):
            continue
        simple = _simple(qualified)
        if simple and not simple.endswith(("Test", "Tests")):
            return simple
    return "Unknown"


# ── Stack trace ──────────────────────────────────────────────────────────────
def _java_stacktrace(text: str) -> str:
    stack_lines = []
    in_stack = False
    for line in text.splitlines():
        if re.match(r"\s+at ", line) or re.match(r"\s+\.\.\.", line):
            in_stack = True
            stack_lines.append(line.strip())
        elif in_stack and line.strip().startswith("Caused by:"):
            stack_lines.append(line.strip())
        elif in_stack and line.strip() == "":
            continue
        elif in_stack:
            break
    return "\n".join(stack_lines[:25])


def _python_stacktrace(text: str) -> str:
    frames = [
        f"{m.group(3)} ({_basename(m.group(1))}:{m.group(2)})"
        for m in re.finditer(r'^\s*File "([^"]+)", line (\d+), in (\S+)', text, re.M)
    ]
    e_lines = [line.strip() for line in text.splitlines() if re.match(r"E[ \t]", line)][:10]
    return "\n".join(frames[-8:] + e_lines)


def extract_stacktrace(text: str) -> str:
    return _java_stacktrace(text) or _python_stacktrace(text) or text[:500]


def chunk_log(log_text: str) -> list[str]:
    """Split a log into header / exception / stacktrace / summary chunks (kept for compatibility)."""
    chunks = []
    header_lines, exception_lines, stack_lines, summary_lines = [], [], [], []
    in_stack = False

    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[INFO]") and not in_stack:
            header_lines.append(stripped)
        elif stripped.startswith("[ERROR]") or "FAILURE" in stripped or "ERROR" in stripped:
            summary_lines.append(stripped)
        elif re.match(r"\s+at ", line) or stripped.startswith("Caused by:"):
            in_stack = True
            stack_lines.append(stripped)
        elif re.match(r"\w+(\.\w+)*(Exception|Error)[^\n]*", stripped):
            exception_lines.append(stripped)
            in_stack = False
        elif in_stack:
            stack_lines.append(stripped)

    if header_lines:
        chunks.append("Header:\n" + "\n".join(header_lines))
    if exception_lines:
        chunks.append("Exception:\n" + "\n".join(exception_lines))
    if stack_lines:
        chunks.append("Stacktrace:\n" + "\n".join(stack_lines[:20]))
    if summary_lines:
        chunks.append("Summary:\n" + "\n".join(summary_lines))
    return chunks or [log_text]


def parse_log(log_text: str, test_name: Optional[str] = None, module: Optional[str] = None) -> ParsedLog:
    """
    Parse a raw CI log string into a structured ParsedLog object.

    Args:
        log_text: Raw log text from CI.
        test_name: Optional override for test name.
        module: Optional override for module name.
    """
    framework = detect_framework(log_text)
    error_type, error_message = extract_error(log_text, framework)
    return ParsedLog(
        test_name=test_name or extract_test_name(log_text),
        module=module or extract_module(log_text),
        error_type=error_type or "UnknownError",
        error_message=error_message or "No message extracted",
        stacktrace=extract_stacktrace(log_text),
        raw_log=log_text,
        chunks=chunk_log(log_text),
        framework=framework,
    )