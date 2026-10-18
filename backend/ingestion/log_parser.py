"""
Log parser and chunker.
Extracts structured fields from raw CI/test log text.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedLog:
    test_name: str
    module: str
    error_type: str
    error_message: str
    stacktrace: str
    raw_log: str
    chunks: list[str] = field(default_factory=list)

    def to_search_text(self) -> str:
        """Compact representation used for embedding."""
        return (
            f"Test: {self.test_name}\n"
            f"Module: {self.module}\n"
            f"Error: {self.error_type}: {self.error_message}\n"
            f"Stacktrace:\n{self.stacktrace}"
        )


def extract_error_type(log_text: str) -> str:
    """Extract the primary exception/error class name."""
    patterns = [
        r"(java\.lang\.\w+(?:Error|Exception))",
        r"(com\.\w+(?:\.\w+)*(?:Exception|Error))",
        r"(org\.\w+(?:\.\w+)*(?:Exception|Error))",
        r"(\w+(?:Exception|Error|Failure))",
    ]
    for pattern in patterns:
        match = re.search(pattern, log_text)
        if match:
            return match.group(1).split(".")[-1]  # Short class name
    return "UnknownError"


def extract_error_message(log_text: str, error_type: str) -> str:
    """Extract the human-readable error message following the exception."""
    pattern = rf"{re.escape(error_type)}[^:]*:\s*(.+)"
    match = re.search(pattern, log_text)
    if match:
        return match.group(1).strip()
    # Fallback: grab the FAILURE!/ERROR! line
    match = re.search(r"<<< (?:FAILURE|ERROR)!\s*\n(.+)", log_text)
    if match:
        return match.group(1).strip()
    return "No message extracted"


def extract_test_name(log_text: str) -> str:
    """Extract the test method name from the log."""
    match = re.search(r"(test\w+|Test\w+)\s*--", log_text)
    if match:
        return match.group(1)
    match = re.search(r"Running (\w+Test)", log_text)
    if match:
        return match.group(1)
    return "UnknownTest"


def extract_module(log_text: str) -> str:
    """Best-effort module extraction from stack trace."""
    match = re.search(r"at com\.example\.(\w+)\.", log_text)
    if match:
        return match.group(1)
    return "Unknown"


def extract_stacktrace(log_text: str) -> str:
    """Extract the stacktrace block from the log."""
    lines = log_text.splitlines()
    stack_lines = []
    in_stack = False
    for line in lines:
        if re.match(r"\s+at ", line) or re.match(r"\s+\.\.\.", line):
            in_stack = True
            stack_lines.append(line.strip())
        elif in_stack and line.strip().startswith("Caused by:"):
            stack_lines.append(line.strip())
        elif in_stack and line.strip() == "":
            continue
        elif in_stack:
            break
    return "\n".join(stack_lines) if stack_lines else log_text[:500]


def chunk_log(log_text: str) -> list[str]:
    """
    Split a log into meaningful chunks:
    - Header (test name + outcome)
    - Primary exception + message
    - Stacktrace
    - Caused-by chain
    - Summary line
    """
    chunks = []
    lines = log_text.splitlines()

    header_lines, exception_lines, stack_lines, summary_lines = [], [], [], []
    in_stack = False

    for line in lines:
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
        else:
            if in_stack:
                stack_lines.append(stripped)

    if header_lines:
        chunks.append("Header:\n" + "\n".join(header_lines))
    if exception_lines:
        chunks.append("Exception:\n" + "\n".join(exception_lines))
    if stack_lines:
        # Limit stacktrace chunk to first 20 frames to avoid huge embeddings
        chunks.append("Stacktrace:\n" + "\n".join(stack_lines[:20]))
    if summary_lines:
        chunks.append("Summary:\n" + "\n".join(summary_lines))

    # If chunking produced nothing useful, fall back to whole log
    if not chunks:
        chunks = [log_text]

    return chunks


def parse_log(log_text: str, test_name: str = None, module: str = None) -> ParsedLog:
    """
    Parse a raw CI log string into a structured ParsedLog object.

    Args:
        log_text: Raw log text from CI.
        test_name: Optional override for test name.
        module: Optional override for module name.
    """
    error_type = extract_error_type(log_text)
    error_message = extract_error_message(log_text, error_type)
    parsed_test_name = test_name or extract_test_name(log_text)
    parsed_module = module or extract_module(log_text)
    stacktrace = extract_stacktrace(log_text)
    chunks = chunk_log(log_text)

    return ParsedLog(
        test_name=parsed_test_name,
        module=parsed_module,
        error_type=error_type,
        error_message=error_message,
        stacktrace=stacktrace,
        raw_log=log_text,
        chunks=chunks,
    )
