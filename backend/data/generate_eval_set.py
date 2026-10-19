"""
Labeled evaluation set generator.

Builds eval_set.json: synthetic Java/Maven failure logs, each labeled with the
KB entry that SHOULD match (known_issue_id), or None for failures that are not in
the knowledge base.

Split design: for every KB issue there are 5 wordings. Wordings 0-2 go to the
"index" split (the system's memory); wordings 3-4 go to the "test" split and use
different class names and wording, so retrieval on them is genuinely held out.
Six extra "no match" failures go to the test split.

Run from the backend folder:  python data/generate_eval_set.py
"""

import json
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "eval_set.json")
rng = random.Random(42)

# issue id -> (package, 5 class names, 5 method names, 5 (exception, message, extra) wordings)
ISSUES = {
    "KB-001": (
        "service",
        ["UserService", "AccountService", "ProfileService", "CustomerService", "MemberService"],
        ["findById", "getAccount", "loadProfile", "findCustomer", "lookupMember"],
        [
            ("java.lang.NullPointerException",
             'Cannot invoke "User.getEmail()" because the return value of "UserRepository.findById(long)" is null', ""),
            ("java.lang.NullPointerException", "null",
             "findById returned null for an unknown id; no Optional guard"),
            ("java.lang.NullPointerException", "Cannot invoke method getName() on null object", ""),
            ("java.util.NoSuchElementException", "No value present",
             "Optional.get() called without an isPresent() check"),
            ("java.lang.NullPointerException", 'Cannot read field "email" because "user" is null',
             "lookup of a missing user returned null"),
        ],
    ),
    "KB-002": (
        "db",
        ["DatabaseIntegration", "UserDao", "InventoryRepository", "BillingDao", "SessionStore"],
        ["connect", "save", "query", "load", "open"],
        [
            ("com.zaxxer.hikari.pool.HikariPool$PoolInitializationException",
             "Failed to initialize pool: Connection timeout",
             "Timeout waiting for connection from pool (connectionTimeout=30000ms)"),
            ("java.sql.SQLTransientConnectionException",
             "HikariPool-1 - Connection is not available, request timed out after 30000ms",
             "pool max=5 active=5 idle=0 waiting=3"),
            ("org.springframework.jdbc.CannotGetJdbcConnectionException",
             "Failed to obtain JDBC Connection",
             "connection pool exhausted during parallel test run"),
            ("java.sql.SQLTimeoutException", "Timeout waiting for idle object in pool",
             "connections opened in tests were never closed"),
            ("org.hibernate.exception.JDBCConnectionException", "Unable to acquire JDBC Connection",
             "all 5 pooled connections in use; leak suspected in test setup"),
        ],
    ),
    "KB-003": (
        "payment",
        ["PaymentService", "RefundCalculator", "TaxService", "InvoiceTotals", "PriceEngine"],
        ["calculateTotal", "computeRefund", "applyTax", "sumLines", "quote"],
        [
            ("org.opentest4j.AssertionFailedError", "expected: <10.005> but was: <10.004999999999999>", ""),
            ("org.opentest4j.AssertionFailedError", "expected: <49.99> but was: <49.989999999999995>",
             "double arithmetic used for a currency amount"),
            ("java.lang.AssertionError", "Expected 19.99 but got 19.990000000000002",
             "floating-point precision mismatch"),
            ("java.lang.AssertionError", "Expecting actual: 0.30000000000000004 to be equal to: 0.3",
             "rounding issue in total calculation"),
            ("org.opentest4j.AssertionFailedError", "expected: <100.10> but was: <100.09999999999999>",
             "tax computed with double instead of BigDecimal"),
        ],
    ),
    "KB-004": (
        "auth",
        ["AuthService", "TokenValidation", "JwtSigner", "SessionAuth", "ApiKeyService"],
        ["validateToken", "verifyExpired", "signPayload", "authenticate", "checkKey"],
        [
            ("java.util.concurrent.TimeoutException", "Test timed out after 30 seconds",
             "KeyPairGenerator RSA-4096 initialized in @BeforeEach, avg 812ms/test"),
            ("java.util.concurrent.TimeoutException", "Test timed out after 30000ms",
             "RSA-4096 key initialized per-test in @BeforeEach, took 841ms average"),
            ("java.util.concurrent.TimeoutException", "execution timed out after 20 seconds",
             "a new key pair is generated for every test method"),
            ("java.util.concurrent.TimeoutException", "JWT verification exceeded 15000 ms",
             "slow RSA key generation repeated before each test"),
            ("java.util.concurrent.TimeoutException", "Timeout waiting for token validation",
             "crypto setup in @BeforeEach is too slow with 4096-bit keys"),
        ],
    ),
    "KB-005": (
        "order",
        ["OrderProcessor", "BatchRunner", "CartService", "ShipmentPlanner", "InvoiceBuilder"],
        ["process", "runBatch", "checkout", "plan", "build"],
        [
            ("java.lang.IndexOutOfBoundsException", "Index 0 out of bounds for length 0", ""),
            ("java.lang.IndexOutOfBoundsException", "Index: 0, Size: 0",
             "mock returned an empty list; no empty-list guard"),
            ("java.lang.ArrayIndexOutOfBoundsException", "Index 0 out of bounds for length 0",
             "orders.get(0) called on empty input"),
            ("java.lang.IndexOutOfBoundsException", "Index 0 out of bounds for length 0",
             "processBatch invoked with zero items"),
            ("java.lang.ArrayIndexOutOfBoundsException", "Index 0 out of bounds for length 0",
             "first element read from an empty array in the test fixture"),
        ],
    ),
    "KB-006": (
        "cache",
        ["CacheManager", "SessionRegistry", "EventBus", "ConnectionTracker", "MetricsCollector"],
        ["evictLRU", "purgeExpired", "publish", "sweep", "flush"],
        [
            ("java.util.ConcurrentModificationException", "", ""),
            ("java.util.ConcurrentModificationException", "", "HashMap modified while iterating during eviction"),
            ("java.util.ConcurrentModificationException", "", "two test threads mutate a shared map concurrently"),
            ("java.util.ConcurrentModificationException", "",
             "listener collection modified during iteration by another thread"),
            ("java.util.ConcurrentModificationException", "",
             "iterator.next() after a concurrent put into a non-thread-safe map"),
        ],
    ),
    "KB-007": (
        "parser",
        ["RecursiveParser", "JsonWalker", "TreeTraverser", "ExpressionEvaluator", "GraphVisitor"],
        ["parse", "walk", "traverse", "evaluate", "visit"],
        [
            ("java.lang.StackOverflowError", "", "... (truncated, 512 identical frames)"),
            ("java.lang.StackOverflowError", "", "parse() recursed 512 times on deeply nested JSON"),
            ("java.lang.StackOverflowError", "", "no depth limit in recursive descent"),
            ("java.lang.StackOverflowError", "", "nested fixture of depth 2000 exhausted the thread stack"),
            ("java.lang.StackOverflowError", "", "expression nesting of 1500 levels overflowed the call stack"),
        ],
    ),
    "KB-008": (
        "report",
        ["ReportGenerator", "InvoiceGenerator", "ExportService", "PdfWriter", "AuditLogger"],
        ["writeToDisk", "exportCsv", "exportAll", "render", "append"],
        [
            ("java.io.FileNotFoundException",
             "/workspace/reports/monthly_report.pdf (No such file or directory)", ""),
            ("java.io.FileNotFoundException",
             "/tmp/ci-workspace/invoices/invoice_2024.csv (No such file or directory)",
             "directory not created before test run"),
            ("java.nio.file.NoSuchFileException", "/build/output/summary.html",
             "output directory missing in CI workspace"),
            ("java.io.FileNotFoundException", "target/exports/q3/export.xlsx (No such file or directory)",
             "parent folder does not exist on a clean checkout"),
            ("java.nio.file.NoSuchFileException", "/home/runner/work/app/reports/audit.log",
             "report path assumed to exist but CI did not create it"),
        ],
    ),
    "KB-009": (
        "plugin",
        ["PluginLoader", "ExtensionRegistry", "ModuleLoader", "DriverAdapter", "HandlerFactory"],
        ["load", "register", "loadModule", "adapt", "create"],
        [
            ("java.lang.ClassCastException",
             "class com.external.MyPlugin cannot be cast to class com.example.plugin.PluginInterface",
             "PluginInterface was updated in v2.3.0; plugin JAR compiled against v2.2.1"),
            ("java.lang.ClassCastException",
             "class com.vendor.ExportExt cannot be cast to class com.example.plugin.Extension",
             "plugin built against an older interface version"),
            ("java.lang.AbstractMethodError",
             "Receiver class com.external.MyPlugin does not define or inherit an implementation of PluginInterface.render()",
             "plugin JAR not recompiled after interface change"),
            ("java.lang.NoSuchMethodError",
             "com.example.plugin.PluginInterface.getVersion()Ljava/lang/String;",
             "external plugin compiled against outdated API"),
            ("java.lang.ClassCastException",
             "com.acme.AuditPlugin cannot be cast to com.example.plugin.PluginInterface",
             "classloader loaded a stale plugin class; version mismatch"),
        ],
    ),
    "KB-010": (
        "notification",
        ["NotificationService", "EmailSender", "WelcomeMailer", "AlertDispatcher", "ReceiptService"],
        ["sendWelcomeEmail", "send", "mailNewUser", "dispatch", "sendReceipt"],
        [
            ("org.springframework.mail.MailSendException",
             "Mail server connection failed; nested exception is java.net.ConnectException: Connection refused",
             "SMTP host: localhost, port: 25 - connection refused"),
            ("java.net.ConnectException", "Connection refused (Connection refused)",
             "no SMTP server listening on localhost:25 in the test environment"),
            ("javax.mail.MessagingException", "Could not connect to SMTP host: localhost, port: 25",
             "JavaMailSender not mocked in test"),
            ("org.springframework.mail.MailSendException",
             "Failed messages: java.net.ConnectException: Connection refused",
             "tests try to send real email"),
            ("jakarta.mail.MessagingException",
             "Could not connect to SMTP host: localhost, port: 25; nested exception is java.net.ConnectException",
             "mail sender hits the network in a unit test"),
        ],
    ),
}

# Failures that are NOT in the knowledge base (expected label: None)
NEGATIVES = [
    ("catalog", "PriceImporter", "parsePrice", "java.lang.NumberFormatException",
     'For input string: "12a"', "price column read from CSV fixture"),
    ("analytics", "ReportAggregator", "loadAll", "java.lang.OutOfMemoryError",
     "Java heap space", "test loaded a 4 GB fixture into memory"),
    ("db", "UserQueryBuilder", "build", "java.sql.SQLSyntaxErrorException",
     "You have an error in your SQL syntax near 'FORM users'", ""),
    ("util", "DateParser", "parse", "java.lang.IllegalArgumentException",
     "Invalid date format: 2024-13-45", ""),
    ("http", "SecureClient", "connect", "javax.net.ssl.SSLHandshakeException",
     "PKIX path building failed: unable to find valid certification path to requested target",
     "test certificate expired"),
    ("util", "ListHelper", "append", "java.lang.UnsupportedOperationException", "",
     "Arrays.asList returned a fixed-size list; add() is not supported"),
]


def build_log(pkg, cls, method, exc, msg, extra):
    test = f"test{method[0].upper()}{method[1:]}Failure"
    kind = "FAILURE" if "Assertion" in exc else "ERROR"
    headline = f"{exc}: {msg}" if msg else exc
    lines = [
        f"[INFO]  Running {cls}Test",
        f"[ERROR] {test} -- Time elapsed: {rng.uniform(0.01, 31):.3f} s <<< {kind}!",
        headline,
        f"    at com.example.{pkg}.{cls}.{method}({cls}.java:{rng.randint(20, 140)})",
        f"    at com.example.{pkg}.{cls}Test.{test}({cls}Test.java:{rng.randint(20, 140)})",
    ]
    if extra:
        lines.append(f"[INFO]  {extra}")
    lines.append(f"[ERROR] Tests run: {rng.randint(2, 8)}, Failures: 1, Errors: 0, Skipped: 0")
    lines.append("[INFO]  BUILD FAILURE")
    return "\n".join(lines)


def main():
    entries = []
    for issue_id, (pkg, classes, methods, variants) in ISSUES.items():
        for i, (exc, msg, extra) in enumerate(variants):
            entries.append({
                "id": f"EVAL-{len(entries) + 1:03d}",
                "split": "index" if i < 3 else "test",
                "known_issue_id": issue_id,
                "log_text": build_log(pkg, classes[i], methods[i], exc, msg, extra),
            })
    for pkg, cls, method, exc, msg, extra in NEGATIVES:
        entries.append({
            "id": f"EVAL-{len(entries) + 1:03d}",
            "split": "test",
            "known_issue_id": None,
            "log_text": build_log(pkg, cls, method, exc, msg, extra),
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    index_n = sum(e["split"] == "index" for e in entries)
    test_pos = sum(1 for e in entries if e["split"] == "test" and e["known_issue_id"])
    test_neg = sum(1 for e in entries if e["split"] == "test" and not e["known_issue_id"])
    print(f"Wrote {len(entries)} logs -> {OUT_PATH}")
    print(f"  index: {index_n} | test: {test_pos + test_neg} ({test_pos} labeled, {test_neg} no-match)")

    # Sanity check: run the real parser over every log and flag anything it can't read
    sys.path.insert(0, os.path.dirname(BASE_DIR))
    from ingestion.log_parser import parse_log

    flagged = 0
    for e, spec in zip(entries, [None] * len(entries)):
        p = parse_log(e["log_text"])
        has_msg = "\n" + p.error_type + ":" in e["log_text"] or (p.error_type + ": ") in e["log_text"]
        if (p.error_type == "UnknownError" or p.module == "Unknown"
                or p.test_name == "UnknownTest"
                or (has_msg and p.error_message == "No message extracted")):
            flagged += 1
            print(f"  FLAGGED {e['id']}: {p.error_type} | {p.module} | {p.test_name} | {p.error_message}")
    print(f"Parser check: {len(entries)} parsed, {flagged} flagged")


if __name__ == "__main__":
    main()