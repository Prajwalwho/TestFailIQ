"""
Harder held-out set (eval_hard.json): 20 paraphrased failures with different wording,
exception types and class names, plus 8 "decoy" failures that share vocabulary with
known issues (report, payment, db, auth...) but have a different cause.

Run from the backend folder:  python data/generate_hard_eval.py
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from generate_eval_set import build_log  # reuse the log template builder

OUT_PATH = os.path.join(BASE_DIR, "eval_hard.json")

# (issue, package, class, method, exception, message, extra note)
HARD = [
    ("KB-001", "customer", "ContactResolver", "resolve", "java.lang.NullPointerException",
     'Cannot invoke "Contact.getPhone()" because the return value of "ContactRepository.lookup(String)" is null',
     "lookup returns null for an unknown key; caller dereferences it without a check"),
    ("KB-001", "billing", "InvoiceOwnerFinder", "ownerOf", "java.lang.NullPointerException",
     'Cannot read field "id" because "owner" is null',
     "repository returned no row for the given id"),
    ("KB-002", "billing", "LedgerWriter", "commit", "java.sql.SQLTransientConnectionException",
     "HikariPool-2 - Connection is not available, request timed out after 30001ms",
     "total=5, active=5, idle=0, waiting=4"),
    ("KB-002", "orders", "OrderStore", "persist", "org.springframework.transaction.CannotCreateTransactionException",
     "Could not open JPA EntityManager for transaction; nested exception is java.sql.SQLTransientConnectionException: connection pool exhausted",
     "earlier tests in the suite left connections open"),
    ("KB-003", "pricing", "DiscountCalc", "apply", "org.opentest4j.AssertionFailedError",
     "expected: <0.1> but was: <0.09999999999999998>", "float arithmetic used for a money amount"),
    ("KB-003", "ledger", "BalanceChecker", "reconcile", "java.lang.AssertionError",
     "Expected 1234.56 but was 1234.5600000000002", "double used for currency values"),
    ("KB-004", "security", "SignatureVerifier", "verify", "java.util.concurrent.TimeoutException",
     "Test timed out after 25 seconds", "each test regenerates a 4096-bit RSA key pair in its setup method"),
    ("KB-004", "auth", "LoginFlow", "signIn", "java.util.concurrent.TimeoutException",
     "execution exceeded 18000 ms", "key pair generation costs about 800 ms for every test method"),
    ("KB-005", "inventory", "StockPicker", "pickFirst", "java.util.NoSuchElementException",
     "", "getFirst() called on an empty list; no isEmpty() guard"),
    ("KB-005", "shipping", "LabelPrinter", "printAll", "java.lang.IndexOutOfBoundsException",
     "Index 0 out of bounds for length 0", "empty shipment list handed over by the mocked upstream"),
    ("KB-006", "events", "ListenerRegistry", "notifyAll", "java.util.ConcurrentModificationException",
     "", "listener list modified by another thread during notification"),
    ("KB-006", "session", "TokenSweeper", "expire", "java.util.ConcurrentModificationException",
     "", "entry removed from the map while the for-each loop was still running"),
    ("KB-007", "json", "NestedFlattener", "flatten", "java.lang.StackOverflowError",
     "", "recursion without a depth guard on a 3000-level fixture"),
    ("KB-007", "rules", "RuleChainRunner", "step", "java.lang.StackOverflowError",
     "", "self-referencing rule graph recursed until the stack was exhausted"),
    ("KB-008", "export", "CsvExporter", "dump", "java.nio.file.NoSuchFileException",
     "build/out/2024/q3/sales.csv", "output folder was never created on the clean agent"),
    ("KB-008", "docs", "ManualBuilder", "emit", "java.io.FileNotFoundException",
     "/var/ci/artifacts/docs/index.html (No such file or directory)", "artifacts directory missing before the test ran"),
    ("KB-009", "addons", "AddonBootstrap", "init", "java.lang.IncompatibleClassChangeError",
     "Class com.thirdparty.ReportAddon does not implement the requested interface com.example.addons.Addon",
     "addon jar built against API v1 while the host runs v2"),
    ("KB-009", "addons", "AddonRegistry", "bind", "java.lang.NoSuchMethodError",
     "com.example.addons.Addon.describe()Ljava/lang/String;", "stale add-on jar after the interface changed"),
    ("KB-010", "alerts", "PagerNotifier", "notifyOnCall", "java.net.ConnectException",
     "Connection refused: connect", "no SMTP relay is listening on localhost:25 during the unit test"),
    ("KB-010", "accounts", "PasswordResetMailer", "sendLink", "org.springframework.mail.MailSendException",
     "Mail server connection failed", "real mail sender used in the test; nothing listens on port 25"),
]

# Decoys: NOT in the knowledge base, but reuse words like report / payment / db / auth / order / cache
DECOYS = [
    ("report", "ReportScheduler", "fire", "java.lang.IllegalStateException",
     "Scheduler has been shut down", "test did not wait for the executor to finish"),
    ("payment", "PaymentGatewayClient", "charge", "javax.net.ssl.SSLHandshakeException",
     "PKIX path building failed: unable to find valid certification path", "expired test certificate"),
    ("db", "SchemaMigrator", "migrate", "org.flywaydb.core.api.FlywayException",
     "Validate failed: Migration checksum mismatch for migration version 4", "an already-applied migration was edited"),
    ("auth", "PasswordPolicy", "check", "java.lang.IllegalArgumentException",
     "Password must contain at least 12 characters", "test used an 8-character password"),
    ("order", "OrderStatusMapper", "map", "java.lang.IllegalArgumentException",
     "No enum constant com.example.order.Status.PENDING_REVIEW", "enum value renamed in the API"),
    ("cache", "CacheWarmup", "run", "java.lang.OutOfMemoryError",
     "Java heap space", "warmup loaded 3 million entries"),
    ("plugin", "PluginManifestParser", "read", "com.fasterxml.jackson.core.JsonParseException",
     "Unexpected character ('}' (code 125)): was expecting double-quote to start field name",
     "malformed manifest in test resources"),
    ("notification", "TemplateRenderer", "render", "java.lang.IllegalArgumentException",
     "Unknown placeholder {{firstName}}", "template variable missing"),
]


def main():
    entries = []
    for issue, pkg, cls, method, exc, msg, extra in HARD:
        entries.append({
            "id": f"HARD-{len(entries) + 1:03d}", "split": "test", "kind": "paraphrase",
            "known_issue_id": issue, "log_text": build_log(pkg, cls, method, exc, msg, extra),
        })
    for pkg, cls, method, exc, msg, extra in DECOYS:
        entries.append({
            "id": f"HARD-{len(entries) + 1:03d}", "split": "test", "kind": "decoy",
            "known_issue_id": None, "log_text": build_log(pkg, cls, method, exc, msg, extra),
        })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(entries)} logs ({len(HARD)} paraphrase + {len(DECOYS)} decoy) -> {OUT_PATH}")

    # Sanity check: the parser must be able to read every generated log
    sys.path.insert(0, os.path.dirname(BASE_DIR))
    from ingestion.log_parser import parse_log

    flagged = 0
    for e in entries:
        p = parse_log(e["log_text"])
        if p.error_type == "UnknownError" or p.module == "Unknown" or p.test_name == "UnknownTest":
            flagged += 1
            print(f"  FLAGGED {e['id']}: {p.error_type} | {p.module} | {p.test_name}")
    print(f"Parser check: {len(entries)} parsed, {flagged} flagged")


if __name__ == "__main__":
    main()