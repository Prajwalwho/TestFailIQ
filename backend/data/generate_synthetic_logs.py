"""
Synthetic CI log generator.
Produces realistic-looking CI/test failure logs for demo and evaluation purposes.
Run directly: python generate_synthetic_logs.py
"""

import json
import os
import random
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "synthetic_logs")

SCENARIOS = [
    {
        "id": "LOG-001",
        "test_name": "UserServiceTest::testFindByIdReturnsUser",
        "module": "UserService",
        "error_type": "NullPointerException",
        "known_issue_id": "KB-001",
        "log_template": """\
[INFO]  Running UserServiceTest
[INFO]  Tests run: 3, Failures: 1, Errors: 0
[ERROR] testFindByIdReturnsUser -- Time elapsed: 0.034 s <<< FAILURE!
java.lang.NullPointerException: Cannot invoke method getEmail() on null object
    at com.example.service.UserService.findById(UserService.java:48)
    at com.example.service.UserServiceTest.testFindByIdReturnsUser(UserServiceTest.java:32)
    at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.lang.reflect.Method.invoke(Method.java:498)
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE
[INFO]  Total time: 4.21 s"""
    },
    {
        "id": "LOG-002",
        "test_name": "DatabaseIntegrationTest::testSaveAndRetrieveUser",
        "module": "DatabaseConfig",
        "error_type": "ConnectionTimeoutException",
        "known_issue_id": "KB-002",
        "log_template": """\
[INFO]  Running DatabaseIntegrationTest
[INFO]  Initializing Spring TestContext Framework
[ERROR] testSaveAndRetrieveUser -- Time elapsed: 30.001 s <<< ERROR!
com.zaxxer.hikari.pool.HikariPool$PoolInitializationException: Failed to initialize pool: Connection timeout
    at com.zaxxer.hikari.pool.HikariPool.throwPoolInitializationException(HikariPool.java:596)
    at com.zaxxer.hikari.pool.HikariPool.checkFailFast(HikariPool.java:580)
    at com.zaxxer.hikari.pool.HikariPool.<init>(HikariPool.java:115)
Caused by: java.sql.SQLException: Timeout waiting for connection from pool (connectionTimeout=30000ms)
    at com.example.db.DatabaseIntegrationTest.testSaveAndRetrieveUser(DatabaseIntegrationTest.java:55)
[ERROR] Tests run: 5, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-003",
        "test_name": "PaymentServiceTest::testCalculateTotalWithTax",
        "module": "PaymentService",
        "error_type": "AssertionError",
        "known_issue_id": "KB-003",
        "log_template": """\
[INFO]  Running PaymentServiceTest
[ERROR] testCalculateTotalWithTax -- Time elapsed: 0.012 s <<< FAILURE!
org.opentest4j.AssertionFailedError: expected: <10.005> but was: <10.004999999999999>
    at org.junit.jupiter.api.AssertEquals.failNotEqual(AssertEquals.java:197)
    at org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:182)
    at com.example.service.PaymentServiceTest.testCalculateTotalWithTax(PaymentServiceTest.java:67)
[INFO]  Tests run: 4, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE
[INFO]  Total time: 2.88 s"""
    },
    {
        "id": "LOG-004",
        "test_name": "AuthServiceTest::testValidateJwtToken",
        "module": "AuthService",
        "error_type": "TimeoutException",
        "known_issue_id": "KB-004",
        "log_template": """\
[INFO]  Running AuthServiceTest
[ERROR] testValidateJwtToken -- Time elapsed: 32.004 s <<< ERROR!
java.util.concurrent.TimeoutException: Test timed out after 30 seconds
    at org.junit.jupiter.api.extension.TimeoutExtension.evaluate(TimeoutExtension.java:83)
    at com.example.auth.AuthServiceTest.testValidateJwtToken(AuthServiceTest.java:88)
    ... 
Caused by: java.lang.InterruptedException
    at java.lang.Thread.sleep(Native Method)
[INFO]  Note: KeyPairGenerator RSA-4096 initialized in @BeforeEach — avg 812ms/test
[ERROR] Tests run: 2, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-005",
        "test_name": "OrderProcessorTest::testProcessEmptyOrderList",
        "module": "OrderProcessor",
        "error_type": "IndexOutOfBoundsException",
        "known_issue_id": "KB-005",
        "log_template": """\
[INFO]  Running OrderProcessorTest
[ERROR] testProcessEmptyOrderList -- Time elapsed: 0.008 s <<< FAILURE!
java.lang.IndexOutOfBoundsException: Index 0 out of bounds for length 0
    at java.base/java.util.Objects.checkIndex(Objects.java:372)
    at java.base/java.util.ArrayList.get(ArrayList.java:459)
    at com.example.order.OrderProcessor.process(OrderProcessor.java:34)
    at com.example.order.OrderProcessorTest.testProcessEmptyOrderList(OrderProcessorTest.java:42)
[INFO]  Tests run: 6, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-006",
        "test_name": "CacheManagerTest::testEvictOnCapacityExceeded",
        "module": "CacheManager",
        "error_type": "ConcurrentModificationException",
        "known_issue_id": "KB-006",
        "log_template": """\
[INFO]  Running CacheManagerTest
[ERROR] testEvictOnCapacityExceeded -- Time elapsed: 0.221 s <<< ERROR!
java.util.ConcurrentModificationException
    at java.base/java.util.HashMap$HashIterator.nextNode(HashMap.java:1597)
    at java.base/java.util.HashMap$EntryIterator.next(HashMap.java:1630)
    at com.example.cache.CacheManager.evictLRU(CacheManager.java:78)
    at com.example.cache.CacheManagerTest.testEvictOnCapacityExceeded(CacheManagerTest.java:91)
[INFO]  Tests run: 3, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-007",
        "test_name": "RecursiveParserTest::testParseDeepNestedJson",
        "module": "RecursiveParser",
        "error_type": "StackOverflowError",
        "known_issue_id": "KB-007",
        "log_template": """\
[INFO]  Running RecursiveParserTest
[ERROR] testParseDeepNestedJson -- Time elapsed: 0.198 s <<< ERROR!
java.lang.StackOverflowError
    at com.example.parser.RecursiveParser.parse(RecursiveParser.java:55)
    at com.example.parser.RecursiveParser.parse(RecursiveParser.java:67)
    at com.example.parser.RecursiveParser.parse(RecursiveParser.java:67)
    ... (truncated, 512 identical frames)
    at com.example.parser.RecursiveParserTest.testParseDeepNestedJson(RecursiveParserTest.java:29)
[INFO]  Tests run: 4, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-008",
        "test_name": "ReportGeneratorTest::testGeneratePdfReport",
        "module": "ReportGenerator",
        "error_type": "FileNotFoundException",
        "known_issue_id": "KB-008",
        "log_template": """\
[INFO]  Running ReportGeneratorTest
[ERROR] testGeneratePdfReport -- Time elapsed: 0.041 s <<< ERROR!
java.io.FileNotFoundException: /workspace/reports/monthly_report.pdf (No such file or directory)
    at java.base/java.io.FileOutputStream.open0(Native Method)
    at java.base/java.io.FileOutputStream.open(FileOutputStream.java:291)
    at com.example.report.ReportGenerator.writeToDisk(ReportGenerator.java:113)
    at com.example.report.ReportGeneratorTest.testGeneratePdfReport(ReportGeneratorTest.java:58)
[INFO]  Tests run: 2, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-009",
        "test_name": "PluginLoaderTest::testLoadExternalPlugin",
        "module": "PluginLoader",
        "error_type": "ClassCastException",
        "known_issue_id": "KB-009",
        "log_template": """\
[INFO]  Running PluginLoaderTest
[ERROR] testLoadExternalPlugin -- Time elapsed: 0.087 s <<< FAILURE!
java.lang.ClassCastException: class com.external.MyPlugin cannot be cast to class com.example.plugin.PluginInterface
    at com.example.plugin.PluginLoader.load(PluginLoader.java:62)
    at com.example.plugin.PluginLoaderTest.testLoadExternalPlugin(PluginLoaderTest.java:45)
[INFO]  Note: PluginInterface was updated in v2.3.0; plugin JAR compiled against v2.2.1
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-010",
        "test_name": "NotificationServiceTest::testSendWelcomeEmail",
        "module": "NotificationService",
        "error_type": "ConnectException",
        "known_issue_id": "KB-010",
        "log_template": """\
[INFO]  Running NotificationServiceTest
[ERROR] testSendWelcomeEmail -- Time elapsed: 10.003 s <<< ERROR!
org.springframework.mail.MailSendException: Mail server connection failed; nested exception is:
    java.net.ConnectException: Connection refused (Connection refused)
    at org.springframework.mail.javamail.JavaMailSenderImpl.doSend(JavaMailSenderImpl.java:448)
    at com.example.notification.NotificationService.sendWelcomeEmail(NotificationService.java:73)
    at com.example.notification.NotificationServiceTest.testSendWelcomeEmail(NotificationServiceTest.java:38)
[INFO]  SMTP host: localhost, port: 25 — connection refused
[ERROR] Tests run: 2, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    # Variants (slightly different wording, same root causes — tests retrieval robustness)
    {
        "id": "LOG-011",
        "test_name": "UserRepositoryTest::testGetUserProfile",
        "module": "UserService",
        "error_type": "NullPointerException",
        "known_issue_id": "KB-001",
        "log_template": """\
[INFO]  Running UserRepositoryTest
[ERROR] testGetUserProfile -- Time elapsed: 0.029 s <<< FAILURE!
java.lang.NullPointerException: null
    at com.example.service.UserService.getUserProfile(UserService.java:82)
    at com.example.repository.UserRepositoryTest.testGetUserProfile(UserRepositoryTest.java:44)
[INFO]  getUserProfile called findById which returned null — no Optional guard
[ERROR] Tests run: 5, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-012",
        "test_name": "PaymentGatewayTest::testRefundCalculation",
        "module": "PaymentService",
        "error_type": "AssertionError",
        "known_issue_id": "KB-003",
        "log_template": """\
[INFO]  Running PaymentGatewayTest
[ERROR] testRefundCalculation -- Time elapsed: 0.015 s <<< FAILURE!
org.opentest4j.AssertionFailedError:
Expected: 49.99
Actual  : 49.989999999999995
    at com.example.payment.PaymentGatewayTest.testRefundCalculation(PaymentGatewayTest.java:101)
[INFO]  Likely floating-point precision error in double arithmetic
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-013",
        "test_name": "OrderServiceTest::testBulkProcessOrders",
        "module": "OrderProcessor",
        "error_type": "IndexOutOfBoundsException",
        "known_issue_id": "KB-005",
        "log_template": """\
[INFO]  Running OrderServiceTest
[ERROR] testBulkProcessOrders -- Time elapsed: 0.011 s <<< FAILURE!
java.lang.IndexOutOfBoundsException: Index: 0, Size: 0
    at java.util.ArrayList.rangeCheck(ArrayList.java:659)
    at java.util.ArrayList.get(ArrayList.java:435)
    at com.example.order.OrderProcessor.processBatch(OrderProcessor.java:57)
    at com.example.order.OrderServiceTest.testBulkProcessOrders(OrderServiceTest.java:78)
[INFO]  Mock returned empty list; no empty-list guard in processBatch
[ERROR] Tests run: 4, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-014",
        "test_name": "TokenValidationTest::testExpiredTokenRejection",
        "module": "AuthService",
        "error_type": "TimeoutException",
        "known_issue_id": "KB-004",
        "log_template": """\
[INFO]  Running TokenValidationTest
[ERROR] testExpiredTokenRejection -- Time elapsed: 31.008 s <<< ERROR!
java.util.concurrent.TimeoutException: Test timed out after 30000ms
    at com.example.auth.TokenValidationTest.testExpiredTokenRejection(TokenValidationTest.java:55)
[INFO]  RSA-4096 key initialized per-test in @BeforeEach, took 841ms average
[ERROR] Tests run: 3, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
    {
        "id": "LOG-015",
        "test_name": "InvoiceGeneratorTest::testExportInvoiceToCsv",
        "module": "ReportGenerator",
        "error_type": "FileNotFoundException",
        "known_issue_id": "KB-008",
        "log_template": """\
[INFO]  Running InvoiceGeneratorTest
[ERROR] testExportInvoiceToCsv -- Time elapsed: 0.038 s <<< ERROR!
java.io.FileNotFoundException: /tmp/ci-workspace/invoices/invoice_2024.csv (No such file or directory)
    at com.example.report.InvoiceGenerator.exportCsv(InvoiceGenerator.java:88)
    at com.example.report.InvoiceGeneratorTest.testExportInvoiceToCsv(InvoiceGeneratorTest.java:34)
[INFO]  /tmp/ci-workspace/invoices directory not created before test run
[ERROR] Tests run: 2, Failures: 0, Errors: 1, Skipped: 0
[INFO]  BUILD FAILURE"""
    },
]


def generate_logs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata = []
    base_time = datetime(2024, 1, 10, 9, 0, 0)

    for i, scenario in enumerate(SCENARIOS):
        timestamp = base_time + timedelta(hours=i * 6 + random.randint(0, 5))
        record = {
            "id": scenario["id"],
            "test_name": scenario["test_name"],
            "module": scenario["module"],
            "error_type": scenario["error_type"],
            "known_issue_id": scenario["known_issue_id"],
            "timestamp": timestamp.isoformat(),
            "log_text": scenario["log_template"],
            "resolution": None,
        }
        # Write individual log file
        log_path = os.path.join(OUTPUT_DIR, f"{scenario['id']}.log")
        with open(log_path, "w") as f:
            f.write(scenario["log_template"])
        metadata.append(record)
        print(f"  [OK] Generated {scenario['id']} - {scenario['test_name']}")

    # Write metadata index
    meta_path = os.path.join(BASE_DIR, "synthetic_logs_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[DONE] Generated {len(SCENARIOS)} synthetic logs -> {OUTPUT_DIR}")
    print(f"   Metadata index -> {meta_path}")
    return metadata


if __name__ == "__main__":
    generate_logs()
