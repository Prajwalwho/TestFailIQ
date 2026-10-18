const SAMPLE_LOGS = [
  {
    label: "NullPointerException",
    log: `[INFO]  Running UserServiceTest
[INFO]  Tests run: 3, Failures: 1, Errors: 0
[ERROR] testFindByIdReturnsUser -- Time elapsed: 0.034 s <<< FAILURE!
java.lang.NullPointerException: Cannot invoke method getEmail() on null object
    at com.example.service.UserService.findById(UserService.java:48)
    at com.example.service.UserServiceTest.testFindByIdReturnsUser(UserServiceTest.java:32)
[ERROR] Tests run: 3, Failures: 1, Errors: 0, Skipped: 0
[INFO]  BUILD FAILURE`,
  },
  {
    label: "DB Connection Timeout",
    log: `[INFO]  Running DatabaseIntegrationTest
[ERROR] testSaveAndRetrieveUser -- Time elapsed: 30.001 s <<< ERROR!
com.zaxxer.hikari.pool.HikariPool$PoolInitializationException: Failed to initialize pool: Connection timeout
Caused by: java.sql.SQLException: Timeout waiting for connection from pool (connectionTimeout=30000ms)
    at com.example.db.DatabaseIntegrationTest.testSaveAndRetrieveUser(DatabaseIntegrationTest.java:55)
[INFO]  BUILD FAILURE`,
  },
  {
    label: "FloatingPoint AssertionError",
    log: `[INFO]  Running PaymentServiceTest
[ERROR] testCalculateTotalWithTax -- Time elapsed: 0.012 s <<< FAILURE!
org.opentest4j.AssertionFailedError: expected: <10.005> but was: <10.004999999999999>
    at org.junit.jupiter.api.AssertEquals.assertEquals(AssertEquals.java:182)
    at com.example.service.PaymentServiceTest.testCalculateTotalWithTax(PaymentServiceTest.java:67)
[INFO]  BUILD FAILURE`,
  },
  {
    label: "StackOverflowError",
    log: `[INFO]  Running RecursiveParserTest
[ERROR] testParseDeepNestedJson -- Time elapsed: 0.198 s <<< ERROR!
java.lang.StackOverflowError
    at com.example.parser.RecursiveParser.parse(RecursiveParser.java:55)
    at com.example.parser.RecursiveParser.parse(RecursiveParser.java:67)
    at com.example.parser.RecursiveParser.parse(RecursiveParser.java:67)
    ... (truncated, 512 identical frames)
[INFO]  BUILD FAILURE`,
  },
  {
    label: "SMTP ConnectException",
    log: `[INFO]  Running NotificationServiceTest
[ERROR] testSendWelcomeEmail -- Time elapsed: 10.003 s <<< ERROR!
org.springframework.mail.MailSendException: Mail server connection failed
    java.net.ConnectException: Connection refused (Connection refused)
    at com.example.notification.NotificationService.sendWelcomeEmail(NotificationService.java:73)
[INFO]  SMTP host: localhost, port: 25 -- connection refused
[INFO]  BUILD FAILURE`,
  },
];

export default SAMPLE_LOGS;
