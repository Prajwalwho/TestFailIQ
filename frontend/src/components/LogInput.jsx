import { useState } from "react";
import SAMPLE_LOGS from "../sampleLogs";

export default function LogInput({ onDiagnose, onIngest, loading }) {
  const [logText, setLogText] = useState("");
  const [testName, setTestName] = useState("");
  const [module, setModule] = useState("");
  const [mode, setMode] = useState("diagnose"); // "diagnose" | "ingest"

  function loadSample(log) {
    setLogText(log);
  }

  function handleSubmit() {
    if (!logText.trim()) return;
    if (mode === "diagnose") {
      onDiagnose({ log_text: logText, test_name: testName || undefined, module: module || undefined });
    } else {
      onIngest({ log_text: logText, test_name: testName || undefined, module: module || undefined });
    }
  }

  return (
    <div className="card log-input-section">
      <div className="card-title">
        <span>🔍</span> CI / Test Log Input
      </div>

      {/* Mode toggle */}
      <div className="tabs" style={{ marginBottom: "1rem" }}>
        <button
          className={`tab ${mode === "diagnose" ? "active" : ""}`}
          onClick={() => setMode("diagnose")}
        >
          🧪 Diagnose
        </button>
        <button
          className={`tab ${mode === "ingest" ? "active" : ""}`}
          onClick={() => setMode("ingest")}
        >
          📥 Add to History
        </button>
      </div>

      {/* Sample loaders */}
      <div style={{ marginBottom: "0.75rem" }}>
        <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginRight: "0.5rem" }}>
          Quick load:
        </span>
        <div className="samples-row">
          {SAMPLE_LOGS.map((s) => (
            <button key={s.label} className="sample-btn" onClick={() => loadSample(s.log)}>
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <textarea
        className="log-textarea"
        placeholder="Paste your CI log or test failure output here…&#10;&#10;Example:&#10;[ERROR] testFindById -- Time elapsed: 0.034 s <<< FAILURE!&#10;java.lang.NullPointerException: Cannot invoke method getEmail() on null object&#10;    at com.example.service.UserService.findById(UserService.java:48)"
        value={logText}
        onChange={(e) => setLogText(e.target.value)}
      />

      <div className="input-row">
        <input
          className="input-field"
          placeholder="Test name (optional override)"
          value={testName}
          onChange={(e) => setTestName(e.target.value)}
        />
        <input
          className="input-field"
          placeholder="Module name (optional override)"
          value={module}
          onChange={(e) => setModule(e.target.value)}
        />
      </div>

      <div className="action-row">
        <button
          id="submit-btn"
          className="btn btn-primary btn-lg"
          onClick={handleSubmit}
          disabled={loading || !logText.trim()}
        >
          {loading ? (
            <>
              <div className="spinner" />
              {mode === "diagnose" ? "Diagnosing…" : "Ingesting…"}
            </>
          ) : mode === "diagnose" ? (
            "→ Run Diagnosis"
          ) : (
            "→ Add to Index"
          )}
        </button>

        {logText && (
          <button
            className="btn btn-ghost"
            onClick={() => { setLogText(""); setTestName(""); setModule(""); }}
          >
            Clear
          </button>
        )}

        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginLeft: "auto" }}>
          {logText.split("\n").length} lines
        </span>
      </div>
    </div>
  );
}
