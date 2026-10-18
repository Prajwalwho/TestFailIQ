function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100);
  const cls =
    pct >= 70 ? "confidence-high" : pct >= 40 ? "confidence-mid" : "confidence-low";
  const label = pct >= 70 ? "High" : pct >= 40 ? "Medium" : "Low";
  const color = pct >= 70 ? "var(--green)" : pct >= 40 ? "var(--orange)" : "var(--red)";

  return (
    <div>
      <div className="confidence-row">
        <span>Confidence</span>
        <span style={{ color, fontWeight: 700 }}>
          {label} — {pct}%
        </span>
      </div>
      <div className="confidence-bar">
        <div className={`confidence-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function DiagnosisCard({ result }) {
  if (!result) return null;

  const { parsed_log, diagnosis } = result;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Parsed log info */}
      <div className="card" style={{ padding: "1rem 1.25rem" }}>
        <div className="card-title" style={{ marginBottom: "0.75rem" }}>
          📋 Parsed Failure
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
          <span className="badge badge-error">⚡ {parsed_log.error_type}</span>
          <span className="badge badge-info">📦 {parsed_log.module}</span>
          <span className="badge badge-accent">🧪 {parsed_log.test_name}</span>
        </div>
        <div style={{ marginTop: "0.75rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
          {parsed_log.error_message}
        </div>
      </div>

      {/* Root Cause */}
      <div className="card">
        <div className="card-title">🎯 Root Cause</div>
        <p className="root-cause-text">{diagnosis.root_cause}</p>
        <p className="explanation-text">{diagnosis.explanation}</p>
        <div className="divider" />
        <ConfidenceBar value={diagnosis.confidence} />
        <div
          style={{
            marginTop: "0.75rem",
            fontSize: "0.75rem",
            color: "var(--text-muted)",
            display: "flex",
            gap: "1rem",
          }}
        >
          <span>Model: <code style={{ color: "var(--accent-light)" }}>{diagnosis.model_used}</code></span>
        </div>
      </div>

      {/* Next Steps */}
      <div className="card">
        <div className="card-title">🛠️ Next Steps</div>
        {diagnosis.next_steps?.length > 0 ? (
          <ul className="steps-list">
            {diagnosis.next_steps.map((step, i) => (
              <li key={i} className="step-item">
                <div className="step-num">{i + 1}</div>
                <span>{step}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
            No next steps available.
          </p>
        )}
      </div>

      {/* Fix Snippet */}
      {diagnosis.fix_snippet && (
        <div className="card">
          <div className="card-title">💊 Fix Snippet</div>
          <pre className="code-block">{diagnosis.fix_snippet}</pre>
        </div>
      )}
    </div>
  );
}
