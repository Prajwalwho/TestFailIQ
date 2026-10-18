import { useEffect, useState } from "react";
import { apiMetrics, apiFailures } from "../api";

function MetricCard({ value, label, suffix = "", color }) {
  return (
    <div className="metric-card">
      <div className="metric-value" style={color ? { WebkitTextFillColor: color } : {}}>
        {typeof value === "number" && value < 1 && value > 0
          ? `${Math.round(value * 100)}%`
          : value}
        {suffix}
      </div>
      <div className="metric-label">{label}</div>
    </div>
  );
}

export default function EvalDashboard() {
  const [metrics, setMetrics] = useState(null);
  const [failures, setFailures] = useState([]);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    try {
      const [m, f] = await Promise.all([apiMetrics(), apiFailures()]);
      setMetrics(m);
      setFailures(f.failures || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, []);

  const feedbackBadge = (fb) => {
    if (fb === "correct") return <span className="badge badge-success">✓ Correct</span>;
    if (fb === "incorrect") return <span className="badge badge-error">✗ Incorrect</span>;
    return <span className="badge" style={{ background: "var(--border)", color: "var(--text-muted)" }}>Pending</span>;
  };

  if (loading) {
    return (
      <div className="page-loading">
        <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
        <span>Loading metrics…</span>
      </div>
    );
  }

  return (
    <div>
      <div className="section-header">
        <h2 className="section-title">📊 Evaluation Dashboard</h2>
        <button className="btn btn-ghost btn-sm" onClick={load}>
          ↻ Refresh
        </button>
      </div>

      {/* Metric cards */}
      {metrics && (
        <div className="metrics-grid">
          <MetricCard value={metrics.total_indexed} label="Total Indexed" />
          <MetricCard
            value={metrics.retrieval_at_1}
            label="Retrieval @ 1"
          />
          <MetricCard
            value={metrics.retrieval_at_3}
            label="Retrieval @ 3"
          />
          <MetricCard
            value={metrics.resolution_rate}
            label="Resolution Rate"
          />
          <MetricCard value={metrics.feedback_correct} label="Correct Diagnoses" />
          <MetricCard value={metrics.feedback_incorrect} label="Incorrect Diagnoses" />
          <MetricCard value={metrics.feedback_pending} label="Awaiting Feedback" />
        </div>
      )}

      {/* Retrieval quality bar */}
      {metrics && (
        <div className="card" style={{ marginBottom: "1.5rem" }}>
          <div className="card-title">📈 Retrieval Quality</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {[
              { label: "Retrieval @ 1 (top result same error type)", value: metrics.retrieval_at_1 },
              { label: "Retrieval @ 3 (within top 3)", value: metrics.retrieval_at_3 },
              { label: "Resolution Rate (marked correct / total feedback)", value: metrics.resolution_rate },
            ].map(({ label, value }) => {
              const pct = Math.round(value * 100);
              const cls = pct >= 70 ? "confidence-high" : pct >= 40 ? "confidence-mid" : "confidence-low";
              return (
                <div key={label}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", marginBottom: "0.4rem" }}>
                    <span style={{ color: "var(--text-secondary)" }}>{label}</span>
                    <span style={{ color: "var(--text-primary)", fontWeight: 700 }}>{pct}%</span>
                  </div>
                  <div className="confidence-bar">
                    <div className={`confidence-fill ${cls}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Failures table */}
      <div className="card">
        <div className="card-title" style={{ marginBottom: "1rem" }}>
          🗃️ Indexed Failures
          <span className="badge badge-accent" style={{ marginLeft: "auto" }}>
            {failures.length}
          </span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Test</th>
                <th>Module</th>
                <th>Error Type</th>
                <th>Timestamp</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {failures.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: "center", color: "var(--text-muted)", padding: "2rem" }}>
                    No failures indexed yet.
                  </td>
                </tr>
              ) : (
                failures.map((f) => (
                  <tr key={f.id}>
                    <td>
                      <code style={{ fontSize: "0.75rem", color: "var(--accent-light)" }}>{f.id}</code>
                    </td>
                    <td title={f.test_name}>{f.test_name}</td>
                    <td>
                      <span className="badge badge-info" style={{ fontSize: "0.65rem" }}>
                        {f.module}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-error" style={{ fontSize: "0.65rem" }}>
                        {f.error_type}
                      </span>
                    </td>
                    <td>
                      {new Date(f.timestamp).toLocaleDateString("en-US", {
                        month: "short", day: "numeric",
                      })}
                    </td>
                    <td>{feedbackBadge(f.feedback)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
