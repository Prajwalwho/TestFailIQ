import { useEffect, useState } from "react";
import "./index.css";
import LogInput from "./components/LogInput";
import DiagnosisCard from "./components/DiagnosisCard";
import SimilarFailures from "./components/SimilarFailures";
import EvalDashboard from "./components/EvalDashboard";
import KnowledgeView from "./components/KnowledgeView";
import { apiDiagnose, apiIngest, apiHealth, apiFeedback } from "./api";

function Toast({ toast }) {
  if (!toast) return null;
  return (
    <div className={`toast toast-${toast.type}`}>
      {toast.type === "success" ? "✓" : "✕"} {toast.message}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("diagnose");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [feedbackGiven, setFeedbackGiven] = useState(null);

  // Check backend health
  useEffect(() => {
    apiHealth()
      .then(() => setBackendStatus("online"))
      .catch(() => setBackendStatus("offline"));
  }, []);

  function showToast(message, type = "success") {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3500);
  }

  async function handleDiagnose(params) {
    setLoading(true);
    setResult(null);
    setFeedbackGiven(null);
    try {
      const data = await apiDiagnose(params);
      setResult(data);
      setTab("diagnose");
      showToast("Diagnosis complete!", "success");
    } catch (e) {
      showToast(`Error: ${e.message}`, "error");
    } finally {
      setLoading(false);
    }
  }

  async function handleIngest(params) {
    setLoading(true);
    try {
      const data = await apiIngest(params);
      showToast(`Ingested as ${data.id}`, "success");
    } catch (e) {
      showToast(`Error: ${e.message}`, "error");
    } finally {
      setLoading(false);
    }
  }

  async function handleFeedback(feedbackValue) {
    if (!result?.diagnosis_id) {
      showToast("Run a diagnosis first.", "error");
      return;
    }
    try {
      await apiFeedback({ diagnosis_id: result.diagnosis_id, feedback: feedbackValue });
      setFeedbackGiven(feedbackValue);
      showToast(`Feedback "${feedbackValue}" recorded for ${result.diagnosis_id}`, "success");
    } catch (e) {
      showToast(`Feedback error: ${e.message}`, "error");
    }
  }

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="header-logo">
            <div className="logo-icon">🧪</div>
            <div>
              <span className="logo-text">TestFailIQ</span>
              <span className="logo-sub">AI-powered test failure analysis</span>
            </div>
          </div>

          <nav className="header-nav">
            {[
              { key: "diagnose", label: "🔍 Diagnose" },
              { key: "knowledge", label: "📚 Knowledge Base" },
              { key: "eval", label: "📊 Evaluation" },
            ].map(({ key, label }) => (
              <button
                key={key}
                className={`nav-btn ${tab === key ? "active" : ""}`}
                onClick={() => setTab(key)}
              >
                {label}
              </button>
            ))}
          </nav>

          <div className="status-dot">
            <div className={`dot ${backendStatus === "offline" ? "offline" : ""}`} />
            {backendStatus === "online"
              ? "Backend online"
              : backendStatus === "offline"
              ? "Backend offline"
              : "Connecting…"}
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="main-content">
        {/* ── Diagnose tab ── */}
        {tab === "diagnose" && (
          <>
            <LogInput onDiagnose={handleDiagnose} onIngest={handleIngest} loading={loading} />

            {loading && (
              <div className="page-loading" style={{ minHeight: "20vh" }}>
                <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
                <span style={{ color: "var(--text-secondary)" }}>
                  Running RAG pipeline — retrieving evidence and generating diagnosis…
                </span>
              </div>
            )}

            {result && !loading && (
              <>
                <div className="diagnosis-section">
                  <DiagnosisCard result={result} />
                  <SimilarFailures evidence={result.evidence} />
                </div>

                {/* Feedback row */}
                <div className="card">
                  <div className="feedback-row">
                    <span>Was this diagnosis helpful?</span>
                    <button
                      className="btn btn-success btn-sm"
                      disabled={feedbackGiven !== null}
                      onClick={() => handleFeedback("correct")}
                    >
                      👍 Yes, correct
                    </button>
                    <button
                      className="btn btn-danger btn-sm"
                      disabled={feedbackGiven !== null}
                      onClick={() => handleFeedback("incorrect")}
                    >
                      👎 No, incorrect
                    </button>
                    <span style={{ marginLeft: "auto", fontSize: "0.75rem", color: "var(--text-muted)" }}>
                      {feedbackGiven
                        ? `Recorded as "${feedbackGiven}" — see the Evaluation tab`
                        : "Feedback is tracked in the Evaluation tab"}
                    </span>
                  </div>
                </div>
              </>
            )}

            {!result && !loading && (
              <div className="empty-state">
                <div className="empty-state-icon">🧬</div>
                <p className="empty-state-title">Ready to investigate</p>
                <p className="empty-state-sub">
                  Paste a CI log above or pick a quick sample, then hit <strong>Run Diagnosis</strong>.
                </p>
              </div>
            )}
          </>
        )}

        {/* ── Knowledge Base tab ── */}
        {tab === "knowledge" && <KnowledgeView />}

        {/* ── Evaluation tab ── */}
        {tab === "eval" && <EvalDashboard />}
      </main>

      <Toast toast={toast} />
    </div>
  );
}
