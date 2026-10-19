const API_BASE = "http://localhost:8000/api";

export async function apiHealth() {
  const r = await fetch(`${API_BASE}/health`);
  return r.json();
}

export async function apiDiagnose({ log_text, test_name, module, top_k = 5 }) {
  const r = await fetch(`${API_BASE}/diagnose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log_text, test_name, module, top_k }),
  });
  if (!r.ok) throw new Error(`Diagnose failed: ${r.status}`);
  return r.json();
}

export async function apiIngest({ log_text, test_name, module, known_issue_id, resolution }) {
  const r = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ log_text, test_name, module, known_issue_id, resolution }),
  });
  if (!r.ok) throw new Error(`Ingest failed: ${r.status}`);
  return r.json();
}

export async function apiFailures() {
  const r = await fetch(`${API_BASE}/failures`);
  return r.json();
}

export async function apiKnowledge() {
  const r = await fetch(`${API_BASE}/knowledge`);
  return r.json();
}

export async function apiFeedback({ diagnosis_id, feedback }) {
  const r = await fetch(`${API_BASE}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ diagnosis_id, feedback }),
  });
  if (!r.ok) throw new Error(`Feedback failed: ${r.status}`);
  return r.json();
}

export async function apiMetrics() {
  const r = await fetch(`${API_BASE}/metrics`);
  return r.json();
}

export async function apiEvalResults() {
  const r = await fetch(`${API_BASE}/eval-results`);
  if (!r.ok) return null;
  return r.json();
}