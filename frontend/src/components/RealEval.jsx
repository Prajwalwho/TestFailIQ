import { useEffect, useState } from "react";
import { apiEvalReal, apiMetrics } from "../api";

const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const BORDER = "var(--border, rgba(255,255,255,0.1))";
const cell = { padding: "0.4rem 0.75rem", textAlign: "right" };
const cellLeft = { ...cell, textAlign: "left" };
const muted = { fontSize: "0.8rem", color: "var(--text-secondary)" };

function wilson(k, n, z = 1.96) {
  if (n === 0) return [0, 0];
  const p = k / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

const LABELS = {
  "embedding / full": "Embeddings · full text (test, module, error, stack)",
  "embedding / no_names": "Embeddings · without test and module names",
  "embedding / no_names_norm": "Embeddings · names removed, numbers normalised",
  "embedding / error_only": "Embeddings · error line only (used by the app)",
  "embedding / raw_head": "Embeddings · raw log head",
  "bm25 / no_names": "BM25 keyword search",
  "hybrid / no_names": "Hybrid (embeddings + BM25)",
  "majority class (baseline)": "Chance (always guess the largest class)",
};

function Section({ title, children }) {
  return (
    <div style={{ marginTop: "1.5rem" }}>
      <div style={{ fontWeight: 600, marginBottom: "0.4rem" }}>{title}</div>
      {children}
    </div>
  );
}

function SystemsTable({ systems }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ fontSize: "0.8rem", borderCollapse: "collapse", width: "100%" }}>
        <thead>
          <tr style={{ color: "var(--text-muted)" }}>
            <th style={cellLeft}>System</th>
            <th style={cell}>Hit@1 (95% CI)</th>
            <th style={cell}>Hit@3</th>
            <th style={cell}>MRR</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(systems).map(([name, r]) => (
            <tr key={name} style={{ borderTop: `1px solid ${BORDER}` }}>
              <td style={cellLeft}>{LABELS[name] || name}</td>
              <td style={cell}>
                {pct(r.hit_at_1)} ({pct(r.hit1_ci?.[0])}–{pct(r.hit1_ci?.[1])})
              </td>
              <td style={cell}>{pct(r.hit_at_3)}</td>
              <td style={cell}>{r.mrr?.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RagBlock({ rag }) {
  const n = rag.failures_scored;
  const rows = [
    ["Plain LLM (no retrieval)", rag.plain_correct],
    ["LLM with retrieved evidence", rag.retrieval_correct],
  ];
  const significant = rag.sign_test_p !== null && rag.sign_test_p < 0.05;
  const ratings = rag.ratings;
  return (
    <>
      <div style={{ overflowX: "auto" }}>
        <table style={{ fontSize: "0.8rem", borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th style={cellLeft}>Condition</th>
              <th style={cell}>Correct failure class</th>
              <th style={cell}>Accuracy (95% CI)</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, k]) => {
              const [lo, hi] = wilson(k, n);
              return (
                <tr key={name} style={{ borderTop: `1px solid ${BORDER}` }}>
                  <td style={cellLeft}>{name}</td>
                  <td style={cell}>{k}/{n}</td>
                  <td style={cell}>{pct(k / n)} ({pct(lo)}–{pct(hi)})</td>
                </tr>
              );
            })}
            <tr style={{ borderTop: `1px solid ${BORDER}`, color: "var(--text-muted)" }}>
              <td style={cellLeft}>Chance (largest class)</td>
              <td style={cell}></td>
              <td style={cell}>{pct(rag.chance_level)}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p style={muted}>
        Model: {rag.model}. Retrieval supplied evidence for {rag.with_evidence} of {n} failures. Paired result:
        retrieval fixed {rag.retrieval_right_plain_wrong} failures the plain model got wrong and broke{" "}
        {rag.plain_right_retrieval_wrong} it got right (sign test p = {rag.sign_test_p}).{" "}
        <strong>
          {significant
            ? "The difference is statistically significant."
            : "Not statistically significant: this experiment cannot tell whether retrieval helps."}
        </strong>
      </p>
      {ratings && (
        <p style={muted}>
          Blind A/B ratings: retrieval preferred {ratings.retrieval_wins} times, plain {ratings.plain_wins} times,{" "}
          {ratings.ties} ties
          {ratings.win_rate_among_decisive !== null && ` (retrieval wins ${pct(ratings.win_rate_among_decisive)} of decisive cases, p = ${ratings.sign_test_p})`}.
        </p>
      )}
    </>
  );
}

function ThresholdBlock({ t }) {
  const kbOff = t.kb_prompt_min > 1;
  return (
    <>
      <p style={muted}>
        Relevance thresholds (tuned on the labeled real failures): similar failures at similarity ≥ {t.similar_min};
        knowledge-base entries {kbOff ? "are not given to the LLM (no threshold reached 40% precision)" : `shown and used at similarity ≥ ${t.kb_prompt_min}`}.
        Confidence in a diagnosis is the measured share of correct top matches at that similarity:
      </p>
      <ul style={{ ...muted, paddingLeft: "1.2rem", margin: "0.25rem 0" }}>
        {(t.calibration || []).filter((_, i) => i % 3 === 0).map((row) => (
          <li key={row.threshold}>
            similarity ≥ {row.threshold}: {pct(row.precision)} correct ({row.answered} answers)
          </li>
        ))}
      </ul>
    </>
  );
}

export default function RealEval() {
  const [data, setData] = useState(undefined); // undefined = loading, null = unavailable
  const [metrics, setMetrics] = useState(null);

  function load() {
    apiEvalReal().then(setData).catch(() => setData(null));
    apiMetrics().then(setMetrics).catch(() => setMetrics(null));
  }
  useEffect(load, []);

  const retrieval = data?.retrieval;
  const kb = data?.kb;
  const rag = data?.rag;
  const thresholds = data?.thresholds;
  const nothing = data && !retrieval && !kb && !rag && !thresholds;

  return (
    <div className="card" style={{ marginBottom: "1.25rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div className="card-title">📈 Results on real CI failures</div>
        <button className="btn btn-ghost btn-sm" onClick={load}>↻ Reload</button>
      </div>

      <p style={muted}>
        Logs from public GitHub Actions runs, labeled by hand into five failure classes. The datasets are small, so
        read the confidence intervals, not just the percentages.
      </p>

      {data === undefined && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {(data === null || nothing) && (
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
          No real-data results found. Run the scripts in <code>backend/evaluation</code> (run_real_eval.py,
          run_real_kb_eval.py, tune_thresholds.py, run_rag_ablation.py).
        </p>
      )}

      {retrieval && (
        <Section title={`Similar-failure retrieval (${retrieval.queries} labeled failures, leave-one-run-out)`}>
          <SystemsTable systems={retrieval.systems} />
        </Section>
      )}
      {kb && (
        <Section title={`Knowledge-base retrieval (${kb.kb_entries} entries, ${kb.queries} labeled failures)`}>
          <SystemsTable systems={kb.systems} />
        </Section>
      )}
      {rag && (
        <Section title="Does retrieval improve the LLM's diagnosis?">
          <RagBlock rag={rag} />
        </Section>
      )}
      {thresholds && (
        <Section title="Thresholds and confidence">
          <ThresholdBlock t={thresholds} />
        </Section>
      )}
      {metrics && (
        <Section title="User feedback">
          <p style={muted}>
            {metrics.total_diagnoses} diagnoses run, {metrics.feedback_correct} marked correct,{" "}
            {metrics.feedback_incorrect} marked incorrect, {metrics.feedback_pending} unrated
            {metrics.feedback_correct + metrics.feedback_incorrect > 0 &&
              ` (resolution rate ${pct(metrics.resolution_rate)})`}
            . Feedback re-orders future matches but never creates evidence.
          </p>
        </Section>
      )}
    </div>
  );
}