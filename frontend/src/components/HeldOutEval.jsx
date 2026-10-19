import { useEffect, useState } from "react";
import { apiEvalResults } from "../api";

const pct = (x) => `${Math.round(x * 100)}%`;
const BORDER = "var(--border, rgba(255,255,255,0.1))";

const SYSTEM_TITLES = {
  failure_retrieval: "Similar-failure retrieval (FAISS over past failures)",
  kb_retrieval: "Knowledge-base retrieval",
};

function Stat({ label, value }) {
  return (
    <div
      style={{
        flex: "1 1 130px",
        padding: "0.9rem 1rem",
        border: `1px solid ${BORDER}`,
        borderRadius: "10px",
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "1.6rem", fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: "0.7rem", letterSpacing: "0.08em", color: "var(--text-muted)" }}>
        {label}
      </div>
    </div>
  );
}

function SystemBlock({ name, res }) {
  const cell = { padding: "0.35rem 0.75rem", textAlign: "right" };
  const cur = res.at_current_threshold;
  return (
    <div style={{ marginTop: "1.5rem" }}>
      <div style={{ fontWeight: 600, marginBottom: "0.75rem" }}>
        {SYSTEM_TITLES[name] || name}
      </div>
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
        <Stat label="HIT @ 1" value={pct(res.hit_at_1)} />
        <Stat label="HIT @ 3" value={pct(res.hit_at_3)} />
        <Stat label="MRR" value={res.mrr.toFixed(2)} />
      </div>
      <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.75rem" }}>
        At the current relevance threshold ({res.current_threshold}): correct answers {pct(cur.recall)},
        wrong answers {pct(cur.wrong)}, no-match failures wrongly matched {pct(cur.negative_false_match)}.
        Suggested threshold from the sweep: {res.suggested_threshold}.
      </p>

      <details>
        <summary style={{ cursor: "pointer", fontSize: "0.8rem" }}>Threshold sweep</summary>
        <table style={{ fontSize: "0.8rem", marginTop: "0.5rem", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th style={cell}>Threshold</th>
              <th style={cell}>Correct</th>
              <th style={cell}>Wrong</th>
              <th style={cell}>No-match false match</th>
            </tr>
          </thead>
          <tbody>
            {res.sweep.map((s) => (
              <tr key={s.threshold} style={{ borderTop: `1px solid ${BORDER}` }}>
                <td style={cell}>{s.threshold.toFixed(2)}</td>
                <td style={cell}>{pct(s.recall)}</td>
                <td style={cell}>{pct(s.wrong)}</td>
                <td style={cell}>{pct(s.negative_false_match)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>

      {res.misses.length > 0 && (
        <details style={{ marginTop: "0.5rem" }}>
          <summary style={{ cursor: "pointer", fontSize: "0.8rem" }}>
            Top-1 misses ({res.misses.length})
          </summary>
          <ul style={{ fontSize: "0.8rem", marginTop: "0.5rem", paddingLeft: "1.2rem" }}>
            {res.misses.map((m) => (
              <li key={m.id}>
                {m.id}: expected {m.gold}, got {m.predicted} (score {m.top1_score}
                {m.gold_rank ? `, correct answer ranked #${m.gold_rank}` : ", correct answer not in top 5"})
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export default function HeldOutEval() {
  const [data, setData] = useState(undefined); // undefined = loading, null = no results yet

  function load() {
    apiEvalResults().then(setData).catch(() => setData(null));
  }
  useEffect(load, []);

  return (
    <div className="card" style={{ marginBottom: "1.25rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div className="card-title">🎯 Held-out retrieval evaluation</div>
        <button className="btn btn-ghost btn-sm" onClick={load}>↻ Reload</button>
      </div>

      {data === undefined && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}

      {data === null && (
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
          No results yet. In the backend folder run <code>python evaluation/run_eval.py</code>, then reload.
        </p>
      )}

      {data && (
        <>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Index of {data.index_size} logs, queried with {data.test_size} held-out logs the system never
            indexed. Scored against labels, not parsed error types. Synthetic data, so treat these numbers
            as a baseline, not a final result. The cards below are the older self-retrieval check.
          </p>
          {Object.entries(data.systems).map(([name, res]) => (
            <SystemBlock key={name} name={name} res={res} />
          ))}
        </>
      )}
    </div>
  );
}