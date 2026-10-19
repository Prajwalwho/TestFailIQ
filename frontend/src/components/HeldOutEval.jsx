import { useEffect, useState } from "react";
import { apiEvalResults } from "../api";

const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const BORDER = "var(--border, rgba(255,255,255,0.1))";
const cell = { padding: "0.4rem 0.75rem", textAlign: "right" };
const cellLeft = { ...cell, textAlign: "left" };

const SYSTEM_LABELS = {
  failures_embedding: "Past failures · embeddings",
  failures_bm25: "Past failures · BM25 keywords",
  failures_hybrid: "Past failures · hybrid",
  kb_embedding: "Knowledge base · embeddings",
  kb_bm25: "Knowledge base · BM25 keywords",
  kb_hybrid: "Knowledge base · hybrid",
};
const SET_TITLES = {
  standard: "Standard held-out set",
  hard: "Hard held-out set (paraphrases and decoys)",
};

function ComparisonTable({ setKey, set }) {
  const first = Object.values(set.systems)[0];
  return (
    <div style={{ marginTop: "1.25rem" }}>
      <div style={{ fontWeight: 600 }}>{SET_TITLES[setKey]}</div>
      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", margin: "0.25rem 0 0.5rem" }}>
        {first.n_positive} labeled failures, {first.n_negative} with no matching known issue
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ fontSize: "0.8rem", borderCollapse: "collapse", width: "100%" }}>
          <thead>
            <tr style={{ color: "var(--text-muted)" }}>
              <th style={cellLeft}>System</th>
              <th style={cell}>Hit@1</th>
              <th style={cell}>Hit@3</th>
              <th style={cell}>MRR</th>
              <th style={cell}>Match vs no-match</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(set.systems).map(([name, r]) => (
              <tr key={name} style={{ borderTop: `1px solid ${BORDER}` }}>
                <td style={cellLeft}>{SYSTEM_LABELS[name] || name}</td>
                <td style={cell}>{pct(r.hit_at_1)}</td>
                <td style={cell}>{pct(r.hit_at_3)}</td>
                <td style={cell}>{r.mrr.toFixed(2)}</td>
                <td style={cell}>{pct(r.separation_auc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TuningBlock({ name, res }) {
  const cur = res.at_current_threshold;
  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ fontSize: "0.85rem", fontWeight: 600 }}>{SYSTEM_LABELS[name]}</div>
      <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", margin: "0.25rem 0" }}>
        Threshold {res.current_threshold}: correct {pct(cur.recall)}, wrong {pct(cur.wrong)},
        no-match failures wrongly matched {pct(cur.negative_false_match)}.
        Suggested: {res.suggested_threshold}.
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
    </div>
  );
}

function MissList({ set }) {
  const rows = Object.entries(set.systems).filter(([, r]) => r.misses.length > 0);
  if (rows.length === 0) return null;
  return (
    <details style={{ marginTop: "0.75rem" }}>
      <summary style={{ cursor: "pointer", fontSize: "0.8rem" }}>Hard-set top-1 misses</summary>
      {rows.map(([name, r]) => (
        <div key={name} style={{ fontSize: "0.8rem", marginTop: "0.5rem" }}>
          <strong>{SYSTEM_LABELS[name]}</strong>
          <ul style={{ paddingLeft: "1.2rem", margin: "0.25rem 0" }}>
            {r.misses.map((m) => (
              <li key={m.id}>
                {m.id}: expected {m.gold}, got {m.predicted} (score {m.top1_score}
                {m.gold_rank ? `, correct answer ranked #${m.gold_rank}` : ", correct answer not in top 5"})
              </li>
            ))}
          </ul>
        </div>
      ))}
    </details>
  );
}

export default function HeldOutEval() {
  const [data, setData] = useState(undefined); // undefined = loading, null = no results yet

  function load() {
    apiEvalResults().then(setData).catch(() => setData(null));
  }
  useEffect(load, []);

  const valid = data && data.test_sets;

  return (
    <div className="card" style={{ marginBottom: "1.25rem" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div className="card-title">🎯 Held-out retrieval evaluation</div>
        <button className="btn btn-ghost btn-sm" onClick={load}>↻ Reload</button>
      </div>

      {data === undefined && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}

      {(data === null || (data && !valid)) && (
        <p style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
          No results yet. In the backend folder run <code>python evaluation/run_eval.py</code>, then reload.
        </p>
      )}

      {valid && (
        <>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Index of {data.index_size} logs. Test logs were never indexed and are scored against labels.
            "Match vs no-match" is how often a real match outscores a failure with no known issue
            (50% = coin flip, 100% = perfect); it decides whether the system can say "I don't know".
            Synthetic data: a baseline, not a final result. The cards below are the older self-retrieval check.
          </p>
          <ComparisonTable setKey="standard" set={data.test_sets.standard} />
          <ComparisonTable setKey="hard" set={data.test_sets.hard} />
          <MissList set={data.test_sets.hard} />

          <div style={{ marginTop: "1.5rem", fontWeight: 600 }}>Relevance thresholds (standard + hard)</div>
          <TuningBlock name="failures_embedding" res={data.test_sets.combined.systems.failures_embedding} />
          <TuningBlock name="kb_embedding" res={data.test_sets.combined.systems.kb_embedding} />
        </>
      )}
    </div>
  );
}