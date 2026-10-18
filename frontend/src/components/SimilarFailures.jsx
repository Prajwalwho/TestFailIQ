function SimScore({ score }) {
  const pct = Math.round(score * 100);
  const color =
    pct >= 70 ? "var(--green)" : pct >= 40 ? "var(--orange)" : "var(--text-muted)";
  return (
    <span
      className="similarity-pill"
      style={{
        background: `${color}22`,
        color,
        border: `1px solid ${color}55`,
      }}
    >
      {pct}%
    </span>
  );
}

export default function SimilarFailures({ evidence }) {
  if (!evidence) return null;
  const { similar_failures, kb_matches } = evidence;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
      {/* Similar historical failures */}
      <div className="card">
        <div className="card-title">
          🔁 Similar Historical Failures
          <span className="badge badge-accent" style={{ marginLeft: "auto" }}>
            {similar_failures.length}
          </span>
        </div>

        {similar_failures.length === 0 ? (
          <div className="empty-state" style={{ padding: "1.5rem" }}>
            <div className="empty-state-icon">🔍</div>
            <p className="empty-state-title">No similar failures found</p>
            <p className="empty-state-sub">This may be a novel issue.</p>
          </div>
        ) : (
          similar_failures.map((f) => (
            <div key={f.id} className="failure-item">
              <div className="failure-item-top">
                <span className="failure-item-name">{f.test_name}</span>
                <SimScore score={f.similarity} />
              </div>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "0.3rem" }}>
                <span className="badge badge-error" style={{ fontSize: "0.65rem" }}>
                  {f.error_type}
                </span>
                <span className="badge badge-info" style={{ fontSize: "0.65rem" }}>
                  {f.module}
                </span>
              </div>
              <div className="failure-item-meta">
                {f.error_message?.slice(0, 100)}
                {f.error_message?.length > 100 ? "…" : ""}
              </div>
              {f.resolution && (
                <div
                  style={{
                    marginTop: "0.5rem",
                    fontSize: "0.75rem",
                    color: "var(--green)",
                    background: "var(--green-glow)",
                    padding: "0.35rem 0.6rem",
                    borderRadius: "var(--radius-sm)",
                  }}
                >
                  ✓ {f.resolution}
                </div>
              )}
              <div style={{ marginTop: "0.4rem", fontSize: "0.7rem", color: "var(--text-muted)" }}>
                {new Date(f.timestamp).toLocaleDateString("en-US", {
                  month: "short", day: "numeric", year: "numeric",
                })}
              </div>
            </div>
          ))
        )}
      </div>

      {/* KB Matches */}
      <div className="card">
        <div className="card-title">
          📚 Knowledge Base Matches
          <span className="badge badge-info" style={{ marginLeft: "auto" }}>
            {kb_matches.length}
          </span>
        </div>

        {kb_matches.length === 0 ? (
          <div className="empty-state" style={{ padding: "1.5rem" }}>
            <div className="empty-state-icon">📭</div>
            <p className="empty-state-title">No KB matches</p>
          </div>
        ) : (
          kb_matches.map((kb) => (
            <div key={kb.id} className="kb-item">
              <div className="failure-item-top">
                <span className="kb-item-title">{kb.title}</span>
                <SimScore score={kb.similarity} />
              </div>
              <p className="kb-item-rc">
                <strong style={{ color: "var(--text-secondary)" }}>Root cause: </strong>
                {kb.root_cause}
              </p>
              <p className="kb-item-rc">
                <strong style={{ color: "var(--text-secondary)" }}>Fix: </strong>
                {kb.resolution}
              </p>
              {kb.fix_snippet && (
                <pre className="code-block" style={{ marginTop: "0.5rem", fontSize: "0.72rem" }}>
                  {kb.fix_snippet}
                </pre>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
