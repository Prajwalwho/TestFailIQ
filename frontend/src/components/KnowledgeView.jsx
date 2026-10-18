import { useEffect, useState } from "react";
import { apiKnowledge } from "../api";

export default function KnowledgeView() {
  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");

  useEffect(() => {
    apiKnowledge()
      .then((d) => setEntries(d.entries || []))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  const filtered = entries.filter((e) => {
    const q = search.toLowerCase();
    return (
      e.title.toLowerCase().includes(q) ||
      e.error_type.toLowerCase().includes(q) ||
      e.module.toLowerCase().includes(q) ||
      e.tags.some((t) => t.includes(q))
    );
  });

  if (loading) {
    return (
      <div className="page-loading">
        <div className="spinner" style={{ width: 36, height: 36, borderWidth: 3 }} />
        <span>Loading knowledge base…</span>
      </div>
    );
  }

  return (
    <div>
      <div className="section-header">
        <h2 className="section-title">📚 Knowledge Base</h2>
        <span style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
          {entries.length} known issues
        </span>
      </div>

      <input
        className="input-field"
        style={{ width: "100%", marginBottom: "1.5rem", padding: "0.75rem 1rem" }}
        placeholder="🔍  Search by error type, module, or tag…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        {filtered.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🔎</div>
            <p className="empty-state-title">No matches</p>
          </div>
        ) : (
          filtered.map((e) => (
            <div key={e.id} className="card" style={{ transition: "all 0.2s" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                <div>
                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.4rem" }}>
                    <code style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>{e.id}</code>
                    <span className="badge badge-error">{e.error_type}</span>
                    <span className="badge badge-info">{e.module}</span>
                  </div>
                  <h3 style={{ fontSize: "1rem", fontWeight: 700 }}>{e.title}</h3>
                </div>
              </div>

              <p style={{ fontSize: "0.875rem", color: "var(--text-secondary)", marginBottom: "0.75rem" }}>
                {e.description}
              </p>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "0.75rem" }}>
                <div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.3rem" }}>
                    Root Cause
                  </div>
                  <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>{e.root_cause}</p>
                </div>
                <div>
                  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "0.3rem" }}>
                    Resolution
                  </div>
                  <p style={{ fontSize: "0.8rem", color: "var(--green)" }}>{e.resolution}</p>
                </div>
              </div>

              <pre className="code-block" style={{ fontSize: "0.75rem", marginBottom: "0.75rem" }}>
                {e.fix_snippet}
              </pre>

              <div className="kb-tags">
                {e.tags.map((t) => (
                  <span key={t} className="kb-tag">{t}</span>
                ))}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
