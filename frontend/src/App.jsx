import { useState, useRef, useEffect, useCallback } from "react";

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8000";

// Icon components (inline SVG, no external deps)
const SignalIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 20h2"/><path d="M7 20v-6"/><path d="M12 20v-10"/><path d="M17 20V4"/><path d="M22 20V2"/>
  </svg>
);
const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);
const BookIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
  </svg>
);
const ChevronIcon = ({ open }) => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
    style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 0.2s" }}>
    <polyline points="6 9 12 15 18 9"/>
  </svg>
);
const TrashIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
  </svg>
);
const AlertIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
  </svg>
);

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse source type from citation string
// e.g. "FAQ #2 [data]" → "faq", "Ticket TK-007 [sim]" → "ticket", else "guide"
// ─────────────────────────────────────────────────────────────────────────────
function sourceType(citation) {
  if (!citation) return "guide";
  const c = citation.toLowerCase();
  if (c.startsWith("faq")) return "faq";
  if (c.startsWith("ticket")) return "ticket";
  return "guide";
}

const SOURCE_LABELS = { faq: "FAQ", ticket: "Ticket", guide: "Guide" };
const SOURCE_COLORS = { faq: "#3b82f6", ticket: "#10b981", guide: "#f59e0b" };

// ─────────────────────────────────────────────────────────────────────────────
// SourceBadge — coloured pill for the sources panel
// ─────────────────────────────────────────────────────────────────────────────
function SourceBadge({ type }) {
  return (
    <span style={{
      fontSize: "10px",
      fontWeight: 700,
      letterSpacing: "0.08em",
      padding: "2px 7px",
      borderRadius: "999px",
      backgroundColor: SOURCE_COLORS[type] + "22",
      color: SOURCE_COLORS[type],
      border: `1px solid ${SOURCE_COLORS[type]}44`,
      textTransform: "uppercase",
      fontFamily: "'IBM Plex Mono', monospace",
    }}>
      {SOURCE_LABELS[type]}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// RelevanceBar — mini horizontal bar showing similarity score
// ─────────────────────────────────────────────────────────────────────────────
function RelevanceBar({ score }) {
  // score is L2 distance (lower = better). Convert to 0–100% display value.
  const pct = Math.max(0, Math.min(100, Math.round((1 - score) * 100)));
  const color = pct > 70 ? "#10b981" : pct > 45 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 4 }}>
      <div style={{ flex: 1, height: 4, background: "#ffffff18", borderRadius: 999, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 999, transition: "width 0.6s ease" }} />
      </div>
      <span style={{ fontSize: 11, color: "#9ca3af", fontFamily: "'IBM Plex Mono', monospace", minWidth: 36 }}>
        {pct}%
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SourcesPanel — collapsible panel listing all citations for a message
// ─────────────────────────────────────────────────────────────────────────────
function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources || sources.length === 0) return null;

  return (
    <div style={{
      marginTop: 10,
      borderRadius: 10,
      border: "1px solid #ffffff18",
      overflow: "hidden",
      background: "#0f172a",
    }}>
      {/* Header (toggle) */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "8px 12px",
          background: "transparent",
          border: "none",
          cursor: "pointer",
          color: "#94a3b8",
          fontSize: 12,
          fontFamily: "'IBM Plex Mono', monospace",
          letterSpacing: "0.04em",
        }}
      >
        <BookIcon />
        <span style={{ flex: 1, textAlign: "left" }}>
          {sources.length} source{sources.length !== 1 ? "s" : ""} retrieved
        </span>
        <ChevronIcon open={open} />
      </button>

      {/* Expanded content */}
      {open && (
        <div style={{ padding: "0 12px 12px" }}>
          {sources.map((src, i) => {
            const type = sourceType(src.citation);
            return (
              <div key={i} style={{
                borderTop: "1px solid #ffffff0d",
                paddingTop: 10,
                marginTop: 10,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <SourceBadge type={type} />
                  <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 600 }}>
                    {src.citation}
                  </span>
                </div>
                <RelevanceBar score={src.score} />
                <p style={{
                  margin: "6px 0 0",
                  fontSize: 11.5,
                  color: "#64748b",
                  lineHeight: 1.6,
                  fontFamily: "'IBM Plex Mono', monospace",
                }}>
                  {src.preview}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MessageBubble — single chat message (user or assistant)
// ─────────────────────────────────────────────────────────────────────────────
function MessageBubble({ msg }) {
  const isUser = msg.role === "user";

  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 16,
      animation: "fadeSlideIn 0.25s ease both",
    }}>
      {/* Assistant avatar */}
      {!isUser && (
        <div style={{
          width: 34, height: 34, borderRadius: "50%",
          background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
          display: "flex", alignItems: "center", justifyContent: "center",
          marginRight: 10, flexShrink: 0, marginTop: 2,
          boxShadow: "0 0 12px #3b82f640",
        }}>
          <SignalIcon />
        </div>
      )}

      <div style={{ maxWidth: "72%", minWidth: 60 }}>
        {/* Bubble */}
        <div style={{
          padding: "11px 16px",
          borderRadius: isUser ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
          background: isUser
            ? "linear-gradient(135deg, #3b82f6, #6366f1)"
            : "#1e293b",
          color: "#f1f5f9",
          fontSize: 14.5,
          lineHeight: 1.65,
          boxShadow: isUser
            ? "0 4px 20px #3b82f640"
            : "0 2px 12px #00000030",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          border: isUser ? "none" : "1px solid #ffffff0f",
        }}>
          {msg.content}
          {/* Streaming cursor */}
          {msg.streaming && (
            <span style={{
              display: "inline-block",
              width: 8, height: 14,
              background: "#3b82f6",
              marginLeft: 3,
              borderRadius: 2,
              animation: "blink 0.8s step-end infinite",
              verticalAlign: "text-bottom",
            }} />
          )}
          {/* Fallback / error badge */}
          {msg.isFallback && (
            <div style={{
              marginTop: 10,
              padding: "6px 10px",
              background: "#ef444420",
              border: "1px solid #ef444440",
              borderRadius: 8,
              fontSize: 12,
              color: "#fca5a5",
              display: "flex", alignItems: "center", gap: 6,
            }}>
              <AlertIcon /> Low-confidence result — no relevant sources found
            </div>
          )}
        </div>

        {/* Citations panel (only for assistant messages with sources) */}
        {!isUser && msg.sources && msg.sources.length > 0 && (
          <SourcesPanel sources={msg.sources} />
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TypingIndicator — three animated dots while waiting for first token
// ─────────────────────────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 10, marginBottom: 16 }}>
      <div style={{
        width: 34, height: 34, borderRadius: "50%",
        background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
      }}>
        <SignalIcon />
      </div>
      <div style={{
        padding: "12px 16px",
        background: "#1e293b",
        border: "1px solid #ffffff0f",
        borderRadius: "4px 18px 18px 18px",
        display: "flex", gap: 6, alignItems: "center",
      }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 7, height: 7,
            borderRadius: "50%",
            background: "#475569",
            animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
          }} />
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar — sample questions + clear button
// ─────────────────────────────────────────────────────────────────────────────
function Sidebar({ questions, onSelect, onClear, loading }) {
  return (
    <aside style={{
      width: 260,
      background: "#0a0f1e",
      borderRight: "1px solid #ffffff0a",
      display: "flex",
      flexDirection: "column",
      padding: "24px 0",
      flexShrink: 0,
    }}>
      {/* Brand */}
      <div style={{ padding: "0 20px 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: "linear-gradient(135deg, #3b82f6, #8b5cf6)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "0 0 20px #3b82f650",
          }}>
            <SignalIcon />
          </div>
          <div>
            <div style={{ fontSize: 14, fontWeight: 700, color: "#f1f5f9", letterSpacing: "-0.02em" }}>
              TeleAssist
            </div>
            <div style={{ fontSize: 10.5, color: "#64748b", fontFamily: "'IBM Plex Mono', monospace" }}>
              RAG · Qwen3-32B · Groq
            </div>
          </div>
        </div>
      </div>

      <div style={{ padding: "0 20px 12px", borderTop: "1px solid #ffffff08" }} />

      {/* Sample questions */}
      <div style={{ padding: "12px 20px 8px" }}>
        <p style={{
          fontSize: 10.5, fontWeight: 700, letterSpacing: "0.1em",
          color: "#475569", textTransform: "uppercase",
          fontFamily: "'IBM Plex Mono', monospace", marginBottom: 10,
        }}>
          Try asking
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {questions.map((q, i) => (
            <button
              key={i}
              onClick={() => !loading && onSelect(q)}
              disabled={loading}
              style={{
                textAlign: "left",
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid transparent",
                background: "transparent",
                color: "#94a3b8",
                fontSize: 12.5,
                lineHeight: 1.4,
                cursor: loading ? "not-allowed" : "pointer",
                transition: "all 0.15s",
                opacity: loading ? 0.5 : 1,
              }}
              onMouseEnter={e => {
                if (!loading) {
                  e.currentTarget.style.background = "#ffffff08";
                  e.currentTarget.style.color = "#e2e8f0";
                  e.currentTarget.style.borderColor = "#ffffff10";
                }
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.color = "#94a3b8";
                e.currentTarget.style.borderColor = "transparent";
              }}
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Spacer */}
      <div style={{ flex: 1 }} />

      {/* Clear conversation */}
      <div style={{ padding: "12px 20px", borderTop: "1px solid #ffffff08" }}>
        <button
          onClick={onClear}
          disabled={loading}
          style={{
            width: "100%",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
            padding: "9px 0",
            borderRadius: 8,
            border: "1px solid #ffffff12",
            background: "transparent",
            color: "#64748b",
            fontSize: 12.5,
            cursor: loading ? "not-allowed" : "pointer",
            transition: "all 0.15s",
            opacity: loading ? 0.5 : 1,
          }}
          onMouseEnter={e => {
            if (!loading) {
              e.currentTarget.style.background = "#ef444415";
              e.currentTarget.style.color = "#fca5a5";
              e.currentTarget.style.borderColor = "#ef444430";
            }
          }}
          onMouseLeave={e => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "#64748b";
            e.currentTarget.style.borderColor = "#ffffff12";
          }}
        >
          <TrashIcon /> Clear conversation
        </button>
      </div>
    </aside>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showTyping, setShowTyping] = useState(false);
  const [sampleQuestions, setSampleQuestions] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Auto-scroll to bottom whenever messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, showTyping]);

  // Fetch sample questions from the API on mount
  useEffect(() => {
    fetch(`${API_BASE}/sample-questions`)
      .then(r => r.json())
      .then(d => setSampleQuestions(d.questions || []))
      .catch(() => {});
  }, []);

  // ── Core: send question, consume SSE stream ────────────────────────────────
  const sendQuestion = useCallback(async (question) => {
    if (!question.trim() || loading) return;

    // Add user message
    const userMsg = { id: Date.now(), role: "user", content: question };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setShowTyping(true);

    // Placeholder for the streaming assistant message
    const assistantId = Date.now() + 1;
    let accumulatedText = "";
    let sources = [];
    let isFallback = false;

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      // Read the SSE stream line by line
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Add empty assistant bubble (will be updated token by token)
      setShowTyping(false);
      setMessages(prev => [...prev, {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        sources: [],
        isFallback: false,
      }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // last incomplete line stays in buffer

        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = JSON.parse(line.slice(5).trim());

          if (payload.type === "sources") {
            // ── Received citation metadata ────────────────────────────────
            sources = payload.data;
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, sources } : m
            ));

          } else if (payload.type === "token") {
            // ── Received one streamed token ───────────────────────────────
            accumulatedText += payload.data;
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, content: accumulatedText } : m
            ));

          } else if (payload.type === "fallback") {
            // ── Confidence gate fired — no good sources found ─────────────
            isFallback = true;
            accumulatedText = payload.data;
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, content: accumulatedText, isFallback: true } : m
            ));

          } else if (payload.type === "error") {
            accumulatedText = `Error: ${payload.data}`;
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, content: accumulatedText } : m
            ));

          } else if (payload.type === "done") {
            // ── Stream finished — mark message as no longer streaming ──────
            setMessages(prev => prev.map(m =>
              m.id === assistantId ? { ...m, streaming: false } : m
            ));
          }
        }
      }
    } catch (err) {
      setShowTyping(false);
      setMessages(prev => {
        // Replace placeholder or add new error message
        const hasPlaceholder = prev.some(m => m.id === assistantId);
        const errorMsg = {
          id: assistantId,
          role: "assistant",
          content: `Connection error: ${err.message}. Is the backend running?`,
          streaming: false,
          sources: [],
          isFallback: false,
        };
        return hasPlaceholder
          ? prev.map(m => m.id === assistantId ? errorMsg : m)
          : [...prev, errorMsg];
      });
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading]);

  const handleSubmit = () => sendQuestion(input);
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* Global styles */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #060d1a; font-family: 'Plus Jakarta Sans', sans-serif; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 999px; }

        @keyframes fadeSlideIn {
          from { opacity: 0; transform: translateY(8px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; } 50% { opacity: 0; }
        }
        @keyframes bounce {
          0%, 60%, 100% { transform: translateY(0); }
          30%           { transform: translateY(-6px); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; } 50% { opacity: 0.5; }
        }
      `}</style>

      <div style={{
        display: "flex",
        height: "100vh",
        overflow: "hidden",
        background: "#060d1a",
        color: "#f1f5f9",
      }}>
        {/* ── Sidebar ───────────────────────────────────────────────────── */}
        <Sidebar
          questions={sampleQuestions}
          onSelect={sendQuestion}
          onClear={() => setMessages([])}
          loading={loading}
        />

        {/* ── Main chat area ────────────────────────────────────────────── */}
        <main style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}>
          {/* Header */}
          <header style={{
            padding: "16px 28px",
            borderBottom: "1px solid #ffffff08",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            background: "#060d1a",
            flexShrink: 0,
          }}>
            <div>
              <h1 style={{
                fontSize: 17,
                fontWeight: 700,
                color: "#f1f5f9",
                letterSpacing: "-0.02em",
              }}>
                Customer Care Assistant
              </h1>
              <p style={{ fontSize: 12, color: "#475569", marginTop: 2 }}>
                Connectivity · Billing · SIM · Roaming
              </p>
            </div>

            {/* Live indicator */}
            <div style={{
              display: "flex", alignItems: "center", gap: 7,
              padding: "6px 12px",
              borderRadius: 999,
              border: "1px solid #10b98130",
              background: "#10b98110",
            }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%",
                background: "#10b981",
                animation: "pulse 2s ease infinite",
              }} />
              <span style={{ fontSize: 11.5, color: "#10b981", fontFamily: "'IBM Plex Mono', monospace" }}>
                RAG · Live
              </span>
            </div>
          </header>

          {/* Messages */}
          <div style={{
            flex: 1,
            overflowY: "auto",
            padding: "28px 28px 12px",
          }}>
            {/* Empty state */}
            {messages.length === 0 && (
              <div style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                gap: 16,
                opacity: 0.6,
              }}>
                <div style={{
                  width: 56, height: 56, borderRadius: 16,
                  background: "linear-gradient(135deg, #3b82f620, #8b5cf620)",
                  border: "1px solid #3b82f630",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <SignalIcon />
                </div>
                <div style={{ textAlign: "center" }}>
                  <p style={{ fontSize: 15, fontWeight: 600, color: "#64748b" }}>
                    Ask me anything about your mobile service
                  </p>
                  <p style={{ fontSize: 12.5, color: "#334155", marginTop: 4 }}>
                    I'll cite my sources so you know where each answer comes from
                  </p>
                </div>
              </div>
            )}

            {/* Message list */}
            {messages.map(msg => (
              <MessageBubble key={msg.id} msg={msg} />
            ))}

            {/* Typing indicator */}
            {showTyping && <TypingIndicator />}

            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </div>

          {/* Input bar */}
          <div style={{
            padding: "16px 28px 20px",
            borderTop: "1px solid #ffffff08",
            background: "#060d1a",
            flexShrink: 0,
          }}>
            <div style={{
              display: "flex",
              gap: 10,
              alignItems: "flex-end",
              background: "#0f172a",
              border: `1px solid ${loading ? "#3b82f640" : "#ffffff10"}`,
              borderRadius: 14,
              padding: "10px 10px 10px 16px",
              transition: "border-color 0.2s",
              boxShadow: loading ? "0 0 0 3px #3b82f615" : "none",
            }}>
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe your issue…"
                disabled={loading}
                rows={1}
                style={{
                  flex: 1,
                  background: "transparent",
                  border: "none",
                  outline: "none",
                  color: "#f1f5f9",
                  fontSize: 14.5,
                  lineHeight: 1.5,
                  resize: "none",
                  fontFamily: "'Plus Jakarta Sans', sans-serif",
                  maxHeight: 120,
                  overflowY: "auto",
                  cursor: loading ? "not-allowed" : "text",
                }}
                onInput={e => {
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
              />
              <button
                onClick={handleSubmit}
                disabled={loading || !input.trim()}
                style={{
                  width: 38, height: 38,
                  borderRadius: 10,
                  border: "none",
                  background: loading || !input.trim()
                    ? "#1e293b"
                    : "linear-gradient(135deg, #3b82f6, #6366f1)",
                  color: loading || !input.trim() ? "#475569" : "#fff",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                  flexShrink: 0,
                  transition: "all 0.15s",
                  boxShadow: loading || !input.trim() ? "none" : "0 4px 14px #3b82f640",
                }}
              >
                <SendIcon />
              </button>
            </div>
            <p style={{
              textAlign: "center",
              fontSize: 11,
              color: "#1e293b",
              marginTop: 10,
              fontFamily: "'IBM Plex Mono', monospace",
            }}>
              Answers grounded in FAQ · Resolved Tickets · Technical Guide
            </p>
          </div>
        </main>
      </div>
    </>
  );
}
