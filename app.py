import streamlit as st
import time
import uuid
import os
from dotenv import load_dotenv
from core.rag_engine import build_rag_chain, ask_question
from core.pipeline_logger import PipelineLogger

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
}

/* ── Progress / Status ── */
.step-track {
    display: flex;
    gap: 4px;
    margin: 0.6rem 0 0.5rem 0;
}

.step-segment {
    flex: 1;
    height: 6px;
    border-radius: 3px;
    background: var(--border);
}

.seg-done    { background: var(--success); }
.seg-active  { background: var(--accent-glow); box-shadow: 0 0 6px var(--accent-glow); animation: pulse 1.5s infinite; }
.seg-pending { background: var(--border); }

.step-current {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.8rem;
    color: var(--text-muted);
}

.stTabs [aria-selected="true"] {
    color: var(--accent-glow) !important;
}

/* ── Chat ── */
.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 90%;
    color: var(--text) !important;
    opacity: 1;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.16); border: 1px solid rgba(6,182,212,0.28);  align-self: flex-start; }

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Stale Streamlit elements ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
    "processing_log": [],
    "pipeline_running": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
PIPELINE_STEPS = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️", "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
    ("highlights", "🎯", "Highlight Detection"),
    ("clips",      "✂️", "Clip Cutting"),
]

def render_pipeline_progress():
    """Compact horizontal step progress: a segmented bar + current-stage label,
    so pipeline status is visible at a glance without scrolling a list."""
    steps = st.session_state.pipeline_steps

    segments_html = ""
    current_label = None
    for key, icon, label in PIPELINE_STEPS:
        state = steps.get(key, "pending")
        seg_class = {"done": "seg-done", "active": "seg-active"}.get(state, "seg-pending")
        segments_html += f'<div class="step-segment {seg_class}"></div>'
        if state == "active":
            current_label = f"{icon} {label}"

    if current_label is None:
        if steps and all(v == "done" for v in steps.values()):
            current_label = "✅ All steps complete"
        else:
            current_label = "Waiting to start…"

    st.markdown(f'<div class="step-track">{segments_html}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-current">{current_label}</div>', unsafe_allow_html=True)

def _fmt_ts(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        render_pipeline_progress()

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Processing Log (persistent; updates live during a run) ───────────────────────
log_expander = st.expander("🪵 Processing Log", expanded=st.session_state.pipeline_running)
log_placeholder = log_expander.empty()
if st.session_state.processing_log:
    log_placeholder.code("\n".join(st.session_state.processing_log))
else:
    log_placeholder.caption("No processing yet — run an analysis to see live logs here.")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        from utils.audio_processor import process_input, acquire_video_source
        from core.transcriber import transcribe_all
        from core.summarizer import summarize, generate_title
        from core.extractor import extract_action_items, extract_key_decisions, extract_questions
        from core.rag_engine import build_rag_chain, ask_question
        from core.highlighter import generate_highlights
        from core.clipper import cut_clips

        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.processing_log = []
        st.session_state.pipeline_running = True

        def on_log(line):
            st.session_state.processing_log.append(line)
            log_placeholder.code("\n".join(st.session_state.processing_log))

        pipeline_logger = PipelineLogger(on_log=on_log)

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            with progress_placeholder.container():
                st.info("⚙️ Pipeline running — see the Processing Log above for live status…")

            update_step("audio", "active")
            chunks = process_input(source, logger=pipeline_logger)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcription = transcribe_all(chunks, language, logger=pipeline_logger)
            transcript = transcription["text"]
            transcript_segments = transcription["segments"]
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript, logger=pipeline_logger)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript, logger=pipeline_logger)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items  = extract_action_items(transcript, logger=pipeline_logger)
            decisions     = extract_key_decisions(transcript, logger=pipeline_logger)
            questions     = extract_questions(transcript, logger=pipeline_logger)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript, logger=pipeline_logger)
            update_step("rag", "done")

            update_step("highlights", "active")
            # English/Whisper path only — transcript_segments is None for Hinglish/Sarvam.
            highlights = generate_highlights(transcript_segments, logger=pipeline_logger) if transcript_segments else []
            update_step("highlights", "done")

            update_step("clips", "active")
            if highlights:
                try:
                    video_path = acquire_video_source(source, logger=pipeline_logger)
                    clip_dir = os.path.join("clips", uuid.uuid4().hex)
                    highlights = cut_clips(video_path, highlights, clip_dir, logger=pipeline_logger)
                except Exception as clip_exc:
                    # Clip cutting is a bonus on top of an already-successful
                    # analysis — don't let it take down the whole result.
                    print(f"Clip cutting failed (non-fatal): {clip_exc}")
            update_step("clips", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "transcript_segments": transcript_segments,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
                "highlights": highlights,
            }
            st.session_state.pipeline_done = True
            st.session_state.pipeline_running = False
            progress_placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            st.session_state.pipeline_running = False
            for k in ["audio","transcript","title","summary","extract","rag","highlights","clips"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    tab_overview, tab_actions, tab_transcript, tab_highlights, tab_chat = st.tabs(
        ["📌 Overview", "✅ Action Items & Decisions", "📝 Transcript", "🎯 Highlights & Reels", "💬 Chat"],
        key="main_tabs",
    )

    # ── Overview ─────────────────────────────────────────────────────────────
    with tab_overview:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📌 Session Title</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
                {r['title']}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Action Items & Decisions ─────────────────────────────────────────────
    with tab_actions:
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">✅ Action Items</div>
                <div class="card-content">{r['action_items']}</div>
            </div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">🔑 Key Decisions</div>
                <div class="card-content">{r['key_decisions']}</div>
            </div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">❓ Open Questions</div>
                <div class="card-content">{r['open_questions']}</div>
            </div>""", unsafe_allow_html=True)

    # ── Transcript ───────────────────────────────────────────────────────────
    with tab_transcript:
        st.markdown(
            f'<div class="transcript-box" style="max-height:65vh">{r["transcript"]}</div>',
            unsafe_allow_html=True,
        )

    # ── Highlight / Reel Candidates ──────────────────────────────────────────
    with tab_highlights:
        if r.get("highlights"):
            for i, clip in enumerate(r["highlights"]):
                with st.container(border=True):
                    header = clip.get("title") or f"Highlight {i + 1}"
                    st.markdown(f"#### {header}")
                    st.caption(f"⏱️ {_fmt_ts(clip['start'])} – {_fmt_ts(clip['end'])}  ·  Score {clip['score']}")

                    clip_path = clip.get("path")
                    vcol, mcol = st.columns([1, 2], gap="medium")

                    with vcol:
                        if clip_path and os.path.exists(clip_path):
                            st.video(clip_path)
                        else:
                            st.markdown(
                                '<div class="card-content" style="color:var(--text-muted)">Clip file not available.</div>',
                                unsafe_allow_html=True,
                            )

                    with mcol:
                        st.markdown(f'<div class="card-content">{clip["reason"]}</div>', unsafe_allow_html=True)

                        if clip.get("description") or clip.get("tags"):
                            with st.expander("📋 Shorts metadata"):
                                if clip.get("description"):
                                    st.markdown(f"**Description:**\n\n{clip['description']}")
                                if clip.get("tags"):
                                    st.markdown(f"**Tags:** {', '.join(clip['tags'])}")

                        if clip_path and os.path.exists(clip_path):
                            with open(clip_path, "rb") as f:
                                st.download_button(
                                    "⬇️ Download clip",
                                    data=f.read(),
                                    file_name=os.path.basename(clip_path),
                                    mime="video/mp4",
                                    key=f"download_clip_{i}",
                                    use_container_width=True,
                                )
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">🎯</div>
                <div style="color:var(--text-muted);font-size:0.85rem">No highlight candidates were found for this video.</div>
            </div>""", unsafe_allow_html=True)

    # ── RAG Chat ─────────────────────────────────────────────────────────────
    with tab_chat:
        if st.session_state.chat_history:
            chat_html = '<div class="chat-container">'
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-end">
                        <span class="chat-label user-label">You</span>
                        <div class="chat-bubble user-bubble">{msg['content']}</div>
                    </div>"""
                else:
                    chat_html += f"""
                    <div class="chat-msg" style="align-items:flex-start">
                        <span class="chat-label bot-label">🤖 Assistant</span>
                        <div class="chat-bubble bot-bubble">{msg['content']}</div>
                    </div>"""
            chat_html += '</div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:2rem">
                <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
                <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
            </div>""", unsafe_allow_html=True)

        chat_col1, chat_col2 = st.columns([5, 1], gap="small")
        with chat_col1:
            user_input = st.text_input("Your question", placeholder="What were the main decisions made?", label_visibility="collapsed")
        with chat_col2:
            send_btn = st.button("Send →", use_container_width=True)

        if send_btn and user_input.strip():
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_input.strip())
            st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

else:
    # Empty state
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)