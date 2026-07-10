#!/usr/bin/env python3
"""
mini_rag_ui.py  -  A local web UI for mini_rag.py.

This is presentation only. It imports and reuses mini_rag.py unchanged, so
chunking, embedding, and retrieval behave exactly as they do on the command
line. It shows the passages retrieved for your question, and - if a generator
is configured (Gemini via GEMINI_API_KEY, or a local Ollama) - the written
answer composed from those passages.

Run it (from inside the mini-RAG folder, with the `rag` env active):

    pip install streamlit
    streamlit run mini_rag_ui.py

It opens in your browser at a local address. Retrieval and embeddings run on
this machine. In Gemini mode, the retrieved passages are sent to Google to
compose the answer - the sidebar tells you which mode is live.
"""

import io
import contextlib

import requests
import streamlit as st

import mini_rag  # single source of truth for config + retrieval logic


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="mini RAG, document retrieval",
    page_icon="::",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
      --rb-ink:    #1b2330;
      --rb-muted:  #586274;
      --rb-line:   #e3e6eb;
      --rb-panel:  #f6f7f9;
      --rb-accent: #1f6f6a;
      --rb-flag:   #a2521c;
    }

    html, body, [data-testid="stAppViewContainer"] {
      font-family: 'IBM Plex Sans', system-ui, sans-serif;
    }

    .rb-title {
      font-family: 'IBM Plex Sans', sans-serif;
      font-weight: 600;
      font-size: 1.55rem;
      letter-spacing: -0.01em;
      color: var(--rb-ink);
      margin: 0;
    }
    .rb-sub {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--rb-muted);
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      margin-top: 0.35rem;
    }
    .rb-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--rb-accent); display: inline-block;
    }
    .rb-rule {
      height: 1px; background: var(--rb-line); border: 0;
      margin: 0.9rem 0 1.4rem 0;
    }

    .rb-cardhead {
      display: flex; align-items: center; gap: 0.6rem;
      margin-bottom: 0.35rem;
    }
    .rb-rank {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem; font-weight: 500;
      color: #fff; background: var(--rb-accent);
      border-radius: 4px; padding: 0.08rem 0.4rem;
      min-width: 1.2rem; text-align: center;
    }
    .rb-tag {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.8rem; color: var(--rb-muted);
      word-break: break-all;
    }

    .rb-note {
      font-family: 'IBM Plex Sans', sans-serif;
      font-size: 0.86rem; color: var(--rb-ink);
      background: #fbf3ec; border-left: 3px solid var(--rb-flag);
      padding: 0.7rem 0.9rem; border-radius: 4px;
      margin-bottom: 1.1rem;
    }
    .rb-answerhead {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase;
      color: var(--rb-accent); margin-bottom: 0.3rem;
    }
    .rb-srcline {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 0.78rem; color: var(--rb-muted); margin-top: 0.4rem;
    }

    [data-testid="stFormSubmitButton"] button {
      background: var(--rb-accent); color: #fff; border: 0;
    }
    [data-testid="stFormSubmitButton"] button:hover {
      background: #195c58; color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Warm-up + helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the embedding model (first launch only)...")
def _warm():
    mini_rag.get_embedder()
    return mini_rag.get_collection()


def ollama_reachable() -> bool:
    base = mini_rag.OLLAMA_URL.rsplit("/api/", 1)[0]
    try:
        requests.get(base + "/api/tags", timeout=1.0)
        return True
    except Exception:
        return False


def generator_status():
    backend = mini_rag.active_generator()
    if backend == "gemini":
        return (
            "gemini",
            f"Gemini - {mini_rag.GEMINI_MODEL}",
            "Connected (hosted). Retrieved passages are sent to Google's Gemini API to compose the answer.",
        )
    if backend == "ollama":
        if ollama_reachable():
            return (
                "ollama",
                f"Ollama - {mini_rag.OLLAMA_MODEL} (local)",
                "Connected (local). Answers are composed on this machine; nothing leaves it.",
            )
        return (
            "none",
            "Ollama selected but not running",
            "Questions return retrieved passages only. Start Ollama, or set GEMINI_API_KEY, to get written answers.",
        )
    return (
        "none",
        "None",
        "No generator configured. Questions return retrieved passages only. Set GEMINI_API_KEY (hosted) or start Ollama (local) to get answers.",
    )


def corpus_status(col):
    state = mini_rag.indexed_state(col)
    return len(state), col.count(), state


def source_disagreement_hint(hits):
    sources = sorted({md.get("source") for _, md in hits if md.get("source")})
    if len(sources) > 1:
        st.info(
            "This answer was retrieved from multiple documents, so Gemini may combine related material across books.",
            icon="ℹ️",
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col = _warm()

st.markdown('<p class="rb-title">mini RAG &middot; document retrieval</p>',
            unsafe_allow_html=True)
st.markdown('<span class="rb-sub"><span class="rb-dot"></span>'
            'Local embeddings &amp; retrieval &middot; pluggable generation</span>',
            unsafe_allow_html=True)
st.markdown('<hr class="rb-rule">', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Guidance + Sidebar
# ---------------------------------------------------------------------------
st.info(
    "For best results, ask about one source at a time.",
    icon="ℹ️",
)
st.caption(
    "Examples: 'In My Korean 1, what are the appendices?' or 'What does Tammy lesson 14 explain?'"
)

with st.sidebar:
    st.markdown("**Corpus**")
    n_files, n_chunks, _ = corpus_status(col)
    c1, c2 = st.columns(2)
    c1.metric("Files indexed", n_files)
    c2.metric("Chunks", n_chunks)

    st.divider()
    st.markdown("**Retrieval depth**")
    top_k = st.slider(
        "Passages to return",
        min_value=1,
        max_value=15,
        value=mini_rag.TOP_K,
        help="How many chunks to pull back per question. More = broader coverage, but noisier.",
    )

    st.divider()
    if st.button("Re-index documents", use_container_width=True,
                 help="Re-scan the docs folder. Only changed files are reprocessed."):
        buf = io.StringIO()
        with st.spinner("Re-indexing..."):
            with contextlib.redirect_stdout(buf):
                mini_rag.ingest()
        st.success(buf.getvalue().strip() or "Done.")
        st.rerun()

    st.divider()
    st.markdown("**Generator**")
    gen_state, gen_label, gen_msg = generator_status()
    st.markdown(f"**{gen_label}**")
    st.caption(gen_msg)

    if gen_state == "gemini":
        st.warning(
            "Gemini mode sends retrieved passages to Google for generation.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------
if "last" not in st.session_state:
    st.session_state.last = None

with st.form("ask", clear_on_submit=False):
    question = st.text_input(
        "Ask the corpus",
        placeholder="e.g. In My Korean 1, what are the appendices?",
    )
    submitted = st.form_submit_button("Search")

if submitted and question.strip():
    mini_rag.TOP_K = top_k
    with st.spinner("Retrieving..."):
        hits = mini_rag.retrieve(question)
        answer = mini_rag.generate(mini_rag.build_prompt(question, hits)) if hits else None
    st.session_state.last = {"q": question, "hits": hits, "answer": answer}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
last = st.session_state.last

if last is None:
    st.markdown(
        '<div class="rb-note">Ask a question above to search the indexed documents. Results show the passages the system retrieved, ranked by relevance.</div>',
        unsafe_allow_html=True,
    )
elif not last["hits"]:
    st.markdown(
        '<div class="rb-note">Nothing was retrieved. If the corpus is empty, add documents (.md, .txt, .pdf, .docx) to the docs folder and use Re-index documents.</div>',
        unsafe_allow_html=True,
    )
else:
    hits = last["hits"]
    sources = sorted({md.get("source") for _, md in hits})

    if mini_rag.active_generator() == "gemini":
        st.warning(
            "Gemini mode is active, so the retrieved passages are sent to Google to compose the answer.",
            icon="⚠️",
        )

    if last["answer"]:
        st.markdown('<div class="rb-answerhead">Answer</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(last["answer"])
        source_disagreement_hint(hits)
        st.markdown(
            '<div class="rb-srcline">Sources: ' + ", ".join(sources) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="rb-note">Showing the passages retrieved for your question. Configure a generator (Gemini or Ollama) to turn these into a written answer.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="rb-answerhead">Retrieved passages</div>', unsafe_allow_html=True)
    for i, (doc, md) in enumerate(hits, 1):
        src = (md.get("source") or "").replace("<", "&lt;").replace(">", "&gt;")
        with st.container(border=True):
            st.markdown(
                f'<div class="rb-cardhead"><span class="rb-rank">{i}</span><span class="rb-tag">{src}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(doc)