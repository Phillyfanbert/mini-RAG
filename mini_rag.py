#!/usr/bin/env python3
"""
mini_rag.py  -  A minimal, general-purpose RAG over a folder of documents.

Point it at ANY folder of documents on ANY topic. Retrieval and embeddings run
locally; generation (writing an answer from the retrieved passages) can run
against a hosted API (Google Gemini, free tier) or a local model (Ollama), or
be turned off entirely.

Design choices:
  - Embeddings run LOCALLY (sentence-transformers). No document text leaves the
    machine at index time.
  - Vectors are stored LOCALLY on disk (Chroma PersistentClient). No server.
  - Generation is PLUGGABLE. Three modes, chosen by RAG_GENERATOR env var:
      * "gemini"  -> Google Gemini API (hosted). Retrieved passages ARE sent to
                     Google. Requires GEMINI_API_KEY (free key from AI Studio).
      * "ollama"  -> local model via Ollama. Nothing leaves the machine.
      * "none"    -> no generation; just return the retrieved passages.
      * "auto"    -> (default) use Gemini if a key is set, else Ollama, else
                     fall back to retrieval-only.
    If the chosen generator isn't reachable, the tool degrades gracefully and
    just returns the retrieved chunks, so it always works with zero setup.
  - Re-ingest is CHEAP: each file's content hash is stored with its chunks.
    On re-run, unchanged files are skipped, changed files are re-embedded
    (old chunks deleted first), and files removed from the folder are purged.
    This is the "delta detection" pattern.

  PRIVACY NOTE: in "gemini" mode the retrieved passages leave your machine and
  are sent to Google. On the Gemini FREE tier, Google may use API inputs and
  outputs to improve their products. Only use Gemini mode with documents you
  are comfortable sending to a third party. For anything sensitive, use
  "ollama" or "none" mode instead (retrieval and embeddings stay local either
  way; only the generation step ever egresses).

Setup:
  pip install sentence-transformers chromadb requests
  # optional, only if you want to index these formats:
  pip install pypdf python-docx
  # optional, for hosted generation:
  export GEMINI_API_KEY="your-key-from-https://aistudio.google.com/apikey"

Usage:
  python mini_rag.py ingest                 # index / re-index ./docs
  python mini_rag.py ask "your question"    # retrieve + answer
  python mini_rag.py status                 # what's currently indexed

Supported file types out of the box: .md, .txt, .pdf (needs pypdf),
.docx (needs python-docx).
"""

import sys
import os
import hashlib
import pathlib

import requests  # talks to Gemini (hosted) and/or Ollama (local)

# ----------------------------------------------------------------------------
# CONFIG  -  edit these to taste (most also read from env so you don't have to)
# ----------------------------------------------------------------------------
DOCS_DIR       = os.environ.get("RAG_DOCS_DIR", "./docs")   # folder of documents
DB_DIR         = os.environ.get("RAG_DB_DIR", "./rag_db")   # where Chroma persists
COLLECTION     = "mini_rag"          # logical name of the vector collection
EMBED_MODEL    = "all-MiniLM-L6-v2"  # local embedding model (small, CPU-fine)
FILE_EXTS      = {".md", ".txt", ".pdf", ".docx"}  # which files to index
CHUNK_SIZE     = 1000                # target chunk length in characters
CHUNK_OVERLAP  = 150                 # characters of overlap between chunks
TOP_K          = 5                   # how many chunks to retrieve per question

# --- Generation backend selection -------------------------------------------
# "gemini" | "ollama" | "none" | "auto"  (auto: Gemini if key set, else Ollama)
GENERATOR      = os.environ.get("RAG_GENERATOR", "auto").lower()

# Hosted generation via Google Gemini (https://aistudio.google.com/apikey).
# Official client libs accept GEMINI_API_KEY or GOOGLE_API_KEY; we honor both.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
# gemini-2.5-flash is a stable, free-tier model as of mid-2026. Newer free
# flagship is gemini-3.5-flash; flash-lite variants exist too. Free lineup
# shifts over time, so this is env-overridable. Check the AI Studio pricing
# page if a model string ever returns 404 / "not available".
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL     = ("https://generativelanguage.googleapis.com/v1beta/models/"
                  f"{GEMINI_MODEL}:generateContent")

# Local generation via Ollama (https://ollama.com). Optional.
OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3.2")  # whatever you've pulled

# ----------------------------------------------------------------------------
# Lazy singletons so importing this file is cheap; heavy libs load on demand.
# ----------------------------------------------------------------------------
_embedder = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)  # downloads once, then cached
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=DB_DIR)
        _collection = client.get_or_create_collection(COLLECTION)
    return _collection


def embed(texts):
    """Return a list of vectors (lists of floats) for a list of strings."""
    vecs = get_embedder().encode(texts, show_progress_bar=False)
    return [v.tolist() for v in vecs]


# ----------------------------------------------------------------------------
# Document reading + chunking
# ----------------------------------------------------------------------------
def read_file(path: pathlib.Path) -> str:
    """Return plain text for a supported file. Unknown/unreadable -> ''."""
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    return ""


def _read_pdf(path: pathlib.Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  (skipping {path.name}: run 'pip install pypdf' to index PDFs)")
        return ""
    try:
        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        print(f"  (could not read {path.name}: {e})")
        return ""


def _read_docx(path: pathlib.Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        print(f"  (skipping {path.name}: run 'pip install python-docx' to index .docx)")
        return ""
    try:
        d = docx.Document(str(path))
        return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  (could not read {path.name}: {e})")
        return ""


def file_hash(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunk_text(text: str):
    """Simple character-window chunker with overlap. Good enough for mini scale.

    Packs whole paragraphs where it can so chunks don't split mid-sentence,
    then falls back to hard slicing for very long paragraphs.
    """
    text = text.strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(p) > CHUNK_SIZE:
            if buf:
                chunks.append(buf)
                buf = ""
            for i in range(0, len(p), CHUNK_SIZE - CHUNK_OVERLAP):
                chunks.append(p[i:i + CHUNK_SIZE])
            continue
        if len(buf) + len(p) + 2 <= CHUNK_SIZE:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            chunks.append(buf)
            # carry a small overlap tail into the next buffer
            tail = buf[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            buf = (tail + "\n\n" + p) if tail else p
    if buf:
        chunks.append(buf)
    return chunks


# ----------------------------------------------------------------------------
# Ingest  (delta detection via per-file content hash)
# ----------------------------------------------------------------------------
def current_files():
    root = pathlib.Path(DOCS_DIR)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in FILE_EXTS
    )


def indexed_state(col):
    """Return {source_relpath: stored_hash} for everything currently in the DB."""
    got = col.get(include=["metadatas"])
    state = {}
    for md in got.get("metadatas", []) or []:
        if md and "source" in md:
            state[md["source"]] = md.get("hash")
    return state


def ingest():
    col = get_collection()
    root = pathlib.Path(DOCS_DIR)
    files = current_files()
    if not files:
        print(f"No {'/'.join(sorted(FILE_EXTS))} files found under {DOCS_DIR!r}. "
              f"Put some documents there and re-run.")
        return

    existing = indexed_state(col)          # source -> hash already in DB
    seen_sources = set()
    added = changed = skipped = 0

    for path in files:
        rel = str(path.relative_to(root))
        seen_sources.add(rel)
        h = file_hash(path)

        if existing.get(rel) == h:
            skipped += 1
            continue

        # New or changed: remove any old chunks for this file, then re-add.
        if rel in existing:
            col.delete(where={"source": rel})
            changed += 1
        else:
            added += 1

        chunks = chunk_text(read_file(path))
        if not chunks:
            continue
        vecs = embed(chunks)
        ids = [f"{rel}::{i}" for i in range(len(chunks))]
        metas = [{"source": rel, "hash": h, "chunk": i} for i in range(len(chunks))]
        col.add(ids=ids, documents=chunks, embeddings=vecs, metadatas=metas)

    # Purge files that were deleted from the folder but still sit in the DB.
    removed = 0
    for stale in set(existing) - seen_sources:
        col.delete(where={"source": stale})
        removed += 1

    print(f"Ingest complete. new={added} changed={changed} "
          f"unchanged={skipped} removed={removed}")


# ----------------------------------------------------------------------------
# Query  (retrieve -> assemble context -> generate)
# ----------------------------------------------------------------------------
def retrieve(question: str):
    col = get_collection()
    q_vec = embed([question])
    res = col.query(
        query_embeddings=q_vec,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def build_prompt(question: str, hits):
    context_blocks = []
    for i, (doc, md) in enumerate(hits, 1):
        context_blocks.append(f"[{i}] (source: {md.get('source')})\n{doc}")
    context = "\n\n".join(context_blocks)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


# --- Generation backends -----------------------------------------------------
def active_generator() -> str:
    """Resolve which backend will actually be used, given config + env.

    Returns one of: "gemini", "ollama", "none". This is the *intended*
    backend; the generate_* functions still return None if the backend turns
    out to be unreachable at call time, so retrieval-only fallback always works.
    """
    if GENERATOR == "gemini":
        return "gemini" if GEMINI_API_KEY else "none"
    if GENERATOR == "ollama":
        return "ollama"
    if GENERATOR == "none":
        return "none"
    # auto
    return "gemini" if GEMINI_API_KEY else "ollama"


def generate_gemini(prompt: str):
    """Hosted generation via Google Gemini. Returns None on any failure.

    Sends the retrieved passages (inside `prompt`) to Google. The API key goes
    in the x-goog-api-key header, not the URL, so it stays out of logs/query
    strings.
    """
    if not GEMINI_API_KEY:
        return None
    try:
        r = requests.post(
            GEMINI_URL,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()
        return text or None
    except Exception:
        return None


def generate_ollama(prompt: str):
    """Local generation via Ollama. Returns None if Ollama isn't reachable."""
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip() or None
    except Exception:
        return None


def generate(prompt: str):
    """Dispatch to the active backend. Returns None -> caller shows passages."""
    backend = active_generator()
    if backend == "gemini":
        return generate_gemini(prompt)
    if backend == "ollama":
        return generate_ollama(prompt)
    return None


def ask(question: str):
    hits = retrieve(question)
    if not hits:
        print("Nothing indexed yet. Run:  python mini_rag.py ingest")
        return

    answer = generate(build_prompt(question, hits))
    sources = sorted({md.get("source") for _, md in hits})

    print("=" * 70)
    if answer:
        print(answer)
    else:
        print("(No generator produced an answer - showing retrieved context "
              "instead. Set GEMINI_API_KEY, or start Ollama, to get written "
              "answers.)\n")
        for i, (doc, md) in enumerate(hits, 1):
            print(f"[{i}] {md.get('source')}\n{doc}\n")
    print("=" * 70)
    print("Sources: " + ", ".join(sources))


def status():
    col = get_collection()
    state = indexed_state(col)
    total_chunks = col.count()
    print(f"Indexed files: {len(state)}  |  total chunks: {total_chunks}")
    print(f"Generator: {active_generator()}"
          + (f" ({GEMINI_MODEL})" if active_generator() == "gemini" else "")
          + (f" ({OLLAMA_MODEL})" if active_generator() == "ollama" else ""))
    for src, h in sorted(state.items()):
        print(f"  {src}  ({h[:8]})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1].lower()
    if cmd == "ingest":
        ingest()
    elif cmd == "ask":
        if len(sys.argv) < 3:
            print('Usage: python mini_rag.py ask "your question"')
            return
        ask(" ".join(sys.argv[2:]))
    elif cmd == "status":
        status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
