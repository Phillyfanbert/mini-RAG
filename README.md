# mini RAG

A minimal, general-purpose retrieval-augmented generation app for a folder of documents.

It indexes local documents with local embeddings, stores vectors locally on disk, and can answer questions using either Google Gemini, a local Ollama model, or retrieval-only mode.

## Features

- Local embeddings with `sentence-transformers`.
- Local persistent vector storage with Chroma.
- Incremental re-indexing with file hash tracking.
- Supports `.md`, `.txt`, `.pdf`, and `.docx`.
- Streamlit UI for browsing retrieved passages and answers.
- Optional Gemini or Ollama generation.

## Project structure

- `mini_rag.py` — core indexing, retrieval, and generation logic.
- `mini_rag_ui.py` — Streamlit UI.
- `docs/` — downloaded source documents.
- `rag_db/` — local Chroma vector database.

## Requirements

- Python 3.10+
- `pip`

Optional:
- Gemini API key for hosted generation.
- Ollama for local generation.

## Install

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, install the core dependencies manually:

```bash
pip install sentence-transformers chromadb requests streamlit pypdf python-docx
```

## Configuration

Environment variables:

- `RAG_DOCS_DIR` — folder containing documents to index. Default: `./docs`
- `RAG_DB_DIR` — folder for the local vector database. Default: `./rag_db`
- `RAG_GENERATOR` — `auto`, `gemini`, `ollama`, or `none`
- `GEMINI_API_KEY` — API key for Gemini
- `GOOGLE_API_KEY` — alternative API key name supported by the script
- `GEMINI_MODEL` — Gemini model name
- `OLLAMA_MODEL` — Ollama model name

## Download documents

If you are using the Korean corpus workflow, run your downloader script first to populate `docs/`.

```bash
python download_korean_sources.py
```

## Ingest documents

```bash
python mini_rag.py ingest
```

## Check status

```bash
python mini_rag.py status
```

## Ask questions from the command line

```bash
python mini_rag.py ask "What are the main components described in My Korean 1?"
```

## Run the Streamlit UI

```bash
streamlit run mini_rag_ui.py
```

## Usage tips

For best results, ask about one source at a time.

Good examples:

- "In My Korean 1, what are the appendices?"
- "What does Tammy lesson 14 explain?"
- "What does the beginning Korean textbook say about grammar?"

Broad questions may combine information from multiple documents.

## Privacy note

Embeddings and retrieval stay local.

If you use `RAG_GENERATOR=gemini`, the retrieved passages are sent to Google for answer generation. Use `ollama` or `none` mode if you want to keep generation local or retrieval-only.

## License

Add your preferred license here.

## Acknowledgements

Built for local document retrieval and lightweight RAG experimentation.