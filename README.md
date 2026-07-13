```markdown
# mini RAG

A minimal, general-purpose retrieval-augmented generation app for a folder of documents.

It indexes local documents with local embeddings, stores vectors locally on disk, and can answer questions using either Google Gemini, a local Ollama model, or retrieval-only mode[cite: 3].

## System Workflow & Architecture

The system operates as a modular three-stage pipeline designed to handle mixed-language content seamlessly:


```

[ Download Stage ]               [ Ingestion Stage ]               [ Query / Generation Stage ]
──────────────────               ───────────────────               ────────────────────────────
download_korean_sources.py  ──>  mini_rag.py ingest            ──>  mini_rag_ui.py (Streamlit)
(Scrapes & aggregates            (Local chunking & parsing          (Local vector retrieval via Chroma +
textbooks into /docs)            via multilingual embeddings)       Bilingual generation via Gemini/Ollama)

```

1. **Data Aggregation:** `download_korean_sources.py` automatically scrapes and pulls down selected English-Korean textbooks, lessons, and PDFs, aggregating them into the local `./docs` directory[cite: 3, 6].
2. **Local Multilingual Ingestion:** Running `mini_rag.py ingest` processes files incrementally using delta detection[cite: 5]. The text is parsed, divided into structural chunks using an overlapping sliding window, and vectorized locally using a multilingual model optimized to map both English and Korean semantics into a shared coordinate space[cite: 5]. The resulting vectors are written directly to your local Chroma database[cite: 5].
3. **Hybrid Generation:** When a user queries the `mini_rag_ui.py` interface, the question is mapped against the local vector store to pull the most contextually relevant language passages[cite: 4, 5]. The UI bundles these multilingual source fragments with strict contextual formatting constraints, handing the payload off to Gemini 2.5 Flash to ensure highly accurate, bilingual, script-native output[cite: 4, 5].

## Features

- **Local Multilingual Embeddings:** Built-in support for Korean and English text using `paraphrase-multilingual-MiniLM-L12-v2` via `sentence-transformers`[cite: 3, 5].
- Local persistent vector storage with Chroma[cite: 3].
- Incremental re-indexing with file hash tracking[cite: 3].
- Supports `.md`, `.txt`, `.pdf`, and `.docx`[cite: 3].
- Streamlit UI for browsing retrieved passages and answers[cite: 3].
- Optional Gemini or Ollama generation[cite: 3].

## Project structure

- `mini_rag.py` — core indexing, retrieval, and generation logic[cite: 3].
- `mini_rag_ui.py` — Streamlit UI[cite: 3].
- `docs/` — downloaded source documents[cite: 3].
- `rag_db/` — local Chroma vector database[cite: 3].

## Requirements

- Python 3.10+[cite: 3]
- `pip`[cite: 3]

Optional:
- Gemini API key for hosted generation[cite: 3].
- Ollama for local generation[cite: 3].

## Install

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

```

If you do not have a `requirements.txt` yet, install the core dependencies manually:

```bash
pip install sentence-transformers chromadb requests streamlit pypdf python-docx beautifulsoup4

```

## Configuration

Environment variables:

* `RAG_DOCS_DIR` — folder containing documents to index. Default: `./docs`

* `RAG_DB_DIR` — folder for the local vector database. Default: `./rag_db`

* `RAG_GENERATOR` — `auto`, `gemini`, `ollama`, or `none`

* `GEMINI_API_KEY` — API key for Gemini


* `GOOGLE_API_KEY` — alternative API key name supported by the script


* `GEMINI_MODEL` — Gemini model name. Default: `gemini-2.5-flash`

* `OLLAMA_MODEL` — Ollama model name



## Download documents

If you are using the Korean corpus workflow, run your downloader script first to populate `docs/`.

```bash
python download_korean_sources.py

```

## Ingest documents

Whenever you modify your documents or change your local embedding configuration, make sure to re-index the corpus.

```bash
python mini_rag.py ingest

```

## Check status

```bash
python mini_rag.py status

```

## Ask questions from the command line

```bash
python mini_rag.py ask "What are common greetings in korean?"

```

## Run the Streamlit UI

```bash
streamlit run mini_rag_ui.py

```

## Usage tips

For best results, ask about one source at a time.

Good examples:

* "In My Korean 1, what are the appendices?"


* "What does Tammy lesson 14 explain?"


* "What common greetings are in korean?"

Broad questions may combine information from multiple documents.

## Privacy note

Embeddings and retrieval stay local.

If you use `RAG_GENERATOR=gemini`, the retrieved passages are sent to Google for answer generation. Use `ollama` or `none` mode if you want to keep generation local or retrieval-only.

## License

This project is licensed under the MIT License, see the LICENSE file for details.
## Acknowledgements

Built for local document retrieval and lightweight RAG experimentation.

```

```