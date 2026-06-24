# Simple-Rag# 🤖 Local RAG Chatbot with Ollama

A fully local Retrieval-Augmented Generation (RAG) chatbot that runs entirely on your machine — no OpenAI API needed.
Built with **Ollama**, **ChromaDB**, and **Gradio**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Ollama](https://img.shields.io/badge/Ollama-local-green)
![Gradio](https://img.shields.io/badge/UI-Gradio-orange)

---

## 📌 Features

- ✅ Fully local — no API key needed
- ✅ RAG pipeline from scratch (no LangChain)
- ✅ Query rewriting for better retrieval
- ✅ Reranking for more accurate results
- ✅ Conversation history / memory
- ✅ Source context displayed alongside answers
- ✅ Built with Gradio UI

---

## 🏗️ Architecture

```
Ingestion Pipeline:
Knowledge Base (.md files)
    → Chunking (qwen2.5:3b)
    → Embedding (nomic-embed-text)
    → ChromaDB (vector store)

Query Pipeline:
User Question
    → Query Rewriting
    → Embedding + Retrieval (ChromaDB)
    → Reranking
    → LLM Answer (qwen2.5:3b)
    → Gradio UI
```

---

## 🛠️ Tech Stack

| Component     | Tool                          |
| ------------- | ----------------------------- |
| LLM           | qwen2.5:3b (via Ollama)       |
| Embedding     | nomic-embed-text (via Ollama) |
| Vector Store  | ChromaDB                      |
| UI            | Gradio                        |
| LLM Interface | LiteLLM                       |

---

## ⚙️ Prerequisites

Before running this project, make sure you have:

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/username/repo-name.git
cd repo-name
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

### 5. Prepare your knowledge base

Place your `.md` files inside the knowledge base folder:

```
Simple-Rag/
└── knowledge-base/
    ├── employees/
    │   └── *.md
    └── company/
        └── *.md
```

---

## 📦 Dependencies

```
chromadb
litellm
gradio
python-dotenv
pydantic
tenacity
requests
tqdm
```

Install all at once:

```bash
pip install chromadb litellm gradio python-dotenv pydantic tenacity requests tqdm
```

---

## ▶️ How to Run

### Step 1 — Start Ollama

# <<<<<<< HEAD

> > > > > > > f658b9d (1st commit)

```bash
ollama serve
```

### Step 2 — Run ingestion (build vector database)

```bash
caffeinate -i python ingest.py  # Mac (prevents sleep)
python ingest.py                # Windows/Linux
```

> ⚠️ Ingestion may take a while depending on your hardware and number of documents.
> A `chunks_backup.json` will be saved — if ingestion fails, re-running will skip chunking and go straight to embedding.

### Step 3 — Run the app

```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## 📁 Project Structure

```
.
├── ingest.py          # Document ingestion pipeline
├── answer.py          # RAG query pipeline
├── app.py             # Gradio UI
├── requirements.txt
├── README.md
├── knowledge-base/    # Your .md documents
└── preprocessed_db/   # ChromaDB vector store (auto-generated)
```

---

## 💡 How It Works

### Ingestion (`ingest.py`)

1. Load all `.md` files from the knowledge base
2. Chunk each document using LLM (with overlap for better retrieval)
3. Embed each chunk using `nomic-embed-text`
4. Store vectors in ChromaDB

### Query (`answer.py`)

1. **Rewrite query** — refine user question for better retrieval
2. **Fetch context** — embed question → search ChromaDB (twice: original + rewritten)
3. **Merge & rerank** — combine results, rerank by relevance
4. **Answer** — send top chunks as context to LLM → generate answer

---

## ⚡ Performance (MacBook Air M1 8GB RAM)

|                      | Time           |
| -------------------- | -------------- |
| Ingestion (~32 docs) | ~50 minutes    |
| First response       | ~50 seconds    |
| Subsequent responses | ~28-35 seconds |

> Performance varies depending on device specifications.

---

## 🔧 Configuration

You can change the model in `answer.py` and `ingest.py`:

```python
MODEL = "ollama/qwen2.5:3b"        # LLM for chunking & answering
EMBED_MODEL = "ollama/nomic-embed-text"  # Embedding model
RETRIEVAL_K = 20  # Number of chunks retrieved from ChromaDB
FINAL_K = 10      # Number of chunks after reranking
```

---

## 📝 License

MIT License

---

## 🙋 Author

Made by [Fauzi Hamdani](https://www.linkedin.com/in/fauzihamdani/)
