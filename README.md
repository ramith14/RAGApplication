> Chat with one or more PDF/TXT files using RAG + LLaMA 3.1 — 100% free to run.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.1-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🎯 What It Does

Upload one or more documents — PDFs, text files, textbooks, research papers, or notes — and ask questions in plain English. The app indexes all uploaded files together, then uses Retrieval-Augmented Generation (RAG) to find the most relevant sections and answer with source references.

Supported uploads:

- `.pdf`
- `.txt`
- `.text`
- One file or multiple files at once

## 🏗️ Architecture
```
PDF files → PyPDF Loader ┐
                         ├→ Text Splitter → HuggingFace Embeddings → FAISS Vector Store
TXT files → Text Loader ─┘
                                                        ↑
User Question → History-Aware Retriever ────────────────┘
                        ↓
              LLaMA 3.1 8B (via Groq) → Answer + Sources
```

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Dhayanidhi-96/ai-study-assistant
cd ai-study-assistant
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment
```bash
cp .env.example .env
# Add your Groq API key to .env
```

Get your free Groq API key at: https://console.groq.com

### 4. Run locally
```bash
streamlit run app.py
```

Then upload one or more PDF/TXT files from the sidebar and click **Process Files**.

## 🐳 Run with Docker
```bash
# Build and run
docker-compose up --build

# Stop
docker-compose down
```

Open http://localhost:7860

## ☁️ Deploy on Hugging Face Spaces (Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Settings:
   - **Name:** ai-study-assistant
   - **SDK:** Docker
   - **Visibility:** Public
4. Push your code:
```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-study-assistant
git push space main
```
5. Go to **Settings → Repository Secrets** → Add `GROQ_API_KEY`
6. Your app is live! 🎉

## ⚙️ Configuration

All settings are in `config.py` — no need to touch core logic:

| Setting | Default | Description |
|--------|---------|-------------|
| `llm_model` | `llama-3.1-8b-instant` | Groq model to use |
| `llm_temperature` | `0.3` | Response creativity |
| `chunk_size` | `1000` | Document chunk size |
| `retriever_k` | `4` | Chunks retrieved per query |
| `max_file_size_mb` | `10` | Max upload size per file |

## 📁 Upload Behavior

The app accepts multiple uploaded files in a single session. Each file is saved temporarily, loaded according to its type, split into chunks, embedded, and added to one shared FAISS vector store.

PDF sources show page numbers in the source panel. Text files show the file name as the source.

If you change `app.py` or `core_rag.py` while Streamlit is already running, restart the Streamlit app so Python reloads the updated code.

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline & chains |
| Groq API | Fast LLM inference (free tier) |
| LLaMA 3.1 8B | Language model |
| PyPDF Loader | PDF extraction |
| HuggingFace Embeddings | Text embeddings (free, local) |
| FAISS | Vector similarity search |
| Streamlit | Web UI |
| Docker | Containerization |
