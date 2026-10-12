# This is for only developer who Developed it. It is not for others

# RAGApplication — 30-Minute Project Study Guide

**Repository:** https://github.com/ramith14/RAGApplication  
**Project:** AI Study Assistant using Retrieval-Augmented Generation (RAG)

> **How to use this guide:** Read Sections 1–5 to understand the architecture, Sections 6–8 to walk through the code, and Sections 9–12 to prepare for an interview. This guide is grounded in the `app.py` code you shared and the repository structure described in our conversation. For exact details inside `core_rag.py`, `config.py`, and `requirements.txt`, compare with the current files in your repository.

## Your 30-minute plan

| Time | Topic | Goal |
|---|---|---|
| 0–3 min | Project purpose and architecture | Explain what it does |
| 3–7 min | RAG basics | Explain retrieval, embeddings and generation |
| 7–12 min | Technologies | Explain why each library/service is used |
| 12–20 min | `app.py` walkthrough | Explain important lines |
| 20–24 min | End-to-end flow and limitations | Explain what happens and what can fail |
| 24–30 min | Interview practice | Answer common questions aloud |

---

# 1. Project overview

## What does the application do?

RAGApplication is an AI study assistant. A user uploads PDF or TXT/TEXT files, processes them, and asks questions about their contents through a chat interface. The application searches the uploaded documents for relevant passages and gives those passages to a language model to help generate an answer.

### The problem it solves

Imagine you have a 100-page Operating Systems PDF and want to know, “What are the four conditions for deadlock?” Normally, you might search the PDF, locate the section, read it, and write an answer. This app makes the interaction conversational.

### Main workflow

1. Upload one or more documents.
2. Extract text from them.
3. Split the text into smaller chunks.
4. Convert chunks into numerical embeddings.
5. Store/index those vectors for similarity search.
6. Ask a question.
7. Retrieve relevant chunks.
8. Give the question and retrieved context to the LLM.
9. Display the answer and retrieved source passages.

### What it does not do

- It does **not** retrain or fine-tune LLaMA every time a file is uploaded.
- It does **not** guarantee every answer is correct.
- The shared UI accepts PDF/TXT/TEXT files, not every possible file format.
- The code shown creates the chain in the running app session. Permanent storage of uploads, the FAISS index, and chat history is not guaranteed unless separately implemented.

---

# 2. Architecture at a glance

```text
                          USER
                            |
                            v
                    Streamlit UI (app.py)
                     /                 \
              Upload documents        Ask a question
                    |                      |
                    v                      v
            Load and extract text     Chat history
                    |                      |
                    v                      v
              Split into chunks    Clarify follow-up query
                    |                      |
                    v                      v
             Embedding model       Retrieve relevant chunks
                    |                      ^
                    v                      |
                FAISS index  <-------------+
                    |
                    v
           Retrieved document context
                    |
                    v
          Prompt + question + history
                    |
                    v
          LLaMA 3.1 8B through Groq API
                    |
                    v
              Answer + sources
                    |
                    v
              Streamlit UI
```

## What each file is responsible for

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, file uploads, session state, calling the RAG chain, displaying chat and sources |
| `core_rag.py` | Main RAG pipeline: loading, splitting, embeddings, FAISS, retrieval, prompt/model and chain construction |
| `config.py` | Central settings such as title, layout, model, temperature, file-size limit, retrieval settings and author/repository details |
| `requirements.txt` | Python packages that need to be installed |
| `README.md` | Project overview, setup and usage instructions |

**Why split the code into files?** This is *separation of concerns*: each file has a clear responsibility. The UI can change without rewriting retrieval logic, and the RAG pipeline can be tested independently of Streamlit.

---

# 3. RAG — the core concept

**Retrieval-Augmented Generation (RAG)** is a pattern where an application retrieves relevant information from an external knowledge source and supplies it to a generative model as context for answering a question.

- **Retrieval:** Search for relevant document passages.
- **Augmented:** Add the retrieved passages to the model's input context.
- **Generation:** Generate a natural-language answer using the question and supplied context.

## Normal chatbot versus RAG

| Normal LLM chatbot | RAG application |
|---|---|
| Primarily uses the model's learned knowledge and current conversation | Searches a supplied knowledge source before answering |
| May not know a private textbook's contents | Can use text extracted from the uploaded textbook |
| Can give a generic answer | Can give an answer grounded in retrieved passages |
| Sources are not guaranteed | Retrieved passages can be displayed for inspection |

RAG can reduce dependence on a model's general knowledge for document-specific questions, but it does **not** eliminate hallucinations. The retriever might select irrelevant passages, or the LLM might misinterpret the evidence.

## Essential definitions

- **LLM (Large Language Model):** A machine-learning model trained to process and generate language.
- **Prompt:** Instructions and input sent to a model.
- **Context:** Information supplied to a model for a particular response.
- **Hallucination:** A fluent response that is unsupported or incorrect.
- **Inference:** Using an already-trained model to produce an output.
- **Fine-tuning:** Further training a model by updating its parameters. Your document upload flow is not fine-tuning.

---

# 4. Technologies: what they are and why you use them

## Python
A general-purpose programming language. It is used throughout the project because it has a strong ecosystem for AI, text processing, and application development.

## Streamlit
**Definition:** A Python framework for building interactive web apps.

**Why this project uses it:** It provides file uploads, sidebars, buttons, chat bubbles, metrics, status messages and expandable sections without requiring a separate JavaScript frontend.

**Where:** `app.py`.

**Alternatives:** Flask or FastAPI with a separate frontend, Gradio, or a JavaScript-based web application.

## LangChain
**Definition:** A framework for connecting language models, prompts, retrievers, documents, and multi-step workflows.

**Why used:** It orchestrates the retrieval and generation steps, including a history-aware retriever and retrieval chain.

**Where:** Mainly `core_rag.py`.

**Important:** LangChain is not the LLM. It coordinates the components that use the model.

## Hugging Face and sentence-transformers
**Hugging Face** is an ecosystem for machine-learning models and tools. **Sentence-transformers** provides models/libraries for creating text embeddings.

The embedding model described for this project is `sentence-transformers/all-MiniLM-L6-v2`.

**Why used:** It converts document chunks and questions into vectors so semantically related passages can be found even if they use different wording.

## Embeddings
**Definition:** Numerical vectors representing text. In semantic search, text with similar meanings often has nearby vector representations.

Example:
- “How does virtual memory work?”
- “Explain the purpose of memory management.”

The wording differs, but the meaning overlaps. Embeddings help represent that similarity numerically.

Embeddings are not answers or summaries; they are numerical representations used for search.

## Vector search and FAISS
**Vector search** finds vectors that are similar to a query vector. **FAISS** is a library for efficient similarity search over dense vectors.

**Why used:** It helps retrieve a small set of relevant chunks instead of passing the entire document to the LLM for each question.

The project configuration described for this app uses `k = 4` by default: the retriever requests four chunks. This is a design choice and should be evaluated; it is not always the optimal value.

FAISS is a vector-search library, not the LLM and not automatically a persistent hosted database. In this design the index is created in the app runtime, so persistence is not guaranteed.

## Groq API
**Definition:** A service/API for running supported language models (inference).

**Why used:** The application can call LLaMA without hosting all the LLM weights on the Streamlit server.

Distinguish:
- **Groq:** The inference service/API.
- **LLaMA:** The language model that generates the answer.

A valid API key is required. Store it in environment variables or Streamlit Secrets, never in public source code.

## LLaMA 3.1 8B
**Definition:** A model in Meta's Llama 3.1 family; “8B” means approximately eight billion parameters.

**Why used:** It generates conversational answers using the question and retrieved document context. The configured model described for this project is `llama-3.1-8b-instant` through Groq.

It is not being retrained on each uploaded file. The app supplies relevant passages as context during inference.

## PyPDF / `PyPDFLoader`
PyPDF is a Python library for reading PDF files. LangChain's `PyPDFLoader` loads PDF text into document objects and can preserve metadata such as page numbers.

**Limitation:** A scanned PDF may contain page images rather than extractable text and could require OCR.

## `python-dotenv` and environment variables
`python-dotenv` loads settings from a local `.env` file into environment variables. The shared `app.py` calls `load_dotenv()`.

A `.env` file is convenient for local development but should not be committed if it contains secrets. On Streamlit Cloud, configure secrets in the app settings.

## `tempfile` and `os`
- `tempfile`: Python standard-library module for creating temporary files.
- `os`: utilities for file paths, environment access and filesystem operations.

The UI writes uploaded files to temporary paths so the loader can process them, then removes them during cleanup.

## `st.session_state`
Streamlit reruns the script when a user interacts with controls. Session state preserves values across those reruns for the current user session.

Your code stores the chain, message histories, file count and chunk count there.

## `HumanMessage` and `AIMessage`
LangChain message objects representing human and assistant turns. The application passes these messages as conversation history to the RAG chain.

---

# 5. The RAG pipeline step by step

This is the central AI workflow, implemented primarily in `core_rag.py`.

## Step 1 — Load documents
The app accepts PDF, TXT and TEXT files. The loader extracts text and may preserve metadata such as filename and page number.

**Metadata** means information describing a document rather than its main text. A filename and page number are examples. Metadata lets the UI tell the user where a retrieved passage came from.

## Step 2 — Split text into chunks
The settings described for the project are around:
```python
chunk_size = 1000
chunk_overlap = 200
```

A **chunk** is a smaller piece of a longer document.

- `chunk_size=1000`: aims for chunks around 1,000 characters.
- `chunk_overlap=200`: neighboring chunks share some text.

Why not use the whole PDF as one item? Smaller chunks make targeted retrieval possible. Overlap helps preserve context when a concept crosses a chunk boundary. Exact chunk boundaries depend on the splitter and its separators.

## Step 3 — Create embeddings
The embedding model turns each chunk into a vector. The question is also embedded when a user asks it, so the query can be compared with document vectors.

## Step 4 — Build the FAISS index
The vectors are indexed in FAISS. The index supports similarity search over the chunk embeddings.

## Step 5 — Retrieve relevant chunks
The retriever searches for vectors near the query vector. The described `k=4` setting requests four chunks.

Too few chunks may omit needed evidence; too many may introduce noise and consume more context. The best value should be chosen through testing.

## Step 6 — Handle follow-up questions
The app uses a **history-aware retriever**.

Example:
- User: “What is a deadlock?”
- Assistant: explains deadlock.
- User: “How can it be prevented?”

The second question is ambiguous by itself. The history-aware retriever uses earlier messages to reformulate it as a standalone query such as “How can a deadlock be prevented?” before retrieval.

This stage clarifies the search query; it does not generate the final answer.

## Step 7 — Construct the prompt
Retrieved chunks are placed into the model's prompt along with the question and relevant history. The system instruction tells the model to use retrieved context and say when it does not know. A placeholder such as `{context}` is replaced with the retrieved passages at runtime.

## Step 8 — Call the model
The application calls `ChatGroq` with the configured model, API key and temperature.

**Temperature** influences variation in generated text. Lower values generally make responses less varied, but do not guarantee correctness.

## Step 9 — Return the answer and sources
The chain returns an answer and the retrieved documents. The UI displays the answer and an expandable source panel with excerpts and source labels.

Displaying a source is not formal verification: the user should check whether the passage really supports the answer.

---

# 6. `app.py` — important lines explained

This walkthrough follows the code you shared in the conversation.

## 6.1 Imports

```python
import streamlit as st
import tempfile, os
from dotenv import load_dotenv
from core_rag import build_rag_chain
from config import config
from langchain_core.messages import HumanMessage, AIMessage
```

- `import streamlit as st`: imports Streamlit under the short name `st`.
- `tempfile, os`: temporary files, paths, and file cleanup.
- `load_dotenv`: reads local `.env` settings.
- `build_rag_chain`: imports the function that builds the RAG pipeline.
- `config`: imports application settings.
- `HumanMessage`, `AIMessage`: represent human and assistant messages in LangChain history.

**Why import `build_rag_chain`?** To keep UI code separate from the AI pipeline.

## 6.2 Load environment settings

```python
load_dotenv()
```

Loads local `.env` variables if a file exists. It does not create an API key. On a hosted deployment, use the hosting platform's Secrets settings.

## 6.3 Configure the page

```python
st.set_page_config(
    page_title=config.app_title,
    page_icon=config.app_icon,
    layout=config.layout
)
```

- `page_title`: browser tab title.
- `page_icon`: browser tab icon.
- `layout`: page layout option.

The values come from `config.py`, allowing settings to be changed centrally. `st.set_page_config()` should be called near the beginning, before other Streamlit UI commands.

## 6.4 CSS styling

```python
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
```

- `st.markdown()` renders Markdown.
- `<style>` contains CSS.
- `.main-header`, `.sub-header`, `.stat-box`, and `.footer` are CSS class selectors.
- `unsafe_allow_html=True` allows HTML/CSS rendering.

This changes appearance, not the RAG algorithm. Avoid rendering untrusted user content as trusted raw HTML.

## 6.5 Render the header

```python
st.markdown(
    f'<p class="main-header">{config.app_title}</p>',
    unsafe_allow_html=True
)
```

The `f` makes this an **f-string**. Python substitutes `{config.app_title}` with the configured title. The next statement does the same for the description.

## 6.6 Initialize session state

```python
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
```

This creates a list only if the key does not already exist. It prevents chat history from resetting on every Streamlit rerun.

| Key | Purpose |
|---|---|
| `chat_history` | LangChain message objects sent to the chain |
| `display_history` | Dictionaries used to redraw messages and sources in the UI |
| `chain` | The built RAG chain; starts as `None` |
| `chunk_count` | Number of indexed chunks |
| `file_count` | Number of successfully processed files |

**Why two histories?** One uses LangChain message objects for retrieval; the other stores UI-friendly dictionaries with roles, text and source chunks.

## 6.7 Format source labels

```python
def format_source_label(source):
    filename = os.path.basename(
        source.metadata.get("source", "Uploaded file")
    )
    page = source.metadata.get("page")

    if page is None:
        return filename
    return f"{filename} · Page {page + 1}"
```

- `def` defines a function.
- `source` is expected to be a document-like object.
- `source.metadata` is a dictionary of metadata.
- `.get("source", "Uploaded file")` returns the source value or a fallback.
- `os.path.basename(...)` extracts the filename from a path.
- `page is None` checks whether page metadata is missing.
- `page + 1` converts a zero-based page index to a human-friendly page number.

If the page index is `0`, the UI displays Page 1. TXT files may have no page metadata, so only the filename is shown.

## 6.8 Sidebar and upload control

```python
with st.sidebar:
    ...
```

`with` creates a context so contained elements appear in the sidebar.

```python
uploaded_files = st.file_uploader(
    "...",
    type=["pdf", "txt", "text"],
    accept_multiple_files=True
)
```

- `st.file_uploader()` creates the upload widget.
- `type` limits the file types offered by the UI.
- `accept_multiple_files=True` allows more than one file.
- `uploaded_files` is a list of uploaded file objects when files are selected.

## 6.9 Check file sizes

```python
oversized_files = [
    uploaded_file.name
    for uploaded_file in uploaded_files
    if uploaded_file.size / (1024 * 1024) > config.max_file_size_mb
]
```

This is a **list comprehension**: it builds a list of filenames that exceed the configured size limit.

- `uploaded_file.size` is in bytes.
- `1024 * 1024` converts bytes to MiB.
- `config.max_file_size_mb` provides the limit.

If any file is too large, the app shows an error instead of processing the batch.

## 6.10 Process Files button

```python
elif st.button("⚡ Process Files", type="primary", use_container_width=True):
```

The button returns `True` when clicked. `elif` means this branch runs only if files exist and none is oversized. `type="primary"` styles the button; `use_container_width=True` fills the sidebar width.

## 6.11 Write uploads to temporary files

```python
temp_paths = []

with st.spinner("Indexing your documents..."):
    try:
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded_file.read())
                temp_paths.append(f.name)
```

- `temp_paths = []`: keeps paths for later cleanup.
- `st.spinner(...)`: displays a busy indicator.
- `try`: starts an error-handling block.
- `for`: loops through uploaded files.
- `os.path.splitext(...)`: separates filename and extension.
- `.lower()`: normalizes extension case.
- `NamedTemporaryFile`: creates a temporary file.
- `delete=False`: keeps the file available after closing, so a loader can reopen it by path.
- `f.write(...)`: writes uploaded bytes.
- `temp_paths.append(f.name)`: records the path.

## 6.12 Build and store the chain

```python
chain, count = build_rag_chain(*temp_paths)
st.session_state.chain = chain
st.session_state.chunk_count = count
st.session_state.file_count = len(uploaded_files)
```

- `build_rag_chain(...)` calls the core RAG pipeline.
- `*temp_paths` is **argument unpacking**: each path is passed as a separate positional argument.
- The function returns a chain and a count, according to the UI's use of it.
- Session state stores them so they survive Streamlit reruns.

The app then clears old chat histories because a new document set has been indexed.

**Accuracy note:** The exact internals and return types must be confirmed in your current `core_rag.py`; this UI code shows how the function is called.

## 6.13 Error handling and temporary-file cleanup

```python
except Exception as e:
    st.error(f"Error processing files: {e}")
finally:
    for tmp_path in temp_paths:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
```

- `except Exception as e`: catches an exception and assigns it to `e`.
- `st.error(...)`: shows an error message.
- `finally`: runs whether the operation succeeded or failed.
- `os.path.exists(...)`: checks whether the file still exists.
- `os.unlink(...)`: deletes the temporary file.

Why use `finally`? Cleanup should happen even when processing fails. In a production app, show safe user-facing messages and keep sensitive diagnostics out of public output.

## 6.14 Session statistics

```python
col1, col2 = st.columns(2)
col1.metric("Chunks", st.session_state.chunk_count)
col2.metric("Files", st.session_state.file_count)
st.metric("Messages", len(st.session_state.display_history))
```

`st.columns(2)` creates two columns. `st.metric()` displays a label and value. `len()` counts list entries.

The message count counts display entries. A normal user/assistant exchange adds two entries.

## 6.15 Clear Chat

```python
if st.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.display_history = []
    st.rerun()
```

The button clears both histories and reruns the app.

**Important:** It does not set `st.session_state.chain = None`. So it clears the conversation, not the processed documents. The user can ask new questions about the same indexed documents.

## 6.16 Display model settings

```python
st.markdown(f"""
**⚙️ Model Config**
🤖 `{config.llm_model}`
🌡️ Temp: `{config.llm_temperature}`
🔍 Top-K: `{config.retriever_k}`
""")
```

This displays settings in the sidebar. Displaying a setting does not itself configure the model; the pipeline must read and use the corresponding value.

- `llm_model`: configured generation model.
- `llm_temperature`: generation-variation setting.
- `retriever_k`: number of chunks requested from retrieval.

## 6.17 Welcome screen and conditional UI

```python
if not st.session_state.chain:
    # Welcome screen
else:
    # Chat history and chat input
```

If no chain exists, the app shows feature cards and an upload instruction. If the chain exists, it displays chat history and the chat input.

This explains why the original textbox appears only after successful document processing: `st.chat_input()` is inside the `else` branch.

The feature card says “100% Free”, but treat that as a UI claim, not an unconditional guarantee. Hosting limits, API usage and service terms may affect availability or cost.

## 6.18 Render prior messages and sources

```python
for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
```

- `for` loops over saved messages.
- `st.chat_message()` creates a user or assistant chat bubble.
- `msg["role"]` chooses the speaker.
- `st.write()` displays the content.

For assistant messages with sources, an expander shows source labels and excerpts. `src.page_content[:400]` uses **string slicing** to display the first 400 characters; `"..."` indicates that the excerpt may be truncated.

## 6.19 Read a question

```python
if question := st.chat_input("Ask anything about your document..."):
```

- `st.chat_input()` creates the chat input.
- `:=` is the **assignment expression**, often called the walrus operator. It assigns the submitted value to `question` and tests whether it is non-empty.
- The body runs when a question is submitted.

Because this line is inside the `else` branch, the input is only rendered after the chain exists in the original version.

## 6.20 Invoke the RAG chain

```python
result = st.session_state.chain.invoke({
    "input": question,
    "chat_history": st.session_state.chat_history
})
answer = result["answer"]
sources = result.get("context", [])
```

- `.invoke(...)` runs the chain with an input dictionary.
- `"input"` carries the question.
- `"chat_history"` carries earlier LangChain messages.
- `result["answer"]` extracts the generated answer.
- `result.get("context", [])` retrieves source documents if that key exists, otherwise an empty list.

These keys must match the output schema created in `core_rag.py`.

## 6.21 Update chat histories

```python
st.session_state.chat_history.extend([
    HumanMessage(content=question),
    AIMessage(content=answer)
])
```

`.extend(...)` adds the two message objects to the existing list. The history-aware retriever can use them to understand follow-up questions.

The app separately appends user and assistant dictionaries to `display_history` for rendering the chat in Streamlit.

## 6.22 Footer

The final `st.markdown(...)` renders a footer with the app name, author, supported file types and technologies. It is a presentation element, not part of the retrieval or generation algorithm.

---

# 7. What `core_rag.py` and `config.py` contribute

## `core_rag.py`

The described pipeline uses these components. Check the current file for the exact implementation and imports.

| Component | Purpose |
|---|---|
| Document loader | Reads PDF/TXT content and metadata |
| Text splitter | Breaks long text into chunks |
| Embedding model | Converts chunks into vectors |
| FAISS vector store | Indexes vectors for similarity search |
| Retriever | Finds chunks relevant to a query |
| History-aware retriever | Uses conversation history to clarify follow-up questions |
| Prompt template | Specifies how context and the question are given to the model |
| Chat model client | Calls the configured model through Groq |
| Retrieval chain | Coordinates retrieval and answer generation |
| `build_rag_chain(...)` | Entry point called by `app.py`; the UI expects a chain and chunk count |

**Why use a chain?** A chain packages multiple steps into one callable workflow. The UI can submit a question and history rather than manually running each operation.

## `config.py`

The shared `app.py` references these settings:

| Setting | Meaning |
|---|---|
| `app_title` | App/page title |
| `app_icon` | Page icon |
| `layout` | Streamlit layout |
| `app_description` | Short description under the title |
| `max_file_size_mb` | Per-file limit checked by the UI |
| `llm_model` | Model name shown in the sidebar |
| `llm_temperature` | Temperature shown in the sidebar |
| `retriever_k` | Retrieval count shown in the sidebar |
| `author_name` | Author text shown in the UI |
| `github_url` | Repository link |

Other settings, such as chunk size, overlap or embedding model, may also be defined in `config.py`. Check whether `core_rag.py` actually reads them.

**Important engineering principle:** A setting only affects behavior if the code consumes it. If a value is defined in configuration but hardcoded separately in the pipeline, changing the config may not change the behavior.

---

# 8. Deployment and debugging

## How Streamlit Community Cloud deployment works

The cloud service builds the application from GitHub, installs packages listed in `requirements.txt`, and runs the selected entry-point file (usually `app.py`). The app also needs a valid Groq API key and enough resources to load the embedding model and build the index.

## Why `ModuleNotFoundError` happened

The traceback you shared pointed to:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

Python could not import that module in the deployed environment. A common cause is that the package was missing from `requirements.txt` or did not install successfully.

**Package/import naming difference:**
- Pip package: `langchain-huggingface`
- Python import: `langchain_huggingface`

If a new missing-module error appears, check the exact module in the traceback, whether the corresponding package is listed, the installation logs, and whether the deployment rebuilt after your GitHub commit.

## API key safety

- Never put a real API key in public Python code or this document.
- Do not commit a `.env` file containing secrets.
- Use environment variables for local development.
- Use Streamlit Secrets for Community Cloud.
- If a key is accidentally exposed, revoke it and create a replacement.

## Deployment checklist

- [ ] `app.py` is the selected entry point.
- [ ] `core_rag.py` and `config.py` are committed.
- [ ] `requirements.txt` includes the required packages.
- [ ] Groq API key is configured in deployment Secrets.
- [ ] App starts without a traceback.
- [ ] A small PDF/TXT can be uploaded and processed.
- [ ] A question returns an answer.
- [ ] Source excerpts are relevant.
- [ ] Follow-up questions use history correctly.
- [ ] Errors are handled sensibly.

---

# 9. Why these design choices?

## Why RAG instead of fine-tuning?
RAG uses uploaded documents at query time, which is practical when each user may provide different material. Fine-tuning changes model parameters and is usually unnecessary merely to answer questions about a new textbook.

## Why chunk documents?
Chunks make targeted retrieval possible and reduce the amount of irrelevant text sent to the model.

## Why overlap chunks?
Overlap helps preserve context near boundaries. Too much overlap creates redundant chunks and adds indexing overhead.

## Why embeddings rather than keyword matching?
Keyword matching depends on words overlapping. Embeddings can retrieve text with related meaning despite different wording. Embeddings are not perfect, and hybrid keyword/vector search could be explored.

## Why FAISS?
It provides efficient vector similarity search without requiring a separate vector-database service for a small prototype.

## Why use a separate embedding model and LLaMA?
They have different jobs: the embedding model converts text into vectors for retrieval; LLaMA generates a natural-language answer.

## Why Groq?
It avoids hosting the full LLM in the Streamlit runtime. The trade-off is dependence on network connectivity, API availability, rate limits and secure key handling.

## Why Streamlit?
It provides a usable Python UI quickly, with upload, chat, sidebar and metrics components, without a separate frontend.

## Why session state?
Streamlit reruns scripts after user interactions. Session state preserves the chain and message history across reruns in the same session.

## Why show sources?
Source excerpts help users inspect the evidence behind an answer. They improve transparency but do not automatically prove the answer is correct.

---

# 10. Limitations and improvements

A good project explanation acknowledges limitations honestly.

| Limitation or risk | Potential improvement |
|---|---|
| Scanned PDFs may have no extractable text | Add OCR and clearer extraction errors |
| One chunk size may not fit every document | Evaluate different chunking strategies |
| Fixed top-k can retrieve too little or too much | Tune top-k on representative questions |
| Retriever may return irrelevant passages | Add similarity thresholds, metadata filtering or reranking |
| Prompt instructions cannot eliminate hallucinations | Evaluate grounding and add abstention behavior |
| Index may exist only in runtime memory | Persist indexes if reuse across sessions is needed |
| Session state is not a shared database | Add durable storage if cross-session history is required |
| API errors/rate limits may interrupt responses | Add safe error handling and appropriate retries |
| Embedding-model loading can consume time/memory | Cache models and monitor resources |
| No evaluation metrics are visible in the shared UI code | Build a test set and measure retrieval/answer quality |
| UI file-size checks are not the only possible limits | Account for hosting limits and validate untrusted files |

## How would you evaluate the app?

1. **Retrieval test:** Does the retrieved top-k include the passage that answers the question?
2. **Grounding test:** Is the generated answer supported by those passages?
3. **Abstention test:** If the document lacks the answer, does the model say so?
4. **Follow-up test:** Does the app resolve “it”, “that method”, and similar references?
5. **Document test:** Does it handle multiple files, empty PDFs, scanned PDFs and oversized files?
6. **Latency test:** How long do embedding, retrieval and generation take?
7. **Cost/limits test:** Does the API stay within applicable usage limits?

---

# 11. Interview questions and model answers

### Q1. Explain your project in 60–90 seconds.
“I built an AI study assistant using Retrieval-Augmented Generation. Users upload PDF or text documents and ask questions through a Streamlit interface. The application extracts the text, splits it into overlapping chunks, creates embeddings using a Hugging Face sentence-transformer, and indexes them with FAISS. When a question arrives, a history-aware retriever uses the conversation to clarify follow-up questions and retrieve relevant chunks. LangChain passes those passages and the question to LLaMA 3.1 8B through the Groq API. The app displays the answer and retrieved sources. The goal is to make document-based learning conversational and reduce the effort needed to find information in lengthy study materials.”

### Q2. What is the most important part of the project?
The retrieval-to-generation pipeline. The UI is how the user interacts with the app; the RAG pipeline connects document processing, vector retrieval, context construction and model inference.

### Q3. Does the model read the entire PDF for every question?
Not in the described design. The app processes and indexes chunks first. For each question, it retrieves a limited number of relevant chunks and sends them as context.

### Q4. Does uploading a document train LLaMA?
No. The app creates embeddings and an index, then supplies retrieved passages as context at inference time. It does not fine-tune LLaMA.

### Q5. What is the difference between retrieval and generation?
Retrieval selects relevant passages from the index. Generation uses the question and supplied context to write the answer.

### Q6. Why use a history-aware retriever?
Follow-up questions can be incomplete by themselves. Chat history helps reformulate them into standalone queries before searching.

### Q7. Why retrieve four chunks?
Four is the configured top-k choice described for the project. It limits context size while attempting to provide enough evidence. The best value must be chosen through evaluation.

### Q8. Why overlap chunks by 200 characters?
It helps retain context when a concept crosses a boundary. It can create redundancy, so the setting should be tested.

### Q9. What if the answer isn't in the document?
The prompt asks the model to say it doesn't know when the context is insufficient, but this is not a guarantee. Test unsupported questions and consider retrieval thresholds or explicit abstention logic.

### Q10. How do you reduce hallucinations?
Retrieve relevant context, instruct the model to ground answers in it, display sources, test unsupported questions, and evaluate whether answers are supported. None of these guarantees zero hallucinations.

### Q11. Why not use a normal database?
A relational database is good for structured records and exact filtering. FAISS is useful for semantic similarity search over vectors. A larger system may use both: a database for metadata and a vector index for retrieval.

### Q12. What does `st.session_state` solve?
Streamlit reruns scripts after interactions. Session state retains the chain and histories across those reruns in a user's session.

### Q13. Why are there two histories?
`chat_history` contains LangChain message objects for the chain. `display_history` contains dictionaries with role, content and sources for the UI.

### Q14. What does Clear Chat do?
It clears both message histories and reruns the app. It does not clear `st.session_state.chain`, so the processed documents remain available in that session.

### Q15. What are the deployment dependencies?
The source files, packages in `requirements.txt`, a valid Groq key configured as a secret, and sufficient runtime resources for embeddings and the vector index.

### Q16. What would you improve first?
I would build an evaluation set, measure retrieval relevance and answer grounding, then tune chunking and top-k based on results. I would also improve error handling and dependency pinning.

### Q17. FAISS versus Hugging Face embeddings?
The embedding model turns text into vectors. FAISS indexes and searches those vectors. One creates numerical representations; the other finds similar representations.

### Q18. What does temperature do?
It affects variation in generated text. A lower temperature usually makes responses less varied, but does not guarantee correctness.

### Q19. Why use an API instead of running LLaMA locally?
Running an LLM locally requires enough compute, memory and model setup. An API simplifies hosting but adds an external service dependency and requires secure key management.

### Q20. Name a limitation.
The system can retrieve irrelevant passages or generate unsupported content, and the index may not persist beyond the runtime. I would evaluate retrieval quality, handle unsupported questions, and add persistent storage if needed.

---

# 12. Glossary

| Term | Definition |
|---|---|
| AI | Systems designed to perform tasks associated with intelligence |
| LLM | Large Language Model; a model trained to process and generate language |
| RAG | Retrieve external information and supply it as context for generation |
| Retrieval | Selecting information relevant to a query |
| Generation | Producing text from a model |
| Chunk | A smaller piece of a long document |
| Chunk size | Target size of each text chunk |
| Chunk overlap | Text shared between neighboring chunks |
| Embedding | Numerical vector representing text |
| Semantic similarity | Similarity in meaning, not just identical words |
| Vector | An ordered collection of numbers |
| Vector index | A data structure organized for vector search |
| FAISS | Library for efficient vector similarity search |
| Embedding model | Model that converts text into vectors |
| Retriever | Component that selects relevant chunks for a query |
| Top-k | Number of retrieval results requested |
| Inference | Using a trained model to produce an output |
| Fine-tuning | Further training a model by updating its parameters |
| Prompt | Instructions and input sent to a model |
| System prompt | High-level instructions for model behavior |
| Context | Information supplied to a model for a response |
| Hallucination | Fluent but unsupported or incorrect output |
| Metadata | Information about content, such as filename/page |
| Chain | A coordinated sequence of framework operations |
| History-aware retrieval | Retrieval that uses prior messages to clarify the query |
| API | Interface that lets software components communicate |
| API key | Secret credential used to authorize API calls |
| Session state | Values preserved across Streamlit reruns in one session |
| Inference provider | Service that runs a model and returns output |
| Temperature | Generation setting that affects output variation |
| OCR | Optical Character Recognition; converts text in images/scans to machine-readable text |
| Dependency | Package required by an application |
| `requirements.txt` | File listing Python packages to install |
| Environment variable | Key-value setting provided to a running process |
| Deployment | Making an application available in an environment |
| Latency | Time taken for an operation or response |
| Persistence | Retaining data beyond the current process/session |
| Rerun | Re-executing a Streamlit script after interaction |
| `try/except/finally` | Python error handling; `finally` runs cleanup whether success or failure |
| F-string | Python string that substitutes expressions inside `{}` |
| List comprehension | Concise syntax for creating a list from an iterable |
| Argument unpacking (`*items`) | Passing items in a list/tuple as separate positional arguments |
| `None` | Python value representing the absence of a value |
| `is None` | Identity check commonly used to test for `None` |
| `.get(key, default)` | Dictionary method that returns a value or a fallback |
| `.extend(items)` | Adds multiple items to a list |
| String slicing (`text[:400]`) | Selects a portion of a string |
| `finally` | Block that runs whether an exception occurred or not |

---

# 13. Two-minute revision sheet

Memorize these 13 points:

1. **Purpose:** Ask questions about uploaded PDF/TXT documents conversationally.
2. **UI:** Streamlit builds the interface.
3. **Orchestration:** LangChain connects retrieval and generation.
4. **Loading:** Extract text and preserve metadata.
5. **Chunking:** Split into pieces around 1,000 characters with 200-character overlap, as described for the project.
6. **Embeddings:** Hugging Face `all-MiniLM-L6-v2` represents text as vectors.
7. **Search:** FAISS indexes vectors; retriever requests top four chunks by default.
8. **Follow-ups:** A history-aware retriever uses prior messages to clarify questions.
9. **Generation:** LLaMA 3.1 8B answers using context via Groq.
10. **Sources:** The UI shows retrieved chunks so users can inspect the evidence.
11. **State:** `st.session_state` preserves chain and histories across reruns.
12. **Security:** Keep API keys in secrets/environment variables, never in public GitHub code.
13. **Limitations:** RAG can still retrieve wrong passages or hallucinate; test retrieval and grounding.

**One sentence to remember:**

> “My project converts uploaded documents into searchable embeddings, retrieves relevant passages for each question, and uses those passages as context for an LLM to generate a conversational answer with inspectable sources.”

---

# 14. Self-test — answer without looking

1. What does RAG stand for?
2. What is the difference between an embedding model and LLaMA?
3. Why split documents into chunks?
4. What does FAISS do?
5. What does top-k = 4 mean?
6. Why does the app need chat history?
7. What role does LangChain play?
8. Why use Streamlit rather than a separate frontend?
9. What happens when Process Files is clicked?
10. What does the `finally` block do?
11. Why is the original chat box only shown after a chain is built?
12. How can you reduce hallucinations without claiming to eliminate them?
13. Where should the Groq API key be stored?
14. Name two limitations and two improvements.
15. How would you test whether answers are grounded in the uploaded document?

If you can answer these in your own words and explain the architecture from memory, you have a strong foundation for a project interview.
