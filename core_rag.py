# core_rag.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Corrected Imports for LangChain v1.0+
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

def normalize_file_paths(file_paths):
    """Return a flat list of file path strings from strings, lists, tuples, or sets."""
    if isinstance(file_paths, (str, os.PathLike)):
        return [str(file_paths)]

    normalized = []
    for file_path in file_paths:
        if isinstance(file_path, (list, tuple, set)):
            normalized.extend(normalize_file_paths(file_path))
        else:
            normalized.append(str(file_path))

    return normalized


def load_text_file(file_path: str):
    """Load a plain text file with a few common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as file:
                text = file.read()
            return [Document(page_content=text, metadata={"source": file_path, "file_type": "text"})]
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read text file: {file_path}")


def load_documents(file_paths):
    """Load one or more PDF/TXT files into LangChain documents."""
    docs = []
    file_paths = normalize_file_paths(file_paths)

    for file_path in file_paths:
        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            docs.extend(PyPDFLoader(file_path).load())
        elif extension in (".txt", ".text"):
            docs.extend(load_text_file(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    return docs


def build_rag_chain(*file_paths):
    """Load one or more PDF/TXT files, create a vector store, and return a conversational RAG chain."""
    
    # 1. Load and split the document
    file_paths = normalize_file_paths(file_paths)
    docs = load_documents(file_paths)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 2. Embed and store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 3. Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant", # <-- Updated model name!
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # 4. Create a History-Aware Retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 5. Create the Question-Answering Chain
    qa_system_prompt = (
        "You are a helpful study assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 6. Combine into final chain and return it along with chunk count
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain, len(chunks)
