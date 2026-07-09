# config.py
import os
from dataclasses import dataclass

@dataclass
class AppConfig:
    # App
    app_title: str = "📚 AI Study Assistant"
    app_icon: str = "📚"
    app_description: str = "Upload any PDF and chat with it instantly using RAG + LLaMA 3"
    layout: str = "wide"

    # Model
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.3
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_k: int = 4

    # UI
    max_file_size_mb: int = 10
    author_name: str = "K.Ramith"
    github_url: str = "https://github.com/ramith14"
config = AppConfig()