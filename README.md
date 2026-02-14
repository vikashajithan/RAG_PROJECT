# 🤖 Adaptive RAG Assistant

An **Adaptive Retrieval-Augmented Generation (RAG) Application** built using **LangChain 1.x**, **FAISS**, **HuggingFace Embeddings**, **OpenAI**, **SerpAPI**, and **Streamlit**.

This system automatically decides whether to answer from uploaded documents (PDF, TXT, DOCX) or fall back to live web search when document knowledge is insufficient.

---

## 🚀 Features

- 📄 Upload PDF, TXT, DOCX documents
- 🧠 Automatic document chunking & embedding
- 🔍 Semantic search using FAISS
- 🌐 Live web search fallback using SerpAPI
- 🔄 Adaptive routing (Document → Web)
- 🧩 Modern LangChain LCEL architecture
- 🖥️ Simple Streamlit UI
- ❌ No hallucination-first design (strict grounding)

---

## 🏗️ Architecture

