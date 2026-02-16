# 🤖 Adaptive RAG Assistant

Document + Web Intelligent Question Answering System

Adaptive RAG Assistant is a modern **Retrieval-Augmented Generation
(RAG)** application that intelligently answers questions from uploaded
documents and automatically falls back to live web search when the
answer is not found locally.

Built using **LangChain (LCEL)**, **FAISS**, **HuggingFace Embeddings**,
**OpenAI**, **SerpAPI**, and **Streamlit**.

------------------------------------------------------------------------

## ✨ Key Highlights

-   Upload **PDF, TXT, DOCX** documents\
-   Automatic document chunking and vectorization\
-   Semantic search using FAISS\
-   Adaptive routing: **Document → Web fallback**\
-   Live Google search via SerpAPI\
-   Strict grounding to reduce hallucinations\
-   Simple and clean Streamlit UI

------------------------------------------------------------------------

## 🧠 How It Works

User Question\
→ If document uploaded\
  → Retrieve relevant chunks\
  → If strong context → Answer from document\
  → Else → Web Search\
→ If no document → Web Search\
→ LLM → Final Answer

------------------------------------------------------------------------

## 🏗 Architecture Components

-   Document Loader -- Loads PDF, TXT, DOCX\
-   Text Splitter -- Breaks documents into manageable chunks\
-   Embedding Model -- Converts text into vector representations\
-   Vector Database -- Stores embeddings using FAISS\
-   Retriever -- Finds relevant chunks\
-   LLM -- Generates grounded answers\
-   Web Search -- Retrieves live data using SerpAPI

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   Streamlit\
-   LangChain (1.x)\
-   HuggingFace Sentence Transformers\
-   FAISS\
-   OpenAI GPT Models\
-   SerpAPI

------------------------------------------------------------------------

## 📁 Project Structure

adaptive-rag-assistant/\
│\
├── app.py\
├── requirements.txt\
├── .env\
└── README.md

------------------------------------------------------------------------

## ⚙️ Installation

1.  Clone Repository

``` bash
git clone https://github.com/vikashajithan/RAG_PROJECT.git
cd adaptive-rag-assistant
```

2.  Create Virtual Environment

``` bash
python -m venv venv
```

Activate:

Windows

``` bash
venv\Scripts\activate
```

Mac/Linux

``` bash
source venv/bin/activate
```

3.  Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔐 Environment Setup

Create a `.env` file in the root folder:

OPENAI_API_KEY=your_openai_api_key\
SERPAPI_API_KEY=your_serpapi_api_key

------------------------------------------------------------------------

## ▶ Run the Application

``` bash
streamlit run app.py
```

Open in browser:

http://localhost:8501

------------------------------------------------------------------------

## 🧪 Example Usage

1.  Upload a document (optional)\
2.  Enter your question\
3.  Receive answer from document\
4.  If not found → system automatically searches the web

------------------------------------------------------------------------

## 🛡 Hallucination Control

-   Strict document-grounded prompts\
-   Retriever quality validation\
-   Web fallback only when needed

Ensures reliable and trustworthy responses.

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Source citations\
-   Multi-file upload\
-   Persistent vector database\
-   Chat history memory\
-   Streaming responses\
-   Hybrid search (BM25 + Vector)

------------------------------------------------------------------------

## 📜 License

MIT License

------------------------------------------------------------------------

## 👨‍💻 Author

Vikash

Feel free to fork, star ⭐, and contribute!


