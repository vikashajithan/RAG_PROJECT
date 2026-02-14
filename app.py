import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_community.utilities import SerpAPIWrapper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Initialize LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# --------------------------------------------------
# Auto Detect & Load File
# --------------------------------------------------
def load_file(file_path):
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")

# --------------------------------------------------
# Split Documents
# --------------------------------------------------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

# --------------------------------------------------
# Create Vector Store
# --------------------------------------------------
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

# --------------------------------------------------
# Build RAG Chain (Modern LCEL)
# --------------------------------------------------
def build_rag_chain(retriever):

    prompt = ChatPromptTemplate.from_template("""
You are a document-grounded assistant.

Use ONLY the information from the provided context.
If the answer is not present in the context, respond exactly:

NOT_FOUND

Context:
{context}

Question:
{question}

Answer:
""")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

# --------------------------------------------------
# Web Search (SerpAPI)
# --------------------------------------------------
def web_search(query):
    search = SerpAPIWrapper()
    return search.run(query)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Adaptive RAG Assistant", page_icon="🤖")
st.title("🤖 Adaptive RAG Assistant")

uploaded_file = st.file_uploader(
    "Upload PDF, TXT, DOCX (optional)",
    type=["pdf", "txt", "docx"]
)

question = st.text_input("Ask your question")

# --------------------------------------------------
# Build Knowledge Base
# --------------------------------------------------
if uploaded_file and "vectorstore" not in st.session_state:

    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Indexing document..."):
        docs = load_file(uploaded_file.name)
        chunks = split_docs(docs)
        vectorstore = create_vectorstore(chunks)

        st.session_state.vectorstore = vectorstore

    st.success("Document indexed successfully!")

# --------------------------------------------------
# Answer Logic
# --------------------------------------------------
if st.button("Get Answer") and question:

    with st.spinner("Thinking..."):

        # Case 1: Document exists
        if "vectorstore" in st.session_state:

            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )

            # Modern LangChain way
            docs = retriever.invoke(question)

            context_text = " ".join(
                [doc.page_content for doc in docs]
            )

            # If weak or empty context → fallback to web
            if len(context_text.strip()) < 200:

                web_data = web_search(question)

                final_answer = llm.invoke(
                    f"Answer using this web information:\n{web_data}"
                ).content

            else:
                rag_chain = build_rag_chain(retriever)
                result = rag_chain.invoke(question).content

                if "NOT_FOUND" in result:
                    web_data = web_search(question)

                    final_answer = llm.invoke(
                        f"Answer using this web information:\n{web_data}"
                    ).content
                else:
                    final_answer = result

        # Case 2: No document uploaded → direct web search
        else:
            web_data = web_search(question)

            final_answer = llm.invoke(
                f"Answer using this web information:\n{web_data}"
            ).content

    st.markdown("### ✅ Answer")
    st.write(final_answer)
