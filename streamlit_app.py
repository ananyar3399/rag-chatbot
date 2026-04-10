import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# --- Page config ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

# --- Header ---
st.title("🤖 RAG Chatbot")
st.markdown("Upload a PDF and ask questions about it. Answers are grounded strictly in your document.")
st.divider()

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --- Helper functions ---
def process_pdf(uploaded_file):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    # Embed and store
    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-english-v3.0"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    return vectorstore, len(documents), len(chunks)

def build_rag_chain(vectorstore):
    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-r-plus-08-2024",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions 
based strictly on the provided document context. 
If the answer is not found in the context, say 
'I could not find this information in the document.'
Keep answers clear and concise.

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

# --- Sidebar: PDF Upload ---
with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF... this may take a moment ⏳"):
                vectorstore, num_pages, num_chunks = process_pdf(uploaded_file)
                rag_chain, retriever = build_rag_chain(vectorstore)
                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []  # reset chat on new PDF

            st.success(f"✅ Ready! Loaded **{num_pages} pages** and **{num_chunks} chunks**")

    if st.session_state.pdf_name:
        st.info(f"📖 Active document: **{st.session_state.pdf_name}**")

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF")
    st.markdown("2. Ask any question")
    st.markdown("3. Get answers from your document")
    st.divider()
    st.markdown("Built with LangChain + Cohere + ChromaDB")

# --- Chat area ---
if not st.session_state.rag_chain:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                st.caption(f"📖 Source pages: {message['sources']}")

    # Chat input
    if question := st.chat_input("Ask a question about your document..."):

        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question)
                source_docs = st.session_state.retriever.invoke(question)
                pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in source_docs))

            st.markdown(answer)
            st.caption(f"📖 Source pages: {pages}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": pages
            })