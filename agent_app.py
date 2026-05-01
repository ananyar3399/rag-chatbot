import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# --- Page config ---
st.set_page_config(
    page_title="Smart Agent",
    page_icon="🕵️",
    layout="centered"
)

st.title("🕵️ Smart RAG Agent")
st.markdown("""
This agent intelligently decides whether to answer from:
- 📄 **Your uploaded PDF** — for document-specific questions
- 🌐 **The web** — for general knowledge and current events
""")
st.divider()

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_with_tools" not in st.session_state:
    st.session_state.llm_with_tools = None
if "tools_map" not in st.session_state:
    st.session_state.tools_map = {}
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --- PDF processing ---
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-english-v3.0"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db_agent"
    )
    return vectorstore, len(documents), len(chunks)

# --- Build tools and LLM ---
def build_agent(vectorstore):
    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-r-plus-08-2024",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Tool 1: RAG
    @tool
    def search_document(query: str) -> str:
        """Search the uploaded PDF document to answer questions about its content.
        Use this when the question is about the uploaded document."""
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the document."
        return "\n\n".join(doc.page_content for doc in docs)

    # Tool 2: Web search
    @tool
    def search_web(query: str) -> str:
        """Search the internet for current events or general knowledge
        not covered in the uploaded document."""
        search = DuckDuckGoSearchRun()
        return search.run(query)

    tools = [search_document, search_web]
    tools_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    return llm_with_tools, tools_map

# --- Run agent loop ---
def run_agent(llm_with_tools, tools_map, question):
    system = """You are a smart assistant with access to two tools:
1. search_document: searches the uploaded PDF document
2. search_web: searches the internet

Think carefully about which tool to use:
- Use search_document for questions about the uploaded document
- Use search_web for general knowledge or current events
- You can use both tools if needed"""

    messages = [
        {"role": "system", "content": system},
        HumanMessage(content=question)
    ]

    tool_used = []

    # Agent loop — max 5 iterations
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls — we have the final answer
        if not response.tool_calls:
            return response.content, tool_used

        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_used.append(tool_name)

            tool_result = tools_map[tool_name].invoke(tool_args)

            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

    return "I was unable to find a satisfactory answer.", tool_used

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF... ⏳"):
                vectorstore, num_pages, num_chunks = process_pdf(uploaded_file)
                llm_with_tools, tools_map = build_agent(vectorstore)
                st.session_state.llm_with_tools = llm_with_tools
                st.session_state.tools_map = tools_map
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

            st.success(f"✅ Ready! Loaded **{num_pages} pages**, **{num_chunks} chunks**")

    if st.session_state.pdf_name:
        st.info(f"📖 Active: **{st.session_state.pdf_name}**")

    st.divider()
    st.markdown("**Agent tools:**")
    st.markdown("📄 RAG — answers from your PDF")
    st.markdown("🌐 DuckDuckGo — searches the web")
    st.divider()
    st.markdown("Built with LangChain + Cohere + ChromaDB")

# --- Chat area ---
if not st.session_state.llm_with_tools:
    st.info("👈 Upload a PDF from the sidebar to get started!")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "tools_used" in message and message["tools_used"]:
                tool_labels = {
                    "search_document": "📄 PDF Search",
                    "search_web": "🌐 Web Search"
                }
                labels = [tool_labels.get(t, t) for t in message["tools_used"]]
                st.caption(f"🔧 Tools used: {', '.join(labels)}")

    if question := st.chat_input("Ask anything — from your document or the web..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Agent thinking... 🤔"):
                answer, tools_used = run_agent(
                    st.session_state.llm_with_tools,
                    st.session_state.tools_map,
                    question
                )

            st.markdown(answer)

            if tools_used:
                tool_labels = {
                    "search_document": "📄 PDF Search",
                    "search_web": "🌐 Web Search"
                }
                labels = [tool_labels.get(t, t) for t in tools_used]
                st.caption(f"🔧 Tools used: {', '.join(labels)}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "tools_used": tools_used
            })