# 🕵️ Smart RAG Agent — Chat with PDFs + Web Search

An intelligent AI agent built with **LangChain**, **Cohere**, **ChromaDB**, and **Streamlit**. Upload any PDF and ask questions — the agent automatically decides whether to answer from your document or search the web using DuckDuckGo.

Built as a capstone project combining concepts from two DeepLearning.AI courses:
- ✅ ChatGPT Prompt Engineering for Developers
- ✅ LangChain for LLM Application Development

---

## 🚀 Demo

```
🕵️ Smart RAG Agent

Upload a PDF and ask anything:

You: What does the document say about neural networks?
🤖 Bot: Neural networks are computational models inspired by the human brain...
🔧 Tools used: 📄 PDF Search

You: Who is the current CEO of Google?
🤖 Bot: Sundar Pichai is the current CEO of Google...
🔧 Tools used: 🌐 Web Search

You: Compare what the document says about AI with recent news
🤖 Bot: According to your document... and based on recent web results...
🔧 Tools used: 📄 PDF Search, 🌐 Web Search
```

---

## 🧠 How the Agent Works

```
Your Question
     │
     ▼
Agent thinks: which tool should I use?
     │
     ├──► 📄 search_document (RAG)
     │         │
     │         ▼
     │    ChromaDB finds top 3 relevant chunks
     │    from your uploaded PDF
     │
     └──► 🌐 search_web (DuckDuckGo)
               │
               ▼
          Searches the internet for
          current events / general knowledge
               │
               ▼
         Final answer grounded in
         real sources — no hallucinations
```

The agent uses **tool calling** — it reasons about which tool is most appropriate, calls it, reads the result, and formulates a final answer. It can use both tools in a single response if needed.

---

## 🛠️ Tech Stack

- **Python** 3.x
- **LangChain** – document loaders, text splitters, prompt templates, tool calling
- **Cohere API** – LLM (`command-r-plus-08-2024`) + Embeddings (`embed-english-v3.0`)
- **ChromaDB** – local vector database for semantic search
- **DuckDuckGo Search** – free web search tool (no API key needed)
- **Streamlit** – web UI and deployment
- **PyPDF** – PDF parsing
- **python-dotenv** – environment variable management

---

## 📚 Concepts Applied

### ChatGPT Prompt Engineering for Developers (DeepLearning.AI)
- Structured system prompt that guides the agent on when to use each tool
- Prompt design that handles both document-specific and general knowledge questions

### LangChain for LLM Application Development (DeepLearning.AI)
- `PyPDFLoader` — loads and parses PDF documents
- `RecursiveCharacterTextSplitter` — chunks text into 500 character pieces with 50 character overlap
- `CohereEmbeddings` — converts chunks into semantic vectors
- `Chroma` — stores and retrieves vectors from a local database
- `@tool` decorator — defines custom tools the agent can call
- `bind_tools` — connects tools to the LLM for tool calling
- `ToolMessage` — passes tool results back into the agent loop

---

## 📖 Project Evolution

This project was built in stages — each stage adding a new layer of complexity:

### v1 — RAG Chatbot (terminal app)
A Python terminal app that loaded a PDF, chunked and embedded it into ChromaDB, and used a LangChain RAG chain to answer questions from the document. Powered by Cohere.

**Key concepts:** PyPDFLoader, RecursiveCharacterTextSplitter, CohereEmbeddings, Chroma, RAG pipeline

### v2 — RAG Chatbot with Streamlit UI
Wrapped the terminal app in a clean Streamlit web interface. Users could upload PDFs through a sidebar, chat in a ChatGPT-style interface, and see source page references for every answer.

**Key concepts:** Streamlit file uploader, session state, chat UI, tempfile handling

### v3 — Smart RAG Agent (current)
Upgraded from a simple RAG chain to a full AI agent with two tools — a PDF search tool (RAG) and a web search tool (DuckDuckGo). The agent now intelligently decides which tool to use for each question, and shows the user which tool was used in the response.

**Key concepts:** Tool calling, agent loop, `bind_tools`, `ToolMessage`, DuckDuckGo search

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ananyar3399/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install langchain langchain-cohere langchain-community langchain-text-splitters langchain-core chromadb pypdf python-dotenv streamlit cohere duckduckgo-search
```

### 3. Get a free Cohere API key
- Sign up at [dashboard.cohere.com](https://dashboard.cohere.com)
- Copy your API key from the dashboard (no credit card required)

### 4. Create a `.env` file
```
COHERE_API_KEY=your_cohere_api_key_here
```

### 5. Run the app
```bash
python -m streamlit run agent_app.py
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── agent_app.py        # Smart Agent UI (current version)
├── streamlit_app.py    # RAG Chatbot UI (v2)
├── app.py              # RAG Chatbot terminal app (v1)
├── requirements.txt    # Dependencies for deployment
├── .env                # API key (never commit this)
├── .gitignore          # Excludes .env and chroma_db/
└── README.md           # You're here
```

> ⚠️ The `chroma_db/` and `chroma_db_agent/` folders are auto-generated when you run the app and are excluded from Git.

---

## 🔒 Important

Never upload your `.env` file to GitHub. Make sure your `.gitignore` contains:
```
.env
chroma_db/
chroma_db_agent/
```

---

## 📖 Courses Referenced

- [ChatGPT Prompt Engineering for Developers – DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [LangChain for LLM Application Development – DeepLearning.AI](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
