# 🕵️ Smart RAG Agent — PDF Chat + Web Search

An intelligent AI agent built with **LangChain**, **Cohere**, **ChromaDB**, and **Streamlit**. Upload any PDF and ask questions — the agent automatically decides whether to answer from your document or search the web using DuckDuckGo. Containerized with **Docker**, deployed on **Render**, and monitored with **LangSmith**.

Built as a capstone project combining concepts from two DeepLearning.AI courses:
- ✅ ChatGPT Prompt Engineering for Developers
- ✅ LangChain for LLM Application Development

🔗 **Live Demo:** https://rag-agent-hwra.onrender.com/

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

You: Compare what the document says about AI with recent developments
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

Every run is traced in **LangSmith** — showing which tools were called, token usage, and latency for each step.

---

## 🛠️ Tech Stack

- **Python 3.11** – core language
- **LangChain** – document loaders, text splitters, prompt templates, tool calling
- **Cohere API** – LLM (`command-r-plus-08-2024`) + Embeddings (`embed-english-v3.0`)
- **ChromaDB** – local vector database for semantic search
- **DuckDuckGo Search** – free web search (no API key needed)
- **Streamlit** – web UI
- **LangSmith** – agent monitoring, tracing, and observability
- **Docker** – containerization
- **Render** – cloud deployment
- **PyPDF** – PDF parsing
- **python-dotenv** – environment variable management

---

## 📚 Concepts Applied

### ChatGPT Prompt Engineering for Developers (DeepLearning.AI)
- Structured system prompt guiding the agent on when to use each tool
- Prompt design that handles both document-specific and general knowledge questions
- Iterative refinement of tool descriptions to improve agent decision making

### LangChain for LLM Application Development (DeepLearning.AI)
- `PyPDFLoader` — loads and parses PDF documents page by page
- `RecursiveCharacterTextSplitter` — chunks text with overlap to preserve context
- `CohereEmbeddings` — converts chunks into semantic vectors
- `Chroma` — stores and retrieves vectors from a local database
- `@tool` decorator — defines custom tools the agent can call
- `bind_tools` — connects tools to the LLM for tool calling
- `ToolMessage` — passes tool results back into the agent loop

---

## 📖 Project Evolution

This project was built iteratively — each stage adding a new layer of complexity:

### v1 — RAG Chatbot (terminal app)
A Python terminal app that loaded a PDF, chunked and embedded it into ChromaDB, and answered questions from the document using a LangChain RAG chain powered by Cohere.

**Key concepts:** PyPDFLoader, RecursiveCharacterTextSplitter, CohereEmbeddings, Chroma, RAG pipeline

### v2 — RAG Chatbot with Streamlit UI
Wrapped the terminal app in a clean Streamlit web interface. Users could upload PDFs through a sidebar, chat in a ChatGPT-style interface, and see source page references for every answer.

**Key concepts:** Streamlit file uploader, session state, chat UI, tempfile handling

### v3 — Smart RAG Agent
Upgraded from a simple RAG chain to a full AI agent with two tools — a PDF search tool (RAG) and a web search tool (DuckDuckGo). The agent intelligently decides which tool to use and shows the user which tool was used in each response.

**Key concepts:** Tool calling, agent loop, `bind_tools`, `ToolMessage`, DuckDuckGo search

### v4 — LangSmith Monitoring
Integrated LangSmith for full observability — every agent run is traced showing tool calls, inputs, outputs, token usage, and latency. No code changes required — configured via environment variables.

**Key concepts:** LangSmith tracing, observability, `LANGCHAIN_TRACING_V2`

### v5 — Docker + Render Deployment (current)
Containerized the app with Docker and deployed it to Render. The entire app runs in a portable container that can be deployed anywhere. Render automatically rebuilds and redeploys on every GitHub push.

**Key concepts:** Dockerfile, .dockerignore, environment variables in containers, cloud deployment

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ananyar3399/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install langchain langchain-cohere langchain-community langchain-text-splitters langchain-core chromadb pypdf python-dotenv streamlit cohere duckduckgo-search langsmith
```

### 3. Get your API keys
- **Cohere** — sign up at [dashboard.cohere.com](https://dashboard.cohere.com) (free, no credit card)
- **LangSmith** — sign up at [smith.langchain.com](https://smith.langchain.com) (free)

### 4. Create a `.env` file
```
COHERE_API_KEY=your_cohere_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=smart-rag-agent
```

### 5. Run locally
```bash
python -m streamlit run agent_app.py
```

---

## 🐳 Running with Docker

### Build the image
```bash
docker build -t rag-agent .
```

### Run the container
```bash
docker run -p 8501:8501 \
  -e COHERE_API_KEY=your_key \
  -e LANGCHAIN_TRACING_V2=true \
  -e LANGCHAIN_API_KEY=your_key \
  -e LANGCHAIN_PROJECT=smart-rag-agent \
  rag-agent
```

Open `http://localhost:8501` in your browser.

---

## ☁️ Deployment on Render

1. Push code to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repo
4. Set **Runtime** to **Docker**
5. Add environment variables (API keys)
6. Click **Deploy**

Render automatically rebuilds and redeploys on every `git push`.

---

## 📊 Monitoring with LangSmith

Every agent run is automatically traced in LangSmith. View at [smith.langchain.com](https://smith.langchain.com) → Projects → **smart-rag-agent**.

Each trace shows:
- Which tools were called and in what order
- Inputs and outputs for every step
- Token usage per call
- Latency for each step

---

## 📁 Project Structure

```
rag-chatbot/
│
├── agent_app.py        # Smart Agent UI (current version)
├── streamlit_app.py    # RAG Chatbot UI (v2)
├── app.py              # RAG Chatbot terminal app (v1)
├── Dockerfile          # Container configuration
├── .dockerignore       # Files excluded from Docker build
├── requirements.txt    # Python dependencies
├── .env                # API keys (never commit this)
├── .gitignore          # Excludes .env and chroma_db/
└── README.md           # You're here
```

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