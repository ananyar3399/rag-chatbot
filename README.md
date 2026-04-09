# 🤖 RAG Chatbot — Chat with Any PDF

A Retrieval-Augmented Generation (RAG) chatbot built with **LangChain**, **Cohere**, and **ChromaDB**. Upload any PDF and ask questions — the chatbot finds the most relevant sections and generates accurate, grounded answers.

---

## 🚀 Demo

```
=== RAG Chatbot — Chat with your PDF ===

Enter the path to your PDF file: C:\Users\hp\Desktop\sample.pdf

📄 Loading PDF: sample.pdf
✅ Loaded 12 pages

✂️  Chunking document...
✅ Created 87 chunks

🔢 Embedding and storing in ChromaDB...
✅ Stored in ChromaDB

🔗 Building RAG chain...
✅ RAG chain ready

💬 Chatbot ready! Type 'quit' to exit.

You: What is this document about?

🤖 Bot: This document is about neural networks and deep learning fundamentals,
covering topics such as forward propagation, backpropagation, and gradient descent.

📖 Source pages: {1, 2, 5}
```

---

## 🧠 How RAG Works

```
Your Question
     │
     ▼
Cohere Embeddings converts question to a vector
     │
     ▼
ChromaDB searches for the 3 most similar chunks
     │
     ▼
Relevant chunks + question sent to Cohere LLM
     │
     ▼
Accurate answer grounded in your document
```

Traditional chatbots hallucinate answers. RAG grounds every answer in your actual document — making it far more reliable and accurate.

---

## 🛠️ Tech Stack

- **Python** 3.x
- **LangChain** – document loaders, text splitters, prompt templates, chains
- **Cohere API** – LLM (`command-r-plus-08-2024`) + Embeddings (`embed-english-v3.0`)
- **ChromaDB** – local vector database for storing and searching embeddings
- **PyPDF** – PDF parsing
- **python-dotenv** – environment variable management

---

## 📚 Concepts Applied

### LangChain for LLM Application Development (DeepLearning.AI)
- `PyPDFLoader` — loads and parses PDF documents page by page
- `RecursiveCharacterTextSplitter` — chunks text into 500 character pieces with 50 character overlap
- `CohereEmbeddings` — converts text chunks into semantic vectors
- `Chroma` — stores and retrieves vectors from a local database
- `RunnablePassthrough` — modern LangChain pipe syntax for building chains

### ChatGPT Prompt Engineering for Developers (DeepLearning.AI)
- Structured system prompt that strictly grounds answers in document context
- Prompt design that handles out-of-context questions gracefully

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/ananyar3399/rag-chatbot.git
cd rag-chatbot
```

### 2. Install dependencies
```bash
pip install langchain langchain-cohere langchain-community langchain-text-splitters chromadb pypdf python-dotenv
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
python app.py
```

### 6. Enter your PDF path when prompted
```
Enter the path to your PDF file: C:\Users\hp\Desktop\yourfile.pdf
```

---

## 📁 Project Structure

```
rag-chatbot/
│
├── app.py          # Main application
├── .env            # API key (never commit this)
├── .gitignore      # Excludes .env and chroma_db/
└── README.md       # You're here
```

> ⚠️ The `chroma_db/` folder is auto-generated when you run the app. It stores your embeddings locally and is excluded from Git.

---

## 🔒 Important

Never upload your `.env` file to GitHub. Make sure your `.gitignore` contains:
```
.env
chroma_db/
```

---

## 📖 Courses Referenced

- [ChatGPT Prompt Engineering for Developers – DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- [LangChain for LLM Application Development – DeepLearning.AI](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)

---

## 👩‍💻 Author

Built by Ananya as a capstone project applying concepts from two DeepLearning.AI courses.