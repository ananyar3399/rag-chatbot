import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# --- Step A: Load the PDF ---
def load_pdf(path):
    print(f"\n Loading PDF: {path}")
    loader = PyPDFLoader(path)
    documents = loader.load()
    print(f" Loaded {len(documents)} pages")
    return documents

# --- Step B: Chunk the document ---
def chunk_documents(documents):
    print("\n  Chunking document...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f" Created {len(chunks)} chunks")
    return chunks

# --- Step C: Embed and store in ChromaDB ---
def create_vectorstore(chunks):
    print("\n Embedding and storing in ChromaDB...")
    embeddings = CohereEmbeddings(
        cohere_api_key=cohere_api_key,
        model="embed-english-v3.0"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(" Stored in ChromaDB")
    return vectorstore

# --- Step D: Build the RAG chain ---
def build_rag_chain(vectorstore):
    print("\n Building RAG chain...")

    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-r-plus-08-2024",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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

    print(" RAG chain ready")
    return rag_chain, retriever

# --- Step E: Chat loop ---
def chat(rag_chain, retriever):
    print("\n Chatbot ready! Type 'quit' to exit.\n")
    print("=" * 50)
    while True:
        question = input("\nYou: ").strip()
        if question.lower() == "quit":
            print("Goodbye!")
            break
        if not question:
            continue

        answer = rag_chain.invoke(question)
        print(f"\n Bot: {answer}")

        # Show which pages the answer came from
        source_docs = retriever.invoke(question)
        pages = set(doc.metadata.get("page", 0) + 1 for doc in source_docs)
        print(f" Source pages: {pages}")
        print("-" * 50)

# --- Main ---
if __name__ == "__main__":
    print("=== RAG Chatbot — Chat with your PDF ===")
    pdf_path = input("\nEnter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(" File not found. Please check the path and try again.")
        exit()

    documents = load_pdf(pdf_path)
    chunks = chunk_documents(documents)
    vectorstore = create_vectorstore(chunks)
    rag_chain, retriever = build_rag_chain(vectorstore)
    chat(rag_chain, retriever)