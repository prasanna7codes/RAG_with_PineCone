import os
from firecrawl import FirecrawlApp, ScrapeOptions
from llama_parse import LlamaParse
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# NEW imports for LangChain ≥0.2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# ============ CONFIG ============
from dotenv import load_dotenv
load_dotenv()  # loads your .env file

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_DIR = "./vector_store"

# ============ STEP 1: Crawl Website ============
def crawl_website(url: str):
    print(f"[INFO] Crawling website: {url}")
    app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    crawl_status = app.crawl_url(
        url,
        limit=5,
        scrape_options=ScrapeOptions(formats=["markdown"]),
        poll_interval=30
    )
    docs = []
    for page in crawl_status.data:
        if hasattr(page, "markdown") and page.markdown.strip():
            docs.append(Document(
                page_content=page.markdown,
                metadata={"source": page.url}
            ))
    print(f"[INFO] Crawled {len(docs)} pages.")
    return docs

# ============ STEP 2: Parse PDF ============
def parse_pdf(pdf_path: str):
    print(f"[INFO] Parsing PDF: {pdf_path}")
    parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
    parsed = parser.load_data(pdf_path)
    docs = [Document(page_content=d.text, metadata={"source": pdf_path}) for d in parsed]
    print(f"[INFO] Extracted {len(docs)} documents from PDF.")
    return docs

# ============ STEP 3: Chunk Documents ============
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Chunked into {len(chunks)} pieces.")
    return chunks

# ============ STEP 4: Store in ChromaDB ============
def store_in_chroma(chunks, client_id: str):
    print(f"[INFO] Storing in ChromaDB under namespace: {client_id}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        collection_name=client_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    db.add_documents(chunks)  # auto-persist now
    return db

# ============ STEP 5: Chatbot Query ============
def chatbot_query(client_id: str, question: str):
    print(f"[INFO] Retrieving from namespace: {client_id}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        collection_name=client_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    prompt = f"""Answer the following question using only the provided context.
    Context:
    {context}
    Question: {question}
    """
    answer = llm.invoke(prompt)
    return answer.content

# ============ MAIN DEMO ============
if __name__ == "__main__":
    client_id = "client_123"  # Example client namespace
    website_url = "https://www.thworks.org/"
    pdf_path = "NIPS-2017-attention-is-all-you-need-Paper.pdf"  # Replace with a real PDF file

    # Crawl + Parse
    web_docs = crawl_website(website_url)
    pdf_docs = parse_pdf(pdf_path)

    # Chunk
    all_docs = web_docs + pdf_docs
    chunks = chunk_documents(all_docs)

    # Store
    store_in_chroma(chunks, client_id)

    # Chat
    while True:
        q = input("Ask a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        ans = chatbot_query(client_id, q)
        print(f"\n[Chatbot]: {ans}\n")
