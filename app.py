import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from firecrawl import FirecrawlApp, ScrapeOptions
from llama_parse import LlamaParse
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_DIR = "./vector_store"

# ================= APP ===================
app = FastAPI(title="SaaS Chatbot Demo")

# Enable CORS for Postman / frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Pydantic Models =================
class QueryJSON(BaseModel):
    client_id: str
    question: str

# ================= FUNCTIONS =================
def crawl_website(url: str):
    app_fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    crawl_status = app_fc.crawl_url(
        url,
        limit=10,
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
    return docs

def parse_pdf(pdf_file: UploadFile):
    temp_path = f"./temp_{pdf_file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)
    parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
    parsed = parser.load_data(temp_path)
    os.remove(temp_path)
    docs = [Document(page_content=d.text, metadata={"source": pdf_file.filename}) for d in parsed]
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    return chunks

def store_in_chroma(chunks, client_id: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        collection_name=client_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    db.add_documents(chunks)
    return db

def chatbot_query(client_id: str, question: str):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        collection_name=client_id,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(question)
    if not relevant_docs:
        return "No relevant data found for this client."
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    prompt = f"""Answer the following question using only the provided context.
Context:
{context}
Question: {question}
"""
    answer = llm.invoke(prompt)
    return answer.content

# ================= API ROUTES =================
@app.post("/ingest/")
async def ingest_data(
    client_id: str = Form(...),       # required
    url: Optional[str] = Form(None),  # optional
    pdf: Optional[UploadFile] = File(None)  # optional
):
    """
    Accepts form-data:
    - client_id (required)
    - url (optional)
    - pdf (optional)
    """
    all_docs = []

    if url:
        web_docs = crawl_website(url)
        all_docs.extend(web_docs)

    if pdf:
        pdf_docs = parse_pdf(pdf)
        all_docs.extend(pdf_docs)

    if not all_docs:
        return JSONResponse({"error": "No URL or PDF provided"}, status_code=400)

    chunks = chunk_documents(all_docs)
    store_in_chroma(chunks, client_id)

    return {"message": f"Data ingested for client {client_id}", "chunks_count": len(chunks)}

@app.post("/query/")
async def query_chatbot_endpoint(json_data: QueryJSON):
    try:
        answer = chatbot_query(json_data.client_id, json_data.question)
        return {"answer": answer}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
