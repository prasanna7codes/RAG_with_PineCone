# app.py

import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from firecrawl import FirecrawlApp, ScrapeOptions
from llama_parse import LlamaParse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pinecone import Pinecone
from supabase import create_client, Client  # MODIFIED: Import Supabase

load_dotenv()

# ================= CONFIG =================
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")  # ADDED
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # ADDED
INDEX_NAME = "client-data"

# ================= INIT CLIENTS =================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)  # ADDED: Init Supabase client

# ================= APP ===================
app = FastAPI(title="SaaS Chatbot Demo")

app.add_middleware(
    CORSMiddleware,
    # IMPORTANT: In production, you should restrict this to your frontend domain
    # and potentially a wildcard for your client's domains.
    allow_origins=["https://rag-chatbot-frontend-orcin.vercel.app/",
                   "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Pydantic Models =================
# MODIFIED: The client no longer sends their ID.
class QueryJSON(BaseModel):
    question: str


# ================= SECURITY DEPENDENCY (NEW) =================
async def get_client_id_from_key(x_api_key: str = Header(None)):
    """Dependency to verify API key and return the client_id (user's UUID)"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key missing in headers")

    # Query Supabase to find the user with this public_api_key
    response = (
        supabase.table("users_extra")
        .select("id")
        .eq("public_api_key", x_api_key)
        .single()
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=403, detail="Invalid API Key provided")

    client_id = response.data.get("id")
    return client_id


# ================= FUNCTIONS =================
def crawl_website(url: str):
    app_fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    crawl_status = app_fc.crawl_url(
        url, limit=10, scrape_options=ScrapeOptions(formats=["markdown"]), poll_interval=30
    )
    docs = [
        Document(page_content=page.markdown, metadata={"source": page.url or ""})
        for page in crawl_status.data
        if hasattr(page, "markdown") and page.markdown.strip()
    ]
    return docs


def parse_pdf(pdf_file: UploadFile):
    temp_path = f"./temp_{pdf_file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(pdf_file.file, f)
    parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
    parsed = parser.load_data(temp_path)
    os.remove(temp_path)
    docs = [
        Document(page_content=d.text, metadata={"source": pdf_file.filename or ""})
        for d in parsed
    ]
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def store_in_pinecone(chunks, client_id: str):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectors = []
    for chunk in chunks:
        vector_id = str(uuid.uuid4())
        embedding = embeddings_model.embed_query(chunk.page_content)
        vectors.append(
            (
                vector_id,
                embedding,
                {
                    "client_id": client_id,
                    "source": str(chunk.metadata.get("source") or ""),
                    "content": chunk.page_content,
                },
            )
        )
    index.upsert(vectors, namespace=client_id)
    return len(vectors)


def chatbot_query(client_id: str, question: str):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    query_embedding = embeddings_model.embed_query(question)
    results = index.query(
        vector=query_embedding, top_k=3, include_metadata=True, namespace=client_id
    )

    if not results.matches:
        return "I'm sorry, I couldn't find any relevant information to answer your question."

    context = "\n\n".join([match.metadata.get("content", "") for match in results.matches])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
    prompt = f"Answer the following question using only the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"
    answer = llm.invoke(prompt)
    return answer.content


# ================= API ROUTES =================
@app.post("/ingest/")
async def ingest_data(
    client_id: str = Form(...), url: Optional[str] = Form(None), pdf: Optional[UploadFile] = File(None)
):
    all_docs = []
    if url:
        all_docs.extend(crawl_website(url))
    if pdf:
        all_docs.extend(parse_pdf(pdf))
    if not all_docs:
        return JSONResponse({"error": "No URL or PDF provided"}, status_code=400)
    chunks = chunk_documents(all_docs)
    vectors_count = store_in_pinecone(chunks, client_id)
    return {"message": f"Data ingested for client {client_id}", "chunks_count": vectors_count}


@app.post("/query/")
async def query_chatbot_endpoint(
    json_data: QueryJSON, client_id: str = Depends(get_client_id_from_key)
):
    """
    MODIFIED: This endpoint is now protected.
    It requires a valid X-API-Key header.
    The client_id is securely retrieved based on the key, not from the request body.
    """
    try:
        answer = chatbot_query(client_id, json_data.question)
        return {"answer": answer}
    except Exception as e:
        print(f"An error occurred during query: {e}")  # Better logging
        return JSONResponse({"error": "An internal server error occurred."}, status_code=500)
