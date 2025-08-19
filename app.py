import os
import shutil
import uuid
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     UploadFile, Request)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from firecrawl import FirecrawlApp, ScrapeOptions
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from llama_parse import LlamaParse
from pinecone import Pinecone
from pydantic import BaseModel
from supabase import Client, create_client

load_dotenv()

# ================= CONFIG =================
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
INDEX_NAME = "clinet-data-google"

# ================= INIT CLIENTS =================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ================= GLOBAL MODELS =================
# Use Google's cloud-based embedding model to avoid local downloads
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# ================= APP ===================
app = FastAPI(title="SaaS Chatbot Demo")

# ⚠️ IMPORTANT: Do NOT leave "*" in production
# - Dev: allow_origins=["*"] is fine
# - Prod: use your frontend domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to ["https://yourfrontend.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Pydantic Models =================
class QueryJSON(BaseModel):
    question: str

# ================= HELPERS =================
def normalize_domain(url: str) -> Optional[str]:
    """
    Normalize URLs to plain domain (example.com).
    Removes scheme, paths, www, etc.
    """
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"https://{url}")
    hostname = parsed.hostname.lower() if parsed.hostname else url.lower()
    # Drop leading "www."
    return hostname[4:] if hostname.startswith("www.") else hostname

# ================= SECURITY =================
async def get_client_id_from_key(request: Request, x_api_key: str = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key missing in headers")

    # Lookup client by API key
    response = (
        supabase.table("users_extra")
        .select("id, allowed_origins")
        .eq("public_api_key", x_api_key)
        .single()
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=403, detail="Invalid API Key provided")

    client_id = response.data.get("id")
    allowed_origins = response.data.get("allowed_origins") or []

    # Get origin/referer header
    origin = request.headers.get("origin") or request.headers.get("referer")
    if not origin:
        raise HTTPException(status_code=403, detail="Missing Origin header")

    parsed_origin = urlparse(origin)
    origin_domain = parsed_origin.hostname.lower() if parsed_origin.hostname else origin.lower()

    # Remove www. prefix
    if origin_domain.startswith("www."):
        origin_domain = origin_domain[4:]

    # Check allowed origins
    normalized_allowed = [d.lower().lstrip("www.") for d in allowed_origins]

    # ✅ Special handling: allow any localhost port
    if origin_domain == "localhost":
        return client_id

    if origin_domain not in normalized_allowed:
        raise HTTPException(status_code=403, detail=f"Unauthorized origin: {origin_domain}")

    return client_id

# ================= FUNCTIONS =================
def crawl_website(url: str):
    app_fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    crawl_status = app_fc.crawl_url(
        url,
        limit=10,
        scrape_options=ScrapeOptions(formats=["markdown"]),
        poll_interval=30,
    )
    docs = [
        Document(page_content=page.markdown, metadata={"source": page.url or ""})
        for page in crawl_status.data
        if hasattr(page, "markdown") and page.markdown.strip()
    ]
    return docs


def parse_pdf(pdf_file: UploadFile):
    # Ensure the temp directory exists
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique temporary file path
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{pdf_file.filename}")
    
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
        
        parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
        parsed = parser.load_data(temp_path)
        
        docs = [
            Document(page_content=d.text, metadata={"source": pdf_file.filename or ""})
            for d in parsed
        ]
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


def store_in_pinecone(chunks, client_id: str):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)  # batch embeddings

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vector_id = str(uuid.uuid4())
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
    query_embedding = embeddings_model.embed_query(question)
    results = index.query(
        vector=query_embedding, top_k=3, include_metadata=True, namespace=client_id
    )

    if not results.matches:
        return "I'm sorry, I couldn't find any relevant information."

    context = "\n\n".join([match.metadata.get("content", "") for match in results.matches])

    prompt = (
        f"Answer the following question using only the provided context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    answer = llm.invoke(prompt)
    return answer.content

# ================= API ROUTES =================
@app.post("/ingest/")
async def ingest_data(
    client_id: str = Form(...),
    url: Optional[str] = Form(None),
    pdf: Optional[UploadFile] = File(None),
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
    try:
        answer = chatbot_query(client_id, json_data.question)
        return {"answer": answer}
    except Exception as e:
        print(f"An error occurred during query: {e}")
        return JSONResponse({"error": "An internal server error occurred."}, status_code=500)
