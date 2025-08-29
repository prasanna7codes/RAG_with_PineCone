import os
import shutil
import uuid
import datetime
import re
from typing import Optional, List, Dict
from urllib.parse import urlparse, urljoin

from dotenv import load_dotenv
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, Request, Path
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from PyPDF2 import PdfReader
from collections import defaultdict

# --- kept for your legacy /ingest/ endpoint + chatbot ---
from firecrawl import FirecrawlApp, ScrapeOptions
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from llama_parse import LlamaParse
from pinecone import Pinecone
from supabase import Client, create_client
import resend
import jwt

# ================= CONFIG =================
load_dotenv()

FIRECRAWL_API_KEY     = os.getenv("FIRECRAWL_API_KEY")
LLAMA_API_KEY         = os.getenv("LLAMA_API_KEY")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
SUPABASE_URL          = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
SUPABASE_ANON_KEY     = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_JWT_SECRET   = os.getenv("SUPABASE_JWT_SECRET")
INDEX_NAME            = os.getenv("INDEX_NAME", "clinet-data-google")
resend.api_key        = os.getenv("RESEND_API_KEY")

ALLOWED_WIDGET_ORIGINS = [
    o.strip() for o in (os.getenv("ALLOWED_WIDGET_ORIGINS") or "").split(",") if o.strip()
]

# ================= INIT CLIENTS =================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ================= GLOBAL MODELS =================
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=GEMINI_API_KEY
)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# ================= APP ===================
app = FastAPI(title="SaaS Chatbot + Live Handoff + Lifetime Credits")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_WIDGET_ORIGINS or [
        "https://rag-cloud-embedding-frontend.vercel.app",
        "https://chatbot-insight-opal.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: catch-all OPTIONS
@app.options("/{rest_of_path:path}")
def preflight_handler():
    return JSONResponse({"ok": True})

# ================= Pydantic Models =================
class QueryJSON(BaseModel):
    question: str

class FeedbackJSON(BaseModel):
    botResponse: str
    userContact: str

class LiveRequestJSON(BaseModel):
    requested_by_contact: Optional[str] = None

class PreviewResponse(BaseModel):
    domain: str
    discovered_pages: int
    top_paths: List[str]
    sample_urls: List[str]
    allowed_pages_for_plan: int  # now = credits remaining (website)
    allowed: bool                # discovered_pages <= remaining ?

class IngestURLJSON(BaseModel):
    url: str
    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    force_limit: Optional[int] = None

# ================= HELPERS =================
def normalize_domain(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"https://{url}")
    hostname = (parsed.hostname or url).lower()
    return hostname[4:] if hostname.startswith("www.") else hostname

def _normalize_url_with_scheme(u: str) -> str:
    if not u:
        return u
    if not u.startswith("http://") and not u.startswith("https://"):
        return "https://" + u
    return u

def send_client_email_resend(to_email: str, bot_response: str, user_contact: str):
    subject = "New Feedback Submitted on Your Chatbot"
    html_body = f"""
    <h2>New Feedback Received</h2>
    <p><b>Bot Response:</b> {bot_response}</p>
    <p><b>User Contact:</b> {user_contact}</p>
    <p>Regards,<br/>InsightBot</p>
    """
    return resend.Emails.send({
        "from": "onboarding@resend.dev",
        "to": [to_email],
        "subject": subject,
        "html": html_body
    })

# ================= SECURITY (HEADERS) =================
async def get_client_id_from_key(
    request: Request,
    x_api_key: str = Header(None, alias="X-API-Key"),
    x_client_domain: str = Header(None, alias="X-Client-Domain"),
) -> str:
    # Let CORS preflight pass
    if request.method == "OPTIONS":
        return "preflight"

    if not x_api_key:
        raise HTTPException(status_code=401, detail="API Key missing")

    resp = (
        supabase.table("users_extra")
        .select("id, allowed_origins")
        .eq("public_api_key", x_api_key)
        .single()
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    client_id = resp.data["id"]
    allowed_origins = (resp.data.get("allowed_origins") or [])
    normalized_allowed = [d.lower().lstrip("www.") for d in allowed_origins]

    claimed = (x_client_domain or "").lower().lstrip("www.")
    origin_hdr = request.headers.get("origin") or request.headers.get("referer")
    origin_host = ""
    if origin_hdr:
        p = urlparse(origin_hdr)
        origin_host = (p.hostname or "").lower()
        if origin_host.startswith("www."):
            origin_host = origin_host[4:]

    candidates = [c for c in [claimed, origin_host] if c]
    if normalized_allowed and not any(c in normalized_allowed for c in candidates):
        raise HTTPException(status_code=403, detail=f"Unauthorized origin: {claimed or origin_host}")

    return client_id

async def get_session_id(x_session_id: str = Header(None, alias="X-Session-Id")) -> str:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    return x_session_id

# ================= CREDITS HELPERS (lifetime) =================
def get_credits_remaining(client_id: str) -> Dict[str, int]:
    row = (
        supabase.table("client_credits")
        .select("website_pages_remaining, pdf_pages_remaining")
        .eq("client_id", client_id)
        .single()
        .execute()
    ).data or {}
    return {
        "website": int(row.get("website_pages_remaining", 0) or 0),
        "pdf": int(row.get("pdf_pages_remaining", 0) or 0),
    }

def ensure_credits_row(client_id: str) -> None:
    """
    Ensure there is a row in client_credits for this client.
    Uses an upsert with defaults (0,0) to avoid PGRST116 when .single() finds 0 rows.
    """
    try:
        res = (
            supabase.table("client_credits")
            .select("client_id")
            .eq("client_id", client_id)
            .execute()
        )
        exists = bool(res.data and len(res.data) > 0)
        if not exists:
            supabase.table("client_credits").upsert({
                "client_id": client_id,
                "website_pages_remaining": 0,
                "pdf_pages_remaining": 0,
            }, on_conflict="client_id").execute()
    except Exception:
        # If select failed for any reason, just attempt an upsert anyway.
        supabase.table("client_credits").upsert({
            "client_id": client_id,
            "website_pages_remaining": 0,
            "pdf_pages_remaining": 0,
        }, on_conflict="client_id").execute()

def reserve_website_credits(client_id: str, pages: int) -> bool:
    # SQL RPC: reserve_website_pages(p_client_id uuid, p_pages int) returns boolean
    res = supabase.rpc("reserve_website_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()
    return bool(res.data)

def refund_website_credits(client_id: str, pages: int) -> None:
    if pages > 0:
        supabase.rpc("refund_website_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()

def reserve_pdf_credits(client_id: str, pages: int) -> bool:
    res = supabase.rpc("reserve_pdf_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()
    return bool(res.data)

def refund_pdf_credits(client_id: str, pages: int) -> None:
    if pages > 0:
        supabase.rpc("refund_pdf_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()

# ================= DISCOVERY (sitemap → Firecrawl map) =================
SITEMAP_RE = re.compile(r"<loc>(.*?)</loc>", re.IGNORECASE)

def _http_get(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "InsightBot/1.0"})
        if r.ok:
            return r.text
    except Exception:
        return None
    return None

def fetch_sitemap_urls(domain_or_url: str, timeout=15) -> list:
    base = domain_or_url if domain_or_url.startswith("http") else f"https://{domain_or_url}"
    p = urlparse(base)
    root = f"{p.scheme}://{p.netloc}"
    urls = set()

    sm_main = _http_get(urljoin(root, "/sitemap.xml"), timeout=timeout)
    if sm_main:
        urls.update(SITEMAP_RE.findall(sm_main))

    robots = _http_get(urljoin(root, "/robots.txt"), timeout=timeout)
    if robots:
        for line in robots.splitlines():
            if line.lower().startswith("sitemap:"):
                sm_url = line.split(":", 1)[1].strip()
                sm_body = _http_get(sm_url, timeout=timeout)
                if sm_body:
                    urls.update(SITEMAP_RE.findall(sm_body))

    urls = {u for u in urls if urlparse(u).netloc == p.netloc}
    return sorted(urls)

def firecrawl_map(url: str) -> list:
    # REST map endpoint is fast & cheap to enumerate
    endpoint = "https://api.firecrawl.dev/v1/map"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
    payload = {"url": url}
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        urls = data.get("urls") or data.get("data") or []
        host = urlparse(url if "://" in url else f"https://{url}").netloc
        return [u for u in urls if urlparse(u).netloc == host]
    except Exception:
        return []

def canonicalize_urls(urls: list) -> list:
    out = set()
    for u in urls:
        p = urlparse(u if "://" in u else f"https://{u}")
        path = p.path
        out.add(f"{p.scheme}://{p.netloc}{path}")
    return sorted(out)

# ================= FIRECRAWL CRAWL (REST) =================
def firecrawl_crawl(url: str, *, limit: int, include_paths=None, exclude_paths=None) -> list:
    endpoint = "https://api.firecrawl.dev/v1/crawl"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}", "Content-Type": "application/json"}

    # Normalize + guard limit
    url = _normalize_url_with_scheme(url)
    safe_limit = max(1, int(limit))
    if safe_limit > 2000:
        safe_limit = 2000

    payload = {
        "url": url,
        "crawlEntireDomain": True,
        # "sitemap": "include",  # optional; some deployments reject this field
        "maxDiscoveryDepth": 4,
        "limit": safe_limit,
        "scrapeOptions": {"formats": ["markdown", "metadata"]},
    }
    if include_paths:
        payload["includePaths"] = include_paths
    if exclude_paths:
        payload["excludePaths"] = exclude_paths

    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    if not r.ok:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise HTTPException(
            status_code=502,
            detail=f"Firecrawl crawl start failed: {r.status_code} {body}"
        )

    data = r.json()
    job_id = data.get("jobId") or data.get("id")
    if not job_id:
        raise HTTPException(status_code=502, detail=f"Firecrawl did not return job id: {data}")

    status_ep = f"{endpoint}/{job_id}"
    pages = []

    import time
    for _ in range(120):  # ~4 minutes
        s = requests.get(status_ep, headers=headers, timeout=30)
        if not s.ok:
            try:
                body = s.json()
            except Exception:
                body = s.text
            raise HTTPException(status_code=502, detail=f"Firecrawl status error: {s.status_code} {body}")

        body = s.json()
        state = body.get("status") or body.get("state")
        if state in ("completed", "finished", "succeeded"):
            pages = body.get("data") or body.get("pages") or []
            break
        if state in ("failed", "error"):
            raise HTTPException(status_code=502, detail=f"Firecrawl crawl failed: {body}")

        time.sleep(2)

    return pages

# ================= CHUNKING =================
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ================= PREVIEW (uses remaining website credits) =================
@app.get("/ingest/preview", response_model=PreviewResponse)
async def ingest_preview(url: str, client_id: str = Depends(get_client_id_from_key)):
    # Make sure the credits row exists so .single() below never 404s
    ensure_credits_row(client_id)

    domain = normalize_domain(url) or url
    urls = fetch_sitemap_urls(domain)
    if not urls:
        urls = firecrawl_map(domain)
    urls = canonicalize_urls(urls)

    buckets = defaultdict(int)
    for u in urls:
        p = urlparse(u)
        part = "/" + (p.path.split("/", 2)[1] if p.path.count("/") >= 1 and p.path != "/" else "")
        buckets[part] += 1
    top_paths = [k for k, _ in sorted(buckets.items(), key=lambda x: (-x[1], x[0]))[:10]]

    credits = get_credits_remaining(client_id)
    remaining = credits["website"]
    allowed = len(urls) <= remaining if remaining > 0 else False

    return {
        "domain": domain,
        "discovered_pages": len(urls),
        "top_paths": top_paths,
        "sample_urls": urls[:10],
        "allowed_pages_for_plan": remaining,  # rename in UI later if you want
        "allowed": allowed
    }

# ================= INGEST WEBSITE (lifetime credits) =================
@app.post("/ingest/url")
async def ingest_url(
    data: IngestURLJSON,
    client_id: str = Depends(get_client_id_from_key),
):
    ensure_credits_row(client_id)

    # 1) discover & compute how many we *intend* to crawl
    preview = await ingest_preview(url=data.url, client_id=client_id)
    credits = get_credits_remaining(client_id)
    remaining = credits["website"]

    if remaining <= 0:
        raise HTTPException(403, detail="No website page credits remaining. Please top up.")

    discovered = int(preview["discovered_pages"])
    force = int(data.force_limit) if data.force_limit is not None else None
    if force is not None and force < 0:
        force = 0

    intended_raw = discovered if force is None else min(discovered, force)
    intended = min(intended_raw, remaining)

    if intended <= 0:
        raise HTTPException(403, detail="Requested 0 pages after applying remaining credits/limit.")

    # 2) reserve credits atomically
    reserved = max(1, intended)  # guard
    ok = reserve_website_credits(client_id, reserved)
    if not ok:
        # someone else might have consumed credits concurrently
        fresh = get_credits_remaining(client_id)["website"]
        raise HTTPException(403, detail=f"Insufficient website credits. Remaining: {fresh}")

    # 3) crawl with 'reserved' cap; refund if we used fewer
    try:
        pages = firecrawl_crawl(
            data.url,
            limit=reserved,
            include_paths=data.include_paths,
            exclude_paths=data.exclude_paths
        )

        actual = 0
        docs = []
        for p in pages:
            md = p.get("markdown") or ""
            if not md.strip():
                continue
            src = p.get("url") or ""
            title = ((p.get("metadata") or {}).get("title")) or ""
            docs.append(Document(page_content=md, metadata={"source": src, "title": title}))
            actual += 1

        if not docs:
            # Nothing extracted: refund everything
            refund_website_credits(client_id, reserved)
            return JSONResponse({"error": "No content extracted"}, status_code=400)

        # chunk (indexing to vector store is your choice; here we only enforce credits)
        chunks = chunk_documents(docs)
        chunks_count = len(chunks)

        # Refund any unused reservations (Firecrawl returned fewer than reserved)
        if actual < reserved:
            refund_website_credits(client_id, reserved - actual)

        # Log event
        supabase.table("ingestion_events").insert({
            "client_id": client_id,
            "event_type": "url",
            "source": data.url,
            "chunks_stored": int(chunks_count),
            "discovered_pages": discovered,
            "website_pages_crawled": int(actual)
        }).execute()

        return {"message": "Website ingested", "chunks_count": chunks_count, "used_pages": actual}

    except HTTPException:
        # bubble up known http errors (already meaningful)
        refund_website_credits(client_id, reserved)
        raise
    except Exception as e:
        # On failure refund full reserved
        refund_website_credits(client_id, reserved)
        raise

# ================= INGEST PDF (lifetime credits) =================
@app.post("/ingest/pdf")
async def ingest_pdf(
    client_id: str = Depends(get_client_id_from_key),
    pdf: UploadFile = File(...),
):
    ensure_credits_row(client_id)

    credits = get_credits_remaining(client_id)
    remaining_pdf = credits["pdf"]
    if remaining_pdf <= 0:
        raise HTTPException(403, detail="No PDF page credits remaining. Please top up.")

    # temp save to count pages
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{pdf.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    # count pages
    try:
        try:
            reader = PdfReader(temp_path)
            page_count = len(reader.pages)
        except Exception:
            # If PyPDF2 fails to count, deny rather than over-consume
            raise HTTPException(400, detail="Unable to read PDF pages. Please upload a valid PDF.")

        if page_count <= 0:
            os.remove(temp_path)
            raise HTTPException(400, detail="PDF has 0 pages.")

        if page_count > remaining_pdf:
            os.remove(temp_path)
            raise HTTPException(403, detail=f"Not enough PDF credits. Needed {page_count}, have {remaining_pdf}.")

        # Reserve exactly the number of pages this file needs
        ok = reserve_pdf_credits(client_id, page_count)
        if not ok:
            fresh = get_credits_remaining(client_id)["pdf"]
            os.remove(temp_path)
            raise HTTPException(403, detail=f"Insufficient PDF credits. Remaining: {fresh}")

        # Parse with LlamaParse
        try:
            parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
            parsed = parser.load_data(temp_path)
            docs = [Document(page_content=d.text, metadata={"source": pdf.filename or ""}) for d in parsed]
        except Exception:
            # refund on parse failure
            refund_pdf_credits(client_id, page_count)
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not docs:
            refund_pdf_credits(client_id, page_count)
            return JSONResponse({"error": "No content extracted from PDF"}, status_code=400)

        # chunk (index later if you want)
        chunks = chunk_documents(docs)
        chunks_count = len(chunks)

        # Log event
        supabase.table("ingestion_events").insert({
            "client_id": client_id,
            "event_type": "pdf",
            "source": pdf.filename,
            "chunks_stored": int(chunks_count),
            "pdf_pages": int(page_count)
        }).execute()

        return {"message": "PDF ingested", "chunks_count": chunks_count, "pdf_pages": page_count}

    except:
        # defensive: ensure temp removed if we raised before the inner finally
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# ================== CHATBOT + LIVE HANDOFF (unchanged) ==================
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
        if hasattr(page, "markdown") and page.markdown and page.markdown.strip()
    ]
    return docs

def parse_pdf(pdf_file: UploadFile):
    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{pdf_file.filename}")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)
        parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
        parsed = parser.load_data(temp_path)
        docs = [Document(page_content=d.text, metadata={"source": pdf_file.filename or ""}) for d in parsed]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return docs

def chunk_documents_legacy(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def store_in_pinecone(chunks, client_id: str):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embeddings_model.embed_documents(texts)
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
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True, namespace=client_id)
    if not results.matches:
        return "I'm sorry, I couldn't find any relevant information."
    context = "\n\n".join([match.metadata.get("content", "") for match in results.matches])
    prompt = f"Answer the following question using only the provided context.\n\nContext:\n{context}\n\nQuestion: {question}"
    answer = llm.invoke(prompt)
    return answer.content

@app.post("/ingest/")  # legacy combined route (does not enforce credits)
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
    chunks = chunk_documents_legacy(all_docs)
    vectors_count = store_in_pinecone(chunks, client_id)
    return {"message": f"Data ingested for client {client_id}", "chunks_count": vectors_count}

@app.post("/query/")
async def query_chatbot_endpoint(
    json_data: QueryJSON,
    client_id: str = Depends(get_client_id_from_key),
    session_id: str = Depends(get_session_id),
):
    try:
        answer = chatbot_query(client_id, json_data.question)
        supabase.table("chat_logs").insert({
            "client_id": client_id,
            "session_id": session_id,
            "user_message": json_data.question,
            "bot_response": answer
        }).execute()
        return {"answer": answer}
    except Exception as e:
        print("query error:", e)
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.post("/feedback/")
async def feedback_endpoint(
    json_data: FeedbackJSON,
    client_id: str = Depends(get_client_id_from_key),
    session_id: str = Depends(get_session_id),
):
    try:
        supabase.table("chat_feedback").insert({
            "client_id": client_id,
            "session_id": session_id,
            "bot_response": json_data.botResponse,
            "user_contact": json_data.userContact
        }).execute()

        email_result = supabase.rpc("get_client_email", {"client_id": client_id}).execute()
        client_email = email_result.data
        if client_email:
            send_client_email_resend(client_email, json_data.botResponse, json_data.userContact)

        return {"message": "Feedback received and client notified"}
    except Exception as e:
        print("feedback error:", e)
        return JSONResponse({"error": "Internal server error"}, status_code=500)

# ========== LIVE HANDOFF ==========
def mint_visitor_jwt(*, client_id: str, session_id: str, conversation_id: str, ttl_minutes: int = 30) -> str:
    if not SUPABASE_JWT_SECRET:
        raise RuntimeError("Missing SUPABASE_JWT_SECRET")
    now = datetime.datetime.utcnow()
    sub = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{client_id}:{session_id}"))
    payload = {
        "aud": "authenticated",
        "sub": sub,
        "exp": now + datetime.timedelta(minutes=ttl_minutes),
        "iat": now,
        "role": "authenticated",
        "actor": "visitor",
        "client_id": client_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
    }
    token = jwt.encode(payload, SUPABASE_JWT_SECRET, algorithm="HS256")
    print("ISSUED VISITOR JWT PAYLOAD:", payload)
    return token

@app.post("/live/request")
async def live_request(
    payload: LiveRequestJSON,
    client_id: str = Depends(get_client_id_from_key),
    session_id: str = Depends(get_session_id),
):
    try:
        resp = (
            supabase.table("live_conversations")
            .select("id,status,created_at")
            .eq("client_id", client_id)
            .eq("session_id", session_id)
            .in_("status", ["pending", "active"])
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if rows:
            conversation_id = rows[0]["id"]
            current_status = rows[0]["status"]
        else:
            conversation_id = str(uuid.uuid4())
            current_status = "pending"
            supabase.table("live_conversations").insert({
                "id": conversation_id,
                "client_id": client_id,
                "session_id": session_id,
                "status": current_status,
                "requested_by_contact": payload.requested_by_contact,
            }).execute()

            supabase.table("live_messages").insert({
                "conversation_id": conversation_id,
                "sender_type": "system",
                "message": "Visitor requested human assistance."
            }).execute()

        token = mint_visitor_jwt(
            client_id=client_id,
            session_id=session_id,
            conversation_id=conversation_id,
            ttl_minutes=30
        )
        return {"conversation_id": conversation_id, "supabase_jwt": token, "status": current_status}

    except HTTPException:
        raise
    except Exception as e:
        print("live_request error:", e)
        raise HTTPException(status_code=500, detail="Unable to create live conversation")

@app.post("/live/join")
async def live_join(
    client_id: str = Depends(get_client_id_from_key),
    session_id: str = Depends(get_session_id),
):
    try:
        resp = (
            supabase.table("live_conversations")
            .select("id,status,created_at")
            .eq("client_id", client_id)
            .eq("session_id", session_id)
            .in_("status", ["pending", "active"])
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="No open conversation")
        conversation_id = rows[0]["id"]
        status = rows[0]["status"]

        token = mint_visitor_jwt(
            client_id=client_id,
            session_id=session_id,
            conversation_id=conversation_id,
            ttl_minutes=30
        )
        return {"conversation_id": conversation_id, "supabase_jwt": token, "status": status}
    except HTTPException:
        raise
    except Exception as e:
        print("live_join error:", e)
        raise HTTPException(status_code=500, detail="Unable to join conversation")

@app.get("/live/history/{conversation_id}")
async def live_history(
    conversation_id: str = Path(...),
    client_id: str = Depends(get_client_id_from_key),
    session_id: str = Depends(get_session_id),
):
    try:
        conv = (
            supabase.table("live_conversations")
            .select("id, client_id, session_id")
            .eq("id", conversation_id)
            .single()
            .execute()
        )
        if not conv.data or conv.data["client_id"] != client_id:
            raise HTTPException(status_code=404, detail="Conversation not found")

        live_msgs = (
            supabase.table("live_messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        ).data or []

        bot_logs = (
            supabase.table("chat_logs")
            .select("user_message, bot_response, timestamp")
            .eq("client_id", client_id)
            .eq("session_id", conv.data["session_id"])
            .order("timestamp", desc=False)
            .execute()
        ).data or []

        return {"live_messages": live_msgs, "bot_transcript": bot_logs}
    except HTTPException:
        raise
    except Exception as e:
        print("live_history error:", e)
        raise HTTPException(status_code=500, detail="Unable to fetch history")

# === CREDITS SNAPSHOT (lifetime) ===
@app.get("/credits")
async def get_credits(client_id: str = Depends(get_client_id_from_key)):
    """
    Returns the current one-time credits for this client.
    Does NOT mutate anything.
    """
    ensure_credits_row(client_id)

    # plan (optional, for UI)
    prof = (
        supabase.table("users_extra")
        .select("plan")
        .eq("id", client_id)
        .single()
        .execute()
    ).data or {}
    plan = prof.get("plan", "starter")

    # remaining balances
    cc = (
        supabase.table("client_credits")
        .select("website_pages_remaining, pdf_pages_remaining")
        .eq("client_id", client_id)
        .single()
        .execute()
    ).data or {"website_pages_remaining": 0, "pdf_pages_remaining": 0}

    return {
        "plan": plan,
        "website_pages_remaining": int(cc.get("website_pages_remaining", 0) or 0),
        "pdf_pages_remaining": int(cc.get("pdf_pages_remaining", 0) or 0),
    }

# === PDF PREVIEW (count pages only; no credits reservation) ===
@app.post("/ingest/pdf/preview")
async def ingest_pdf_preview(
    client_id: str = Depends(get_client_id_from_key),
    pdf: UploadFile = File(...),
):
    """
    Uploads file temporarily, counts pages, then deletes.
    Does NOT consume credits. Used by dashboard before actual ingest.
    """
    ensure_credits_row(client_id)

    # fetch remaining credits to inform the UI
    cc = (
        supabase.table("client_credits")
        .select("pdf_pages_remaining")
        .eq("client_id", client_id)
        .single()
        .execute()
    ).data or {"pdf_pages_remaining": 0}
    remaining = int(cc.get("pdf_pages_remaining", 0) or 0)

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"preview_{uuid.uuid4()}_{pdf.filename}")

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(pdf.file, f)

        try:
            reader = PdfReader(temp_path)
            page_count = len(reader.pages)
        except Exception:
            page_count = 0  # unreadable PDFs show 0 and allowed=False

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "filename": pdf.filename,
        "page_count": page_count,
        "pdf_pages_remaining": remaining,
        "allowed": page_count > 0 and page_count <= remaining,
    }
