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




#new imports for voice

import openai
from elevenlabs.client import ElevenLabs
from fastapi.responses import StreamingResponse

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


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")

openai.api_key = OPENAI_API_KEY
eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)

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
        # add localhost here if needed during local dev:
        # "http://localhost:5173",
        # "http://localhost:3000",
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
    try:
        res = (
            supabase.table("client_credits")
            .select("client_id")
            .eq("client_id", client_id)
            .execute()
        )
        exists = bool(res.data and len(res.data) > 0)
        if not exists:
            print(f"[credits] creating credits row for {client_id}")
            supabase.table("client_credits").upsert({
                "client_id": client_id,
                "website_pages_remaining": 0,
                "pdf_pages_remaining": 0,
            }, on_conflict="client_id").execute()
    except Exception as e:
        print(f"[credits] ensure_credits_row select failed, trying upsert anyway: {e}")
        supabase.table("client_credits").upsert({
            "client_id": client_id,
            "website_pages_remaining": 0,
            "pdf_pages_remaining": 0,
        }, on_conflict="client_id").execute()

def reserve_website_credits(client_id: str, pages: int) -> bool:
    print(f"[credits] reserving website pages: client={client_id} pages={pages}")
    res = supabase.rpc("reserve_website_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()
    print(f"[credits] reserve_website_pages RPC -> {res.data}")
    return bool(res.data)

def refund_website_credits(client_id: str, pages: int) -> None:
    if pages > 0:
        print(f"[credits] refunding website pages: client={client_id} pages={pages}")
        supabase.rpc("refund_website_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()

def reserve_pdf_credits(client_id: str, pages: int) -> bool:
    print(f"[credits] reserving pdf pages: client={client_id} pages={pages}")
    res = supabase.rpc("reserve_pdf_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()
    print(f"[credits] reserve_pdf_pages RPC -> {res.data}")
    return bool(res.data)

def refund_pdf_credits(client_id: str, pages: int) -> None:
    if pages > 0:
        print(f"[credits] refunding pdf pages: client={client_id} pages={pages}")
        supabase.rpc("refund_pdf_pages", {"p_client_id": client_id, "p_pages": int(pages)}).execute()

# ================= DISCOVERY (sitemap → Firecrawl map) =================
SITEMAP_RE = re.compile(r"<loc>(.*?)</loc>", re.IGNORECASE)

def _http_get(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "InsightBot/1.0"})
        if r.ok:
            return r.text
    except Exception as e:
        print(f"[http_get] error fetching {url}: {e}")
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
    endpoint = "https://api.firecrawl.dev/v1/map"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
    payload = {"url": url}
    try:
        print(f"[firecrawl_map] POST {endpoint} payload={payload}")
        r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = r.text
            print(f"[firecrawl_map] error {r.status_code} {body}")
            return []
        data = r.json()
        urls = data.get("urls") or data.get("data") or []
        host = urlparse(url if "://" in url else f"https://{url}").netloc
        urls = [u for u in urls if urlparse(u).netloc == host]
        print(f"[firecrawl_map] mapped={len(urls)}")
        return urls
    except Exception as e:
        print(f"[firecrawl_map] exception: {e}")
        return []

def canonicalize_urls(urls: list) -> list:
    out = set()
    for u in urls:
        p = urlparse(u if "://" in u else f"https://{u}")
        path = p.path
        out.add(f"{p.scheme}://{p.netloc}{path}")
    return sorted(out)

# ======== Fallback helpers (map + scrape with logging) =========
def _path_glob_predicate(include_paths: Optional[List[str]], exclude_paths: Optional[List[str]]):
    inc = include_paths or []
    exc = exclude_paths or []

    def normalize_glob(g: str) -> str:
        g = g.strip()
        if g.endswith("*"):
            g = g[:-1]
        return g

    inc_norm = [normalize_glob(g) for g in inc]
    exc_norm = [normalize_glob(g) for g in exc]

    def allowed(url: str) -> bool:
        p = urlparse(url)
        path = p.path or "/"
        if inc_norm:
            ok = any(path.startswith(prefix) for prefix in inc_norm)
            if not ok:
                return False
        if exc_norm and any(path.startswith(prefix) for prefix in exc_norm):
            return False
        return True

    return allowed

def firecrawl_scrape(url: str) -> Optional[Dict]:
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
    # formats: only markdown (no links/metadata requested)
    payload = {"url": url, "formats": ["markdown"]}
    try:
        print(f"[firecrawl_scrape] POST {endpoint} url={url}")
        r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = r.text
            print(f"[firecrawl_scrape] error {r.status_code} {body}")
            return None
        data = r.json() or {}
        return {
            "url": url,
            "markdown": data.get("markdown") or "",
            "metadata": data.get("metadata") or {}  # tolerate if the API returns it anyway
        }
    except Exception as e:
        print(f"[firecrawl_scrape] exception url={url} err={e}")
        return None

def firecrawl_crawl_fallback(start_url: str, *, limit: int,
                             include_paths: Optional[List[str]] = None,
                             exclude_paths: Optional[List[str]] = None) -> List[Dict]:
    start_url = _normalize_url_with_scheme(start_url)
    host = urlparse(start_url).netloc
    print(f"[crawl_fallback] start={start_url} limit={limit} include={include_paths} exclude={exclude_paths}")

    mapped = firecrawl_map(start_url)
    if not mapped:
        mapped = [start_url]

    mapped = [u for u in mapped if urlparse(u).netloc == host]
    pred = _path_glob_predicate(include_paths, exclude_paths)
    filtered = [u for u in mapped if pred(u)]
    take = filtered[: max(1, int(limit))]
    print(f"[crawl_fallback] mapped={len(mapped)} filtered={len(filtered)} taking={len(take)}")

    pages = []
    for u in take:
        item = firecrawl_scrape(u)
        if item and (item.get("markdown") or "").strip():
            pages.append(item)
    print(f"[crawl_fallback] scraped_ok={len(pages)}")
    return pages

# ================= FIRECRAWL CRAWL (REST) with logs + fallback =================
def firecrawl_crawl(url: str, *, limit: int, include_paths=None, exclude_paths=None) -> list:
    endpoint = "https://api.firecrawl.dev/v1/crawl"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}", "Content-Type": "application/json"}

    url = _normalize_url_with_scheme(url)
    safe_limit = max(1, int(limit))
    if safe_limit > 2000:
        safe_limit = 2000

    # formats: only markdown
    payload = {
        "url": url,
        "crawlEntireDomain": True,
        "maxDiscoveryDepth": 4,
        "limit": safe_limit,
        "scrapeOptions": {"formats": ["markdown"]},
    }
    if include_paths:
        payload["includePaths"] = include_paths
    if exclude_paths:
        payload["excludePaths"] = exclude_paths

    try:
        print(f"[firecrawl_crawl] start POST {endpoint} payload={payload}")
        r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        if not r.ok:
            try:
                body = r.json()
            except Exception:
                body = r.text
            print(f"[firecrawl_crawl] start failed {r.status_code} {body} -> fallback")
            return firecrawl_crawl_fallback(url, limit=safe_limit,
                                            include_paths=include_paths, exclude_paths=exclude_paths)

        data = r.json()
        job_id = data.get("jobId") or data.get("id")
        if not job_id:
            print(f"[firecrawl_crawl] missing job id: {data} -> fallback")
            return firecrawl_crawl_fallback(url, limit=safe_limit,
                                            include_paths=include_paths, exclude_paths=exclude_paths)

        status_ep = f"{endpoint}/{job_id}"
        print(f"[firecrawl_crawl] job_id={job_id} polling {status_ep}")

        import time
        for i in range(120):  # ~4 minutes
            s = requests.get(status_ep, headers=headers, timeout=30)
            if not s.ok:
                try:
                    body = s.json()
                except Exception:
                    body = s.text
                print(f"[firecrawl_crawl] status error {s.status_code} {body} -> fallback")
                return firecrawl_crawl_fallback(url, limit=safe_limit,
                                                include_paths=include_paths, exclude_paths=exclude_paths)

            body = s.json()
            state = body.get("status") or body.get("state")
            print(f"[firecrawl_crawl] poll#{i} state={state}")
            if state in ("completed", "finished", "succeeded"):
                pages = body.get("data") or body.get("pages") or []
                print(f"[firecrawl_crawl] completed pages={len(pages)}")
                return pages
            if state in ("failed", "error"):
                print(f"[firecrawl_crawl] job failed: {body} -> fallback")
                return firecrawl_crawl_fallback(url, limit=safe_limit,
                                                include_paths=include_paths, exclude_paths=exclude_paths)

            time.sleep(2)

        print("[firecrawl_crawl] timeout -> fallback")
        return firecrawl_crawl_fallback(url, limit=safe_limit,
                                        include_paths=include_paths, exclude_paths=exclude_paths)

    except Exception as e:
        print(f"[firecrawl_crawl] exception {e} -> fallback")
        return firecrawl_crawl_fallback(url, limit=safe_limit,
                                        include_paths=include_paths, exclude_paths=exclude_paths)

# ================= CHUNKING =================
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# ================= PINECONE UPSERT =================
def store_in_pinecone(chunks, client_id: str):
    print(f"[pinecone] preparing embeddings: n={len(chunks)} namespace={client_id}")
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
                    "title": str(chunk.metadata.get("title") or ""),
                },
            )
        )
    print(f"[pinecone] upserting to index={INDEX_NAME} namespace={client_id} count={len(vectors)}")
    index.upsert(vectors, namespace=client_id)
    return len(vectors)

# ================= PREVIEW (uses remaining website credits) =================
@app.get("/ingest/preview", response_model=PreviewResponse)
async def ingest_preview(url: str, client_id: str = Depends(get_client_id_from_key)):
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
        "allowed_pages_for_plan": remaining,
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
    reserved = max(1, intended)
    ok = reserve_website_credits(client_id, reserved)
    if not ok:
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
            # title may come from metadata or not; tolerate both
            title = ((p.get("metadata") or {}).get("title")) or p.get("title") or ""
            docs.append(Document(page_content=md, metadata={"source": src, "title": title}))
            actual += 1

        if not docs:
            refund_website_credits(client_id, reserved)
            return JSONResponse({"error": "No content extracted"}, status_code=400)

        # chunk + INDEX to Pinecone
        chunks = chunk_documents(docs)
        chunks_count = len(chunks)
        try:
            print(f"[pinecone] upserting url chunks: client={client_id} namespace={client_id} n_chunks={chunks_count}")
            vectors_count = store_in_pinecone(chunks, client_id)
            print(f"[pinecone] upsert done: vectors={vectors_count}")
        except Exception as e:
            print(f"[pinecone] upsert failed: {e}")
            # refund the full reservation since indexing failed
            refund_website_credits(client_id, reserved)
            raise HTTPException(status_code=500, detail="Indexing to Pinecone failed")

        if actual < reserved:
            refund_website_credits(client_id, reserved - actual)

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
        refund_website_credits(client_id, reserved)
        raise
    except Exception as e:
        refund_website_credits(client_id, reserved)
        print(f"[ingest_url] exception: {e}")
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

    temp_dir = "./temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{pdf.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(pdf.file, f)

    try:
        try:
            reader = PdfReader(temp_path)
            page_count = len(reader.pages)
        except Exception:
            raise HTTPException(400, detail="Unable to read PDF pages. Please upload a valid PDF.")

        if page_count <= 0:
            os.remove(temp_path)
            raise HTTPException(400, detail="PDF has 0 pages.")

        if page_count > remaining_pdf:
            os.remove(temp_path)
            raise HTTPException(403, detail=f"Not enough PDF credits. Needed {page_count}, have {remaining_pdf}.")

        ok = reserve_pdf_credits(client_id, page_count)
        if not ok:
            fresh = get_credits_remaining(client_id)["pdf"]
            os.remove(temp_path)
            raise HTTPException(403, detail=f"Insufficient PDF credits. Remaining: {fresh}")

        try:
            parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
            parsed = parser.load_data(temp_path)
            docs = [Document(page_content=d.text, metadata={"source": pdf.filename or ""}) for d in parsed]
        except Exception as e:
            refund_pdf_credits(client_id, page_count)
            print(f"[ingest_pdf] llama_parse exception: {e}")
            raise
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not docs:
            refund_pdf_credits(client_id, page_count)
            return JSONResponse({"error": "No content extracted from PDF"}, status_code=400)

        # chunk + INDEX to Pinecone
        chunks = chunk_documents(docs)
        chunks_count = len(chunks)
        try:
            print(f"[pinecone] upserting pdf chunks: client={client_id} namespace={client_id} n_chunks={chunks_count}")
            vectors_count = store_in_pinecone(chunks, client_id)
            print(f"[pinecone] upsert done: vectors={vectors_count}")
        except Exception as e:
            print(f"[pinecone] upsert failed: {e}")
            refund_pdf_credits(client_id, page_count)
            raise HTTPException(status_code=500, detail="Indexing to Pinecone failed")

        supabase.table("ingestion_events").insert({
            "client_id": client_id,
            "event_type": "pdf",
            "source": pdf.filename,
            "chunks_stored": int(chunks_count),
            "pdf_pages": int(page_count)
        }).execute()

        return {"message": "PDF ingested", "chunks_count": chunks_count, "pdf_pages": page_count}

    except:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# ================== CHATBOT + LIVE HANDOFF (unchanged) ==================
def crawl_website(url: str):
    app_fc = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
    crawl_status = app_fc.crawl_url(
        url,
        limit=10,
        scrape_options=ScrapeOptions(formats=["markdown"]),  # legacy path ok, only markdown
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

def store_in_pinecone_legacy(chunks, client_id: str):
    # kept for legacy /ingest/
    return store_in_pinecone(chunks, client_id)

def chatbot_query(client_id: str, question: str):
    print(f"[query] namespace={client_id} q={question!r}")
    query_embedding = embeddings_model.embed_query(question)
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True, namespace=client_id)
    print(f"[query] pinecone matches={len(results.matches) if hasattr(results, 'matches') else 0}")
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
    vectors_count = store_in_pinecone_legacy(chunks, client_id)
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
    ensure_credits_row(client_id)

    prof = (
        supabase.table("users_extra")
        .select("plan")
        .eq("id", client_id)
        .single()
        .execute()
    ).data or {}
    plan = prof.get("plan", "starter")

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
    ensure_credits_row(client_id)

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
            page_count = 0

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "filename": pdf.filename,
        "page_count": page_count,
        "pdf_pages_remaining": remaining,
        "allowed": page_count > 0 and page_count <= remaining,
    }


# ================= SPEECH TO TEXT (Whisper) =================
@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    try:
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with open(temp_path, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        os.remove(temp_path)
        return {"text": transcript.text}
    except Exception as e:
        print("STT error:", e)
        raise HTTPException(status_code=500, detail="Speech-to-text failed")


# ================= TEXT TO SPEECH (ElevenLabs) =================
@app.get("/tts")
async def text_to_speech(text: str):
    try:
        audio = eleven_client.generate(
            voice=ELEVEN_VOICE_ID,
            model="eleven_multilingual_v2",
            text=text
        )
        return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    except Exception as e:
        print("TTS error:", e)
        raise HTTPException(status_code=500, detail="Text-to-speech failed")