import os
import shutil
import uuid
import datetime
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from fastapi import (
    Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, Request, Path
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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
app = FastAPI(title="SaaS Chatbot + Live Handoff")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_WIDGET_ORIGINS or [
        "https://rag-cloud-embedding-frontend.vercel.app",
        "https://chatbot-insight-opal.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Pydantic Models =================
class QueryJSON(BaseModel):
    question: str

class FeedbackJSON(BaseModel):
    botResponse: str
    userContact: str

class LiveRequestJSON(BaseModel):
    requested_by_contact: Optional[str] = None

# ================= HELPERS =================
def normalize_domain(url: str) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"https://{url}")
    hostname = (parsed.hostname or url).lower()
    return hostname[4:] if hostname.startswith("www.") else hostname

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

    if x_client_domain:
        client_domain = x_client_domain.lower().lstrip("www.")
    else:
        origin = request.headers.get("origin") or request.headers.get("referer")
        if not origin:
            raise HTTPException(status_code=403, detail="Missing Origin/Referer")
        parsed_origin = urlparse(origin)
        client_domain = (parsed_origin.hostname or origin).lower()
        if client_domain.startswith("www."):
            client_domain = client_domain[4:]

    if normalized_allowed and client_domain not in normalized_allowed:
        raise HTTPException(status_code=403, detail=f"Unauthorized origin: {client_domain}")

    return client_id

async def get_session_id(x_session_id: str = Header(None, alias="X-Session-Id")) -> str:
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
    return x_session_id

# ================= LLM / INGEST HELPERS =================
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

def chunk_documents(documents):
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

# ================= Visitor JWT (for Supabase Realtime) =================
def mint_visitor_jwt(*, client_id: str, session_id: str, conversation_id: str, ttl_minutes: int = 30) -> str:
    if not SUPABASE_JWT_SECRET:
        raise RuntimeError("Missing SUPABASE_JWT_SECRET")
    now = datetime.datetime.utcnow()
    payload = {
        "aud": "authenticated",
        "exp": now + datetime.timedelta(minutes=ttl_minutes),
        "iat": now,
        "role": "visitor",
        "client_id": client_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
    }
    return jwt.encode(payload, SUPABASE_JWT_SECRET, algorithm="HS256")

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
@app.post("/live/request")
async def live_request(
    payload: LiveRequestJSON,
    client_id: str = Depends(get_client_id_from_key),
    session_id: str = Depends(get_session_id),
):
    """
    Create (or reuse) a live conversation for this {client_id, session_id}.
    Returns {conversation_id, supabase_jwt, status}.
    """
    try:
        # Look for an existing open conversation
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
            # Pre-generate ID to avoid insert→select chaining
            conversation_id = str(uuid.uuid4())
            current_status = "pending"
            supabase.table("live_conversations").insert({
                "id": conversation_id,
                "client_id": client_id,
                "session_id": session_id,
                "status": current_status,
                "requested_by_contact": payload.requested_by_contact,
            }).execute()

            # optional: system seed
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
    """
    Mint a fresh visitor JWT to rejoin an open conversation (pending/active).
    """
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
    """
    Return:
      - live_messages for the conversation
      - prior bot/visitor Q&A (chat_logs) for the same session
    """
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
