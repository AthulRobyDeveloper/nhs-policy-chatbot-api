import os
import math
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from rag import query_policies, init_audit_db

load_dotenv()

# ─── APP SETUP ────────────────────────────────────────────────
app = FastAPI(
    title="UHP NHS Policy Chatbot API",
    description=(
        "RAG-powered policy assistant for University Hospitals "
        "Plymouth NHS Trust. Compliant with UHP AI Policy "
        "TRW.D&I.POL.1502.1.1 (Pathway 1)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ─── CORS ─────────────────────────────────────────────────────
# In production: restrict to NHS domain only
# e.g. "https://nhs.arp-infotech.com"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── RATE LIMITING ────────────────────────────────────────────
# Simple in-memory rate limiting
# In production: use Redis for distributed rate limiting
request_counts: dict[str, list] = {}
RATE_LIMIT     = 20    # requests per window
RATE_WINDOW    = 60    # seconds


def check_rate_limit(client_ip: str) -> bool:
    """
    Simple rate limiting per IP address.
    Prevents abuse and controls API costs.
    NHS deployment would use proper Redis-based limiting.
    """
    now      = time.time()
    window   = now - RATE_WINDOW

    if client_ip not in request_counts:
        request_counts[client_ip] = []

    # Remove old requests outside window
    request_counts[client_ip] = [
        t for t in request_counts[client_ip]
        if t > window
    ]

    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return False

    request_counts[client_ip].append(now)
    return True


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Policy question to ask",
        example="What is UHP's policy on using ChatGPT?"
    )


def normalise_confidence(raw_score: float) -> dict:
    """
    Convert raw cross-encoder logit to human-readable confidence.
    Sigmoid maps any real number to 0–1; multiply by 100 for %.
    """
    percentage = round(100 / (1 + math.exp(-raw_score)), 1)
    if percentage >= 70:
        label = "High"
    elif percentage >= 45:
        label = "Medium"
    else:
        label = "Low — please verify with original document"
    return {"percentage": percentage, "label": label}


class PolicyResponse(BaseModel):
    answer:              str
    source:              str
    reference:           str
    page:                int | str
    version:             str
    confidence_score:    float
    confidence_percent:  float
    confidence_label:    str
    pathway:             str
    pdf_url:             str
    all_pages:           list
    timestamp:           str
    disclaimer:          str


class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: str
    documents: int
    compliant: str


class ErrorResponse(BaseModel):
    error:     str
    timestamp: str
    support:   str


# ─── STARTUP ──────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialise audit database on startup."""
    init_audit_db()
    print("✅ UHP Policy Chatbot API started")
    print("✅ Audit database initialised")
    print("✅ Compliant with TRW.D&I.POL.1502.1.1")


# ─── HEALTH CHECK ─────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API status and compliance information"
)
async def health_check():
    """
    Health check endpoint.
    Returns system status and NHS compliance information.
    Used by monitoring systems and deployment pipelines.
    """
    from rag import _all_texts
    return {
        "status":    "healthy",
        "version":   "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "documents": len(_all_texts),
        "compliant": "UHP AI Policy TRW.D&I.POL.1502.1.1 Pathway 1"
    }


# ─── MAIN ASK ENDPOINT ────────────────────────────────────────
@app.post(
    "/ask",
    response_model=PolicyResponse,
    summary="Ask a policy question",
    description=(
        "Submit a question about UHP policies. "
        "Returns grounded answer with source citation. "
        "Do NOT include patient or staff personal data."
    )
)
async def ask_question(
    request: Request,
    body:    QuestionRequest
):
    """
    Main RAG endpoint.

    NHS Compliance Notes:
    - No patient identifiable data should be submitted
    - All queries are logged (hash only — GDPR compliant)
    - Responses grounded in UHP policy documents only
    - Human-in-the-loop: all answers require human verification
    - Compliant with UHP AI Policy Pathway 1
    """

    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many requests. Please wait before "
                "submitting another question."
            )
        )

    # Process question
    try:
        result = query_policies(body.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    # Handle errors from pipeline
    if result.get("error"):
        raise HTTPException(
            status_code=400,
            detail=result["answer"]
        )

    confidence_data = normalise_confidence(result.get("confidence", 0.0))
    pdf_file = result.get("pdf_file", "")
    page     = result.get("page", "")
    pdf_url  = f"/policies/{pdf_file}#page={page}" if pdf_file and page else ""

    return {
        "answer":             result["answer"],
        "source":             result.get("source", ""),
        "reference":          result.get("reference", ""),
        "page":               page,
        "version":            result.get("version", ""),
        "confidence_score":   result.get("confidence", 0.0),
        "confidence_percent": confidence_data["percentage"],
        "confidence_label":   confidence_data["label"],
        "pathway":            result.get("pathway", ""),
        "pdf_url":            pdf_url,
        "all_pages":          result.get("all_pages", []),
        "timestamp":          datetime.now().isoformat(),
        "disclaimer": (
            "This guidance is based on UHP policy documents. "
            "Always verify critical decisions with the original "
            "policy or your line manager. Human judgement must "
            "prevail over AI guidance."
        )
    }


# ─── POLICIES LIST ENDPOINT ───────────────────────────────────
@app.get(
    "/policies",
    summary="List available policies",
    description="Returns list of policy documents loaded in the system"
)
async def list_policies():
    """
    Returns metadata for all loaded policy documents.
    Includes version, review dates for transparency.
    """
    from rag import _all_metas

    # Get unique documents
    seen  = set()
    docs  = []

    for meta in _all_metas:
        title = meta.get("doc_title", "")
        if title and title not in seen:
            seen.add(title)
            docs.append({
                "title":     title,
                "reference": meta.get("policy_reference", ""),
                "version":   meta.get("version", ""),
                "issue_date":   meta.get("issue_date", ""),
                "review_date":  meta.get("review_date", ""),
                "department":   meta.get("department", ""),
                "pathway":      meta.get("pathway", ""),
            })

    return {
        "total_documents": len(docs),
        "documents":       docs,
        "last_updated":    datetime.now().isoformat(),
        "note": (
            "All documents sourced from UHP public policy library. "
            "Review dates indicate when policies require updating."
        )
    }


# ─── ERROR HANDLERS ───────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error":     "Endpoint not found",
            "timestamp": datetime.now().isoformat(),
            "support":   "Contact D&I Digital team"
        }
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error":     "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "support":   "Contact D&I Digital team"
        }
    )


# ─── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )