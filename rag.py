import os
import re
import hashlib
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

load_dotenv()

# ─── CONFIGURATION ────────────────────────────────────────────
CHROMA_DIR     = "chroma_db"
EMBED_MODEL    = "all-MiniLM-L6-v2"
TOP_K_RETRIEVE = 30
TOP_K_RERANK   = 12
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
AUDIT_DB       = "audit_log.db"

# ─── NHS SYSTEM PROMPT ────────────────────────────────────────
NHS_SYSTEM_PROMPT = """
You are a UHP Policy Assistant for University Hospitals
Plymouth NHS Trust (UHP).

STRICT RULES — NEVER VIOLATE:
1. ONLY answer using the provided policy context below
2. NEVER use general knowledge or make assumptions
3. NEVER provide clinical advice or diagnosis
4. NEVER process or repeat any patient information
5. If the answer is not clearly in the context, respond ONLY with:
   "I cannot find this information in the current UHP
   policy library. Please consult the original policy
   document or your line manager."
6. ALWAYS cite the source document and page number
7. ALWAYS add this disclaimer at the end of every answer:
   "⚠️ This guidance is based on UHP policy documents.
   For critical decisions always verify with the original
   policy or your line manager. Human judgement must
   prevail over AI guidance."
8. If the question asks for ALL items, ALL methods, a
   COMPLETE LIST, or a FULL SUMMARY, always end with:
   "📄 Note: This answer is based on retrieved sections
   of the policy. For complete information please view
   the full document using the citation link below."

FORMAT YOUR RESPONSE AS:
Answer: [your answer here]

Source: [document title]
Reference: [policy reference number]
Page: [page number]
Version: [version number]
⚠️ Disclaimer: This guidance is based on UHP policy documents.
Always verify critical decisions with the original policy
or your line manager.
"""

# ─── LOAD MODELS ONCE AT STARTUP ──────────────────────────────
print("Loading embedding model (local — no external calls)...")
_embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Loading vectorstore...")
_vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=_embeddings
)

print("Loading reranker...")
_reranker = CrossEncoder(RERANK_MODEL)

# Pre-load all documents for BM25
print("Preparing BM25 index...")
_all_data  = _vectorstore.get()
_all_texts = _all_data["documents"]
_all_metas = _all_data["metadatas"]
_bm25      = BM25Okapi([t.lower().split() for t in _all_texts])

print(f"✅ Ready — {len(_all_texts)} chunks indexed\n")


# ─── AUDIT LOGGER ─────────────────────────────────────────────
def init_audit_db():
    """
    Initialise SQLite audit database.
    GDPR compliant — stores query HASH not raw text.
    """
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            query_hash      TEXT NOT NULL,
            source_doc      TEXT,
            page_retrieved  INTEGER,
            policy_ref      TEXT,
            confidence      REAL,
            response_length INTEGER,
            pathway         TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_query(query: str, source_doc: str, page: int,
              policy_ref: str, confidence: float,
              response_length: int, pathway: str):
    """
    Log query audit record.
    GDPR NOTE: Raw query text is NOT stored.
    Only a SHA256 hash is kept for audit trail.
    """
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute("""
        INSERT INTO audit_log
        (timestamp, query_hash, source_doc, page_retrieved,
         policy_ref, confidence, response_length, pathway)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        query_hash,
        source_doc,
        page,
        policy_ref,
        confidence,
        response_length,
        pathway
    ))
    conn.commit()
    conn.close()


# ─── INPUT SANITISATION ───────────────────────────────────────
def sanitise_input(query: str) -> str:
    """
    Sanitise user input — prevent prompt injection.
    Cyber security requirement for NHS deployment.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if len(query) > 500:
        raise ValueError(
            "Query too long. Please ask a specific "
            "question under 500 characters."
        )

    injection_patterns = [
        r"ignore (previous|above|all) instructions",
        r"forget (everything|your instructions)",
        r"you are now",
        r"new instructions",
        r"system prompt",
        r"jailbreak",
        r"act as",
        r"pretend (you are|to be)",
    ]
    query_lower = query.lower()
    for pattern in injection_patterns:
        if re.search(pattern, query_lower):
            raise ValueError(
                "Your query contains invalid content. "
                "Please ask a genuine policy question."
            )

    return query.strip()


# ─── REVIEW DATE CHECKER ──────────────────────────────────────
def check_review_status(metadata: dict) -> str:
    """
    Warn if retrieved policy is past review date.
    UHP AI Policy is due April 2026 — THIS MONTH.
    Critical for NHS staff to know about outdated policies.
    """
    review_date = metadata.get("review_date", "")
    doc_title   = metadata.get("doc_title", "this policy")

    # Policies currently overdue or due this month
    overdue = ["April 2026"]

    if review_date in overdue:
        return (
            f"\n\n⚠️ POLICY REVIEW ALERT: '{doc_title}' "
            f"was due for review in {review_date}. "
            f"Please verify you are reading the most "
            f"current version before acting on this guidance."
        )
    return ""


# ─── PATHWAY CLASSIFIER ───────────────────────────────────────
def classify_pathway(query: str, metadata: dict) -> str:
    """
    Detect if query relates to clinical AI (Pathway 2).
    Per UHP AI Policy TRW.D&I.POL.1502.1.1 Section 2.
    Pathway 2 requires DCB0129/DCB0160 governance.
    """
    clinical_keywords = [
        "patient", "clinical", "diagnosis", "treatment",
        "medical", "imaging", "prescri", "ward",
        "doctor", "nurse", "surgeon", "theatre",
        "pathway 2", "dcb0129", "dcb0160"
    ]
    query_lower = query.lower()
    is_clinical = any(kw in query_lower for kw in clinical_keywords)
    doc_pathway = metadata.get("pathway", "Pathway 1")

    if is_clinical or doc_pathway == "Pathway 2":
        return (
            "\n\n⚠️ PATHWAY 2 NOTICE: This query relates to "
            "clinical AI applications. Per UHP AI Policy, "
            "Pathway 2 governance applies — requiring "
            "DCB0129/DCB0160 clinical safety documentation "
            "and Clinical Design Authority (CDA) approval."
        )
    return ""


# ─── HYBRID SEARCH ────────────────────────────────────────────
def hybrid_search(original: str, rewritten: str = None) -> list:
    """
    BM25 uses the original query to preserve exact NHS terms
    (DCB0129, TDA, ChatGPT). Semantic uses the rewritten query
    for expanded NHS terminology. Separate queries per signal
    gives the best of both.
    """
    bm25_scores = _bm25.get_scores(original.lower().split())
    max_bm25    = max(bm25_scores) if max(bm25_scores) > 0 else 1

    semantic_query   = rewritten if rewritten else original
    semantic_results = _vectorstore.similarity_search_with_score(
        semantic_query, k=TOP_K_RETRIEVE
    )
    semantic_lookup = {
        doc.page_content: float(1 - score)
        for doc, score in semantic_results
    }

    combined = []
    for i, (text, meta) in enumerate(zip(_all_texts, _all_metas)):
        bm25_norm      = float(bm25_scores[i]) / max_bm25
        semantic_score = semantic_lookup.get(text, 0.0)
        combined_score = (0.2 * bm25_norm) + (0.8 * semantic_score)
        combined.append((combined_score, Document(
            page_content=text,
            metadata=meta
        )))

    combined.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in combined[:TOP_K_RETRIEVE]]


# ─── RERANKER ─────────────────────────────────────────────────
def rerank(question: str, docs: list) -> tuple[list, float]:
    """
    Cross-encoder reranker with expanded window (top 12).
    Scores all hybrid candidates; returns top 12 so that
    domain-specific chunks aren't dropped by the general-
    purpose reranker before the LLM sees them.
    """
    pairs  = [(question, doc.page_content) for doc in docs]
    scores = _reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    top_docs  = [doc for _, doc in scored[:TOP_K_RERANK]]
    top_score = float(scored[0][0]) if scored else 0.0
    return top_docs, top_score


# ─── LONG CONTEXT REORDER ─────────────────────────────────────
def reorder_for_llm(docs: list) -> list:
    """
    Reorder chunks to fix 'Lost in the Middle' problem.
    LLMs best retain content at the start and end of context;
    placing the most relevant chunks there improves answer quality.
    """
    from langchain_community.document_transformers import LongContextReorder
    return LongContextReorder().transform_documents(docs)


# ─── FORMAT CONTEXT ───────────────────────────────────────────
def format_context(docs: list) -> tuple[str, dict]:
    """
    Format retrieved chunks for LLM context.
    Includes full NHS compliance metadata
    for source citation and audit trail.
    """
    parts        = []
    primary_meta = {}
    all_pages    = []

    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        if i == 1:
            primary_meta = meta

        page = meta.get("page")
        if page:
            all_pages.append(page)

        parts.append(
            f"[Source {i}]\n"
            f"Document: {meta.get('doc_title', 'Unknown')}\n"
            f"Reference: {meta.get('policy_reference', 'N/A')}\n"
            f"Version: {meta.get('version', 'N/A')}\n"
            f"Page: {meta.get('page', 'N/A')}\n"
            f"Review Due: {meta.get('review_date', 'N/A')}\n"
            f"Content:\n{doc.page_content}\n"
        )

    return "\n---\n".join(parts), primary_meta, all_pages


# ─── OUTPUT VALIDATOR ─────────────────────────────────────────
def validate_output(response: str) -> str:
    """
    Validate LLM response before returning to user.
    Ensures disclaimer present.
    Checks no patient identifiers in response.
    """
    # Ensure disclaimer present
    if "verify" not in response.lower():
        response += (
            "\n\n⚠️ Always verify critical decisions "
            "with the original policy or your line manager."
        )

    # Check for patient identifiers
    patient_patterns = [
        r"\b\d{10}\b",
        r"\bNHS\s*number\b",
        r"\bpatient\s+name\b",
        r"\bdate\s+of\s+birth\b",
    ]
    for pattern in patient_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return (
                "This request cannot be processed as it may "
                "contain patient identifiable information. "
                "Please rephrase without patient details."
            )

    return response


# ─── PAGE EXTRACTOR ───────────────────────────────────────────
def extract_page_from_answer(answer_text: str, fallback) -> any:
    """
    Extract page number cited by LLM in its response.
    More accurate than primary_meta page, which reflects
    the top reranked chunk rather than the cited source.
    """
    match = re.search(r'Page:\s*(\d+)', answer_text)
    if match:
        return int(match.group(1))
    return fallback


# ─── QUERY REWRITER ───────────────────────────────────────────
def rewrite_query(question: str) -> str:
    """
    Rewrite user query into formal NHS policy terminology.

    Staff use informal language; policy documents use formal NHS
    terms. This gap causes retrieval failures even when the answer
    exists (e.g. staff ask about "ChatGPT", documents say "Copilot"
    and "generative AI"). LLM bridges this automatically — no
    hardcoded synonyms, adapts to any query.
    """
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=100,
        temperature=0
    )
    prompt = (
        "Rewrite this question using formal NHS policy document "
        "terminology to improve document search.\n"
        "Keep the same meaning but use terms likely found in NHS "
        "policy documents.\n"
        "Return ONLY the rewritten query, nothing else.\n\n"
        f"Question: {question}\n"
        "Rewritten:"
    )
    response  = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()
    print(f"  Query rewritten: {rewritten}")
    return rewritten


# ─── MAIN QUERY FUNCTION ──────────────────────────────────────
def query_policies(question: str) -> dict:
    """
    Main RAG pipeline:

    1.  Sanitise input (prompt injection prevention)
    2.  Hybrid search — BM25 + semantic top 20
    3.  Rerank — cross-encoder top 5
    4.  Format context with NHS metadata
    5.  Check review dates
    6.  Classify pathway (1 or 2)
    7.  Generate grounded response (temperature=0)
    8.  Validate output
    9.  Audit log (GDPR compliant — hash only)
    10. Return response + full metadata
    """

    # Step 1: Sanitise
    try:
        question = sanitise_input(question)
    except ValueError as e:
        return {"answer": str(e), "source": "N/A", "error": True}

    # Step 1b: Rewrite query into NHS policy terminology
    rewritten = rewrite_query(question)

    # Step 2: Hybrid search — BM25 on original, semantic on rewritten
    docs = hybrid_search(original=question, rewritten=rewritten)

    if not docs:
        return {
            "answer": (
                "I cannot find relevant information in the "
                "current UHP policy library. Please consult "
                "the original policy documents or your "
                "line manager."
            ),
            "source": "N/A",
            "error":  False
        }

    # Step 3: Rerank
    top_docs, confidence = rerank(question, docs)

    # Step 3b: Reorder for LLM context window
    top_docs = reorder_for_llm(top_docs)

    # Step 4: Format context
    context, primary_meta, all_pages = format_context(top_docs)

    # Step 5: Review date check
    review_warning = check_review_status(primary_meta)

    # Step 6: Pathway classification
    pathway_warning = classify_pathway(question, primary_meta)

    # Step 7: Generate response
    llm = ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000,
        temperature=0  # deterministic — critical for NHS
    )

    full_prompt = (
        f"POLICY CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}"
    )

    messages = [
        SystemMessage(content=NHS_SYSTEM_PROMPT),
        HumanMessage(content=full_prompt)
    ]

    response    = llm.invoke(messages)
    answer_text = response.content

    # Step 8: Add NHS-specific warnings
    if review_warning:
        answer_text += review_warning
    if pathway_warning:
        answer_text += pathway_warning

    # Step 9: Validate output
    answer_text = validate_output(answer_text)

    # Step 10: Audit log
    log_query(
        query           = question,
        source_doc      = primary_meta.get("doc_title", ""),
        page            = primary_meta.get("page", 0),
        policy_ref      = primary_meta.get("policy_reference", ""),
        confidence      = confidence,
        response_length = len(answer_text),
        pathway         = primary_meta.get("pathway", "Pathway 1")
    )

    return {
        "answer":     answer_text,
        "source":     primary_meta.get("doc_title", ""),
        "reference":  primary_meta.get("policy_reference", ""),
        "page":       extract_page_from_answer(answer_text, all_pages[0] if all_pages else primary_meta.get("page", "")),
        "version":    primary_meta.get("version", ""),
        "confidence": round(confidence, 3),
        "pathway":    primary_meta.get("pathway", ""),
        "pdf_file":   primary_meta.get("pdf_file", ""),
        "error":      False
    }


# ─── TEST ─────────────────────────────────────────────────────
if __name__ == "__main__":
    init_audit_db()

    # DIAGNOSTIC — ChatGPT retrieval
    print("\nDIAGNOSTIC: Top 5 chunks for ChatGPT question")
    print("-" * 60)
    diag_q = "What is UHP's policy on using ChatGPT?"
    diag_docs = hybrid_search(diag_q)
    for i, doc in enumerate(diag_docs[:5], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source:  {doc.metadata.get('doc_title')}")
        print(f"Page:    {doc.metadata.get('page')}")
        print(f"Preview: {doc.page_content[:200]}")
    print("\n" + "=" * 60)

    test_questions = [
        "What is UHP's policy on using ChatGPT?",
        "Who must approve AI systems at UHP?",
        "What is the process for reporting a patient safety incident?",
        "What does DCB0129 require?",
        "What are the data quality standards at UHP?",
    ]

    # DIAGNOSTIC — direct ChromaDB search for ChatGPT content
    print("\nDIAGNOSTIC: Direct search for ChatGPT content")
    print("-" * 60)
    direct_results = _vectorstore.similarity_search(
        "ChatGPT Copilot generative AI staff guidance", k=5
    )
    for i, doc in enumerate(direct_results, 1):
        print(f"\n{i}. {doc.metadata.get('doc_title')} p{doc.metadata.get('page')}")
        print(f"   {doc.page_content[:150]}")

    print("\n" + "=" * 60)
    print("UHP NHS POLICY CHATBOT — RAG PIPELINE TEST")
    print("=" * 60)

    for question in test_questions:
        print(f"\nQ: {question}")
        print("-" * 40)
        result = query_policies(question)
        print(f"A: {result['answer'][:400]}...")
        print(f"\nSource:     {result['source']}")
        print(f"Reference:  {result['reference']}")
        print(f"Page:       {result['page']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Pathway:    {result['pathway']}")