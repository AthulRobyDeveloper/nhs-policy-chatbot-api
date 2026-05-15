import os
import fitz
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

POLICIES_DIR  = "policies"
CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200

# Full document registry with compliance metadata
DOCUMENT_REGISTRY = {
    "TRW.DI.POL.1502.1-Artificial-Intelligence-Policy.pdf": {
        "doc_title":        "Artificial Intelligence Policy",
        "policy_reference": "TRW.D&I.POL.1502.1.1",
        "version":          "1.1",
        "issue_date":       "April 2025",
        "review_date":      "April 2026",
        "department":       "Digital & Innovation",
        "pathway":          "Pathway 1",
        "pdf_file":         "ai_policy.pdf",
    },
    "TRW.IGT.POL.139.7-Information-Security-Policy.pdf": {
        "doc_title":        "Information Security Policy",
        "policy_reference": "TRW.IGT.POL.139.7",
        "version":          "7.0",
        "issue_date":       "October 2024",
        "review_date":      "October 2029",
        "department":       "Digital & Innovation",
        "pathway":          "Pathway 1",
        "pdf_file":         "info_security.pdf",
    },
    "TRW.IGT.POL.373.6.1-Information-Governance-Policy.pdf": {
        "doc_title":        "Information Governance Policy",
        "policy_reference": "TRW.IGT.POL.373.6.1",
        "version":          "6.1",
        "issue_date":       "January 2024",
        "review_date":      "January 2029",
        "department":       "Information Governance",
        "pathway":          "Pathway 1",
        "pdf_file":         "info_governance.pdf",
    },
    "TRW.HGV.POL.1467.2-Patient-Safety-Incident-Response.pdf": {
        "doc_title":        "Patient Safety Incident Response Policy",
        "policy_reference": "TRW.HGV.POL.1467.2",
        "version":          "V2",
        "issue_date":       "March 2025",
        "review_date":      "May 2029",
        "department":       "Healthcare Governance",
        "pathway":          "Pathway 2",
        "pdf_file":         "patient_safety.pdf",
    },
    "TRW.CRM.POL.103.5-Data-Quality-Policy.pdf": {
        "doc_title":        "Data Quality Policy",
        "policy_reference": "TRW.CRM.POL.103.5",
        "version":          "5",
        "issue_date":       "October 2024",
        "review_date":      "October 2027",
        "department":       "Clinical Records Management",
        "pathway":          "Pathway 1",
        "pdf_file":         "data_quality.pdf",
    },
}

def load_pdfs(folder: str) -> list[dict]:
    """
    Load all PDFs from the policies folder.
    Extracts text page by page to preserve
    page number metadata for source citation.
    """
    documents = []
    ingested_at = datetime.now().isoformat()

    for filename in os.listdir(folder):
        if not filename.endswith(".pdf"):
            continue

        path     = os.path.join(folder, filename)
        registry = DOCUMENT_REGISTRY.get(filename, {})

        print(f"\nLoading: {filename}")
        print(f"  Title: {registry.get('doc_title', 'Unknown')}")
        print(f"  Reference: {registry.get('policy_reference', 'Unknown')}")

        doc   = fitz.open(path)
        pages = []

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:  # skip blank pages
                pages.append({
                    "text":        text,
                    "page":        page_num,
                    "source":      filename,
                    "ingested_at": ingested_at,
                    **registry,   # adds all registry fields
                })

        documents.append({
            "filename": filename,
            "pages":    pages,
            "registry": registry,
        })

        print(f"  Pages extracted: {len(pages)}")
    return documents


def chunk_documents(documents: list[dict]) -> tuple[list, list]:
    """
    Split documents into chunks with overlap.
    Each chunk keeps full metadata for source citation
    and compliance audit trail.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks   = []
    all_metadata = []
    chunk_index  = 0

    for doc in documents:
        doc_chunk_count = 0

        for page_data in doc["pages"]:
            chunks = splitter.split_text(page_data["text"])

            for chunk in chunks:
                if not chunk.strip():
                    continue

                all_chunks.append(chunk)
                all_metadata.append({
                    # Source citation fields
                    "source":           page_data["source"],
                    "doc_title":        page_data.get("doc_title", ""),
                    "page":             page_data["page"],
                    "policy_reference": page_data.get("policy_reference", ""),
                    "version":          page_data.get("version", ""),
                    "issue_date":       page_data.get("issue_date", ""),
                    "review_date":      page_data.get("review_date", ""),
                    "department":       page_data.get("department", ""),
                    "pathway":          page_data.get("pathway", ""),
                    "pdf_file":         page_data.get("pdf_file", ""),

                    # Audit fields
                    "ingested_at":      page_data.get("ingested_at", ""),
                    "chunk_index":      chunk_index,
                })

                chunk_index     += 1
                doc_chunk_count += 1

        print(f"  {doc['filename']}: {doc_chunk_count} chunks")

    return all_chunks, all_metadata


def store_in_chromadb(chunks: list, metadata: list):
    """
    Embed chunks using local sentence-transformers model.
    Store in ChromaDB — runs entirely locally.
    No data sent externally — NHS compliant.
    """
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  Running locally — no data sent externally")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"Storing {len(chunks)} chunks in ChromaDB...")

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        metadatas=metadata,
        persist_directory=CHROMA_DIR
    )

    print(f"ChromaDB saved to: {CHROMA_DIR}/")
    return vectorstore


def print_summary(documents, chunks, metadata):
    """Print ingestion summary for audit log."""
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Timestamp:        {datetime.now().isoformat()}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks:     {len(chunks)}")
    print(f"Embedding model:  {EMBED_MODEL} (local)")
    print(f"Vector store:     {CHROMA_DIR}/ (local)")
    print(f"External calls:   None — NHS compliant")
    print("\nDocuments ingested:")
    for doc in documents:
        r = doc["registry"]
        print(f"  → {r.get('doc_title', doc['filename'])}")
        print(f"     Ref: {r.get('policy_reference', 'N/A')}")
        print(f"     Version: {r.get('version', 'N/A')}")
        print(f"     Review due: {r.get('review_date', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    print("=" * 60)
    print("UHP NHS POLICY CHATBOT — DOCUMENT INGESTION")
    print("Compliant with UHP AI Policy TRW.D&I.POL.1502.1.1")
    print("=" * 60)

    print("\nStep 1: Loading PDFs...")
    documents = load_pdfs(POLICIES_DIR)

    print("\nStep 2: Chunking with overlap...")
    chunks, metadata = chunk_documents(documents)

    print("\nStep 3: Embedding locally + storing in ChromaDB...")
    store_in_chromadb(chunks, metadata)

    print_summary(documents, chunks, metadata)
    print("\n✅ Ingestion complete — ready for querying")