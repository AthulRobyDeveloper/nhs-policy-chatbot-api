"""
NHS Policy Chatbot — RAGAS Evaluation Pipeline

25 questions with ground truths taken directly
from UHP policy documents. Legitimate evaluation.

Targets (2026 production standards):
- Faithfulness:      >= 0.90
- Answer Relevancy:  >= 0.85
- Context Precision: >= 0.80
- Context Recall:    >= 0.80
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms        import LangchainLLMWrapper
from ragas.embeddings  import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset

from rag import (
    query_policies,
    hybrid_search,
    rewrite_query,
    rerank,
    init_audit_db,
    _embeddings as _hf_embeddings,
)

load_dotenv()

# ─── TARGETS ──────────────────────────────────────────────────
TARGETS = {
    "faithfulness":      0.90,
    "answer_relevancy":  0.85,
    "context_precision": 0.80,
    "context_recall":    0.80,
}

# ─── GOLDEN TEST SET ──────────────────────────────────────────
# Ground truths taken directly from UHP policy PDFs
GOLDEN_TEST_SET = [

    # ── DATA QUALITY POLICY (TRW.CRM.POL.103.5) ───────────────
    {
        "question": (
            "Why is good quality data important for "
            "patient care and Trust operations?"
        ),
        "ground_truth": (
            "Good quality information supports the delivery "
            "of good quality patient care and helps ensure "
            "timely, appropriate and effective treatment of "
            "patients. High quality data also supports "
            "governance, service planning, accountability, "
            "monitoring of activity and performance, and "
            "delivery of the Trust's goals and business "
            "priorities."
        ),
        "source": "Data Quality Policy",
        "page": 4
    },
    {
        "question": (
            "What are the characteristics of high quality "
            "data according to the policy?"
        ),
        "ground_truth": (
            "High quality data is described as accurate, "
            "up to date, comprehensive, free from duplication, "
            "valid, available when needed in a timely manner, "
            "and easily read and understood. Indicators of "
            "poor quality include demographic or activity "
            "inaccuracies within electronic or paper records."
        ),
        "source": "Data Quality Policy",
        "page": 6
    },
    {
        "question": (
            "What responsibilities does the Caldicott and "
            "Information Governance Assurance Committee "
            "have regarding data quality?"
        ),
        "ground_truth": (
            "The Committee is responsible for seeking "
            "assurance that the Trust is publishing high "
            "quality data and using high quality data to "
            "inform planning processes. It also ensures "
            "that effective systems and controls are "
            "maintained, reviews compliance with the policy, "
            "and reports to the Trust Board."
        ),
        "source": "Data Quality Policy",
        "page": 7
    },
    {
        "question": (
            "What are the responsibilities of the "
            "RTT and Data Quality Validators?"
        ),
        "ground_truth": (
            "RTT and Data Quality Validators monitor entries "
            "on the Patient Administration System to identify "
            "potential data entry errors and weaknesses in "
            "controls. They produce regular data quality "
            "reports, validate in-house reports, escalate "
            "recurring problems, and carry out monthly audits "
            "of Referral To Treatment standards and waiting "
            "list compliance."
        ),
        "source": "Data Quality Policy",
        "page": 9
    },
    {
        "question": (
            "What duties does the Clinical Systems Team "
            "have in relation to data quality?"
        ),
        "ground_truth": (
            "The Clinical Systems Team monitors the Patient "
            "Administration System for errors such as "
            "incorrect NHS Numbers, missing postcodes and "
            "duplicate registrations. The team ensures "
            "records are accurate and up to date, manages "
            "system access, investigates interface failures "
            "between systems, and updates GP and dental "
            "information when changes are identified."
        ),
        "source": "Data Quality Policy",
        "page": 10
    },

    # ── ARTIFICIAL INTELLIGENCE POLICY (TRW.D&I.POL.1502.1.1) ─
    {
        "question": (
            "What is the purpose of the Artificial "
            "Intelligence Policy at UHP?"
        ),
        "ground_truth": (
            "The policy sets out how Artificial Intelligence "
            "should be considered and used within the "
            "organisation. It supports the Trust's aim to "
            "embrace appropriately regulated and governed "
            "AI technologies that address clinical or "
            "business needs while ensuring systems are "
            "fair, explainable and secure."
        ),
        "source": "Artificial Intelligence Policy",
        "page": 1
    },
    {
        "question": (
            "What is the difference between Pathway 1 "
            "and Pathway 2 AI solutions?"
        ),
        "ground_truth": (
            "Pathway 1 relates to Generative AI tools such "
            "as email auto-response suggestions, software "
            "development auto-complete and generative "
            "documents. Pathway 2 relates to AI applications "
            "in healthcare, including image analysis, "
            "prioritisation of patient waiting lists and "
            "generative analysis of patient data."
        ),
        "source": "Artificial Intelligence Policy",
        "page": 3
    },
    {
        "question": (
            "What approvals are required before using "
            "an AI tool within the Trust?"
        ),
        "ground_truth": (
            "No AI tool or system should be used without "
            "prior consent from the Technical Design "
            "Authority and relevant governance groups. "
            "For Pathway 2 solutions, approval is also "
            "required from the Clinical Design Authority, "
            "and suppliers must provide DTAC documentation "
            "together with DCB0129 and DCB0160 safety "
            "documentation."
        ),
        "source": "Artificial Intelligence Policy",
        "page": 4
    },
    {
        "question": (
            "What guidance does the policy give about "
            "using patient data with Generative AI tools?"
        ),
        "ground_truth": (
            "The policy states that no member of the Trust "
            "should use patient or employee data with "
            "unapproved AI models. Patient data should "
            "never be uploaded to Generative AI tools "
            "such as Microsoft Copilot, Chat GPT or "
            "similar services."
        ),
        "source": "Artificial Intelligence Policy",
        "page": 7
    },
    {
        "question": (
            "What factors should staff consider when "
            "assessing an AI project?"
        ),
        "ground_truth": (
            "Staff should consider factors including data "
            "quality, fairness, accountability, privacy, "
            "explainability, transparency and costs. The "
            "policy also highlights the importance of "
            "complying with GDPR and ensuring stakeholders "
            "understand how AI models reach decisions."
        ),
        "source": "Artificial Intelligence Policy",
        "page": 8
    },

    # ── PATIENT SAFETY INCIDENT RESPONSE POLICY ───────────────
    {
        "question": (
            "What is the purpose of the Patient Safety "
            "Incident Response Policy?"
        ),
        "ground_truth": (
            "The policy supports the requirements of the "
            "Patient Safety Incident Response Framework "
            "and sets out the Trust's approach to developing "
            "and maintaining effective systems and processes "
            "for responding to patient safety incidents. "
            "Its purpose is to support learning and "
            "improvement in patient safety through a "
            "co-ordinated and data-driven approach."
        ),
        "source": "Patient Safety Incident Response Policy",
        "page": 4
    },
    {
        "question": (
            "What is meant by a systems-based approach "
            "to patient safety incident responses?"
        ),
        "ground_truth": (
            "A systems-based approach recognises that "
            "patient safety is created through interactions "
            "between components of the healthcare system "
            "rather than focusing on one individual. "
            "Responses do not take a person-focused "
            "approach or aim to apportion blame, determine "
            "liability or identify human error as the "
            "cause of an incident."
        ),
        "source": "Patient Safety Incident Response Policy",
        "page": 4
    },
    {
        "question": (
            "How can patient safety incidents be "
            "reported according to the policy?"
        ),
        "ground_truth": (
            "Patient safety incidents can be reported "
            "through multiple sources including Datix "
            "incident reporting, patient feedback via "
            "Complaints or PALS, and staff concerns "
            "raised through Freedom to Speak Up. The "
            "policy defines patient safety incidents as "
            "unintended or unexpected incidents that "
            "could have or did lead to harm for patients."
        ),
        "source": "Patient Safety Incident Response Policy",
        "page": 5
    },
    {
        "question": (
            "What is the purpose of a Patient Safety Review?"
        ),
        "ground_truth": (
            "A Patient Safety Review is a structured "
            "facilitated discussion that helps individuals "
            "involved in an event understand why the "
            "outcome differed from what was expected. "
            "It focuses on learning and improvement by "
            "examining expected outcomes, actual outcomes, "
            "differences between them and lessons learned."
        ),
        "source": "Patient Safety Incident Response Policy",
        "page": 6
    },
    {
        "question": (
            "What are the main principles underpinning "
            "the Trust's implementation of PSIRF?"
        ),
        "ground_truth": (
            "The Trust's implementation of PSIRF is "
            "underpinned by principles including improvement "
            "being the focus, blame restricting insight, "
            "learning as a proactive step toward improvement, "
            "collaboration, psychological safety and "
            "curiosity. These principles support a "
            "restorative just culture."
        ),
        "source": "Patient Safety Incident Response Policy",
        "page": 9
    },

    # ── INFORMATION GOVERNANCE POLICY (TRW.IGT.POL.373.6.1) ───
    {
        "question": (
            "What is the purpose of the "
            "Information Governance Policy?"
        ),
        "ground_truth": (
            "This policy highlights staff responsibilities "
            "when processing personal and corporate data "
            "in line with legislation, national guidance "
            "and reviews. It also signposts staff to key "
            "Standard Operating Procedures and explains "
            "how information should be processed."
        ),
        "source": "Information Governance Policy",
        "page": 1
    },
    {
        "question": (
            "Who should read the "
            "Information Governance Policy?"
        ),
        "ground_truth": (
            "All staff that handle personal data or "
            "corporate information must have an awareness "
            "of the principles set out in this policy."
        ),
        "source": "Information Governance Policy",
        "page": 1
    },
    {
        "question": (
            "What legislation governs information "
            "governance at UHP?"
        ),
        "ground_truth": (
            "The UK Data Protection Act 2018 and the UK "
            "General Data Protection Regulation govern "
            "the processing of personal data of living "
            "individuals. The Common Law of Confidentiality "
            "extends after death, and the Freedom of "
            "Information Act 2000 allows individuals to "
            "request corporate information from public "
            "sector organisations."
        ),
        "source": "Information Governance Policy",
        "page": 1
    },
    {
        "question": (
            "What is a Record of Processing "
            "Activities at UHP?"
        ),
        "ground_truth": (
            "Under Section 61 of the DPA 2018 and Article "
            "30 of GDPR, the Trust must keep a record of "
            "its processing activities and the legal basis. "
            "The Trust's approach includes an Information "
            "Asset Register, Records Inventories and "
            "Data Flows."
        ),
        "source": "Information Governance Policy",
        "page": 6
    },
    {
        "question": (
            "How are Freedom of Information requests "
            "handled at UHP?"
        ),
        "ground_truth": (
            "The Freedom of Information Act 2000 provides "
            "individuals with a right of access to corporate "
            "information held by the Trust. There are 23 "
            "exemptions to disclosure, and for a potential "
            "qualified exemption the Trust must apply a "
            "public interest test."
        ),
        "source": "Information Governance Policy",
        "page": 11
    },

    # ── INFORMATION SECURITY POLICY (TRW.IGT.POL.139.7) ───────
    {
        "question": (
            "What is the purpose of the "
            "Information Security Policy?"
        ),
        "ground_truth": (
            "This policy defines information security "
            "protocols, procedures and controls to protect "
            "all electronic information assets held on and "
            "processed by Trust systems, staff and "
            "contractors from internal or external damage, "
            "either deliberately or accidentally."
        ),
        "source": "Information Security Policy",
        "page": 1
    },
    {
        "question": (
            "What are the main security principles "
            "set out in the Information Security Policy?"
        ),
        "ground_truth": (
            "The purpose of the Information Security Policy "
            "is to preserve confidentiality and integrity. "
            "Confidentiality means access to data must be "
            "confined to those with specific authority to "
            "view it, and integrity means information is "
            "to be complete and accurate."
        ),
        "source": "Information Security Policy",
        "page": 5
    },
    {
        "question": (
            "What does the policy require for staff "
            "information security training?"
        ),
        "ground_truth": (
            "The mandatory Data Security Awareness training "
            "contains key information security and governance "
            "principles, and all staff are to complete it "
            "annually. The Information Governance Team and "
            "Cyber Security Team can provide additional "
            "awareness and training."
        ),
        "source": "Information Security Policy",
        "page": 8
    },
    {
        "question": (
            "What are the rules for using Trust "
            "devices and software?"
        ),
        "ground_truth": (
            "Staff must not install software on the "
            "organisation's property without permission "
            "from the D&I Service, and only approved "
            "software may be installed on Trust systems. "
            "Trust devices are issued for work purposes "
            "and must not be used for personal use or "
            "social purposes such as gaming or "
            "entertainment streaming."
        ),
        "source": "Information Security Policy",
        "page": 9
    },
    {
        "question": (
            "What should staff do if a "
            "security incident happens?"
        ),
        "ground_truth": (
            "All security incidents and weaknesses relating "
            "to any information asset are to be reported "
            "to the D&I Service Desk immediately. Any "
            "compromise to patient safety, confidentiality "
            "or integrity of the clinical record should "
            "also be reported on Datix and to the "
            "Information Governance Team."
        ),
        "source": "Information Security Policy",
        "page": 14
    },
]


# ─── COLLECT RAG OUTPUTS ──────────────────────────────────────
def collect_rag_outputs(test_set: list) -> dict:
    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []
    total = len(test_set)

    for i, item in enumerate(test_set, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]
        print(f"\n[{i}/{total}] {question[:60]}...")

        try:
            result      = query_policies(question)
            answer      = result.get("answer", "")
            rewritten   = rewrite_query(question)
            docs        = hybrid_search(rewritten)
            top_docs, _ = rerank(question, docs)
            chunk_texts = [doc.page_content for doc in top_docs]

            questions.append(question)
            answers.append(answer)
            contexts.append(chunk_texts)
            ground_truths.append(ground_truth)

            expected = item["source"]
            actual   = result.get("source", "N/A")
            match    = "✅" if expected in actual else "⚠️"
            print(f"  {match} Got: {actual} | Expected: {expected}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            questions.append(question)
            answers.append("Error retrieving answer")
            contexts.append([""])
            ground_truths.append(ground_truth)

    return {
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    }


# ─── RUN RAGAS ────────────────────────────────────────────────
def run_ragas_evaluation(data: dict) -> dict:
    print("\n" + "=" * 60)
    print("Running RAGAS evaluation...")
    print("LLM judge:   Claude Haiku (temperature=0)")
    print("Embeddings:  all-MiniLM-L6-v2 (local)")
    print("=" * 60)

    dataset = Dataset.from_dict(data)

    llm = LangchainLLMWrapper(
        ChatAnthropic(
            model="claude-haiku-4-5",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=2048,
            temperature=0,
        )
    )

    embeddings = LangchainEmbeddingsWrapper(_hf_embeddings)

    results = evaluate(
        dataset    = dataset,
        metrics    = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm        = llm,
        embeddings = embeddings,
        batch_size = 2,
    )

    return results


# ─── PRINT REPORT ─────────────────────────────────────────────
def print_report(results, data: dict) -> dict:
    print("\n" + "=" * 60)
    print("UHP POLICY CHATBOT — RAGAS EVALUATION REPORT")
    print(f"Timestamp:  {datetime.now().isoformat()}")
    print(f"Questions:  {len(data['question'])}")
    print(f"Documents:  5 UHP policy PDFs")
    print(f"Ground truths: from actual PDF content")
    print("=" * 60)

    def safe_score(val) -> float:
        if isinstance(val, list):
            cleaned = [v for v in val if v is not None and not (isinstance(v, float) and np.isnan(v))]
            return float(np.mean(cleaned)) if cleaned else 0.0
        if val is None:
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    scores = {
        "faithfulness":      safe_score(results["faithfulness"]),
        "answer_relevancy":  safe_score(results["answer_relevancy"]),
        "context_precision": safe_score(results["context_precision"]),
        "context_recall":    safe_score(results["context_recall"]),
    }

    print("\nRESULTS vs TARGETS:")
    print("-" * 60)

    all_pass = True
    for metric, score in scores.items():
        target = TARGETS[metric]
        passed = score >= target
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False

        filled = int(score * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        print(f"\n{metric}:")
        print(f"  Score:  {score:.3f}  |{bar}|")
        print(f"  Target: {target:.3f}  {status}")

    print("\n" + "=" * 60)
    overall = (
        "✅ ALL METRICS PASS — ready for TDA review"
        if all_pass else
        "⚠️  SOME METRICS BELOW TARGET — review needed"
    )
    print(f"OVERALL: {overall}")
    print("=" * 60)

    print("\nIMPROVEMENT NOTES:")
    if scores["context_precision"] < TARGETS["context_precision"]:
        print("  → Context precision: consider BiomedBERT")
        print("    embeddings or fine-tune on UHP query pairs")
    if scores["context_recall"] < TARGETS["context_recall"]:
        print("  → Context recall: increase TOP_K_RETRIEVE")
    if scores["faithfulness"] < TARGETS["faithfulness"]:
        print("  → Faithfulness: strengthen system prompt")
    if scores["answer_relevancy"] < TARGETS["answer_relevancy"]:
        print("  → Answer relevancy: increase TOP_K_RERANK")
    if all_pass:
        print("  → All metrics above target.")
        print("    Ready for TDA approval and deployment.")

    report = {
        "timestamp":    datetime.now().isoformat(),
        "questions":    len(data["question"]),
        "documents":    5,
        "scores":       scores,
        "targets":      TARGETS,
        "passed":       all_pass,
        "ground_truth_source": (
            "Questions and answers extracted directly "
            "from UHP policy PDFs"
        )
    }

    with open("ragas_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved: ragas_results.json")
    return report


# ─── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    init_audit_db()

    print("=" * 60)
    print("UHP NHS POLICY CHATBOT — RAGAS EVALUATION")
    print("Compliant with UHP AI Policy TRW.D&I.POL.1502.1.1")
    print(f"Test set: {len(GOLDEN_TEST_SET)} questions")
    print("Ground truths: extracted from actual UHP PDFs")
    print("=" * 60)

    print("\nStep 1: Collecting RAG outputs...")
    data_file = "ragas_data.pkl"
    if os.path.exists(data_file):
        print("  Loading cached RAG outputs (delete ragas_data.pkl to re-run)...")
        with open(data_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = collect_rag_outputs(GOLDEN_TEST_SET)
        with open(data_file, "wb") as f:
            pickle.dump(data, f)
        print("  RAG outputs cached to ragas_data.pkl")

    print("\nStep 2: Running RAGAS evaluation...")
    results = run_ragas_evaluation(data)

    print_report(results, data)