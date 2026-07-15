import os
import re
from typing import TypedDict, List, Optional, Literal
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

AUDIT_LOG_PATH = "audit_log.jsonl"
HUMAN_REVIEW_QUEUE_PATH = "human_review_queue.jsonl"

# ---------- Model configuration ----------
GRADER_MODEL = "llama-3.1-8b-instant"  # Adjusted to standard Groq model (change back to gpt-oss-20b if using a proxy)
DECISION_MODEL = "llama-3.3-70b-versatile" # Adjusted to standard Groq model

TOKEN_LIMITS = {
    "build_query": 100,
    "grade_relevance": 250,
    "rewrite_query": 100,
    "decision": 600,
    "grounding_check": 350,
}

PROMPT_VERSION = "claims-agent-prompts-v2.0"
MAX_RETRIEVAL_RETRIES_CAP = 2


# ---------- Sanitization ----------
def sanitize_text(text: str) -> str:
    """
    Remove all non-ASCII characters from the text.
    Ensures no special characters (emojis, smart quotes, etc.) cause encoding errors or tool choice crashes.
    """
    return re.sub(r'[^\x00-\x7F]+', ' ', text)


# ---------- Schemas ----------
class ClaimInput(BaseModel):
    """Structured, validated representation of an incoming claim request."""
    claim_query: str = Field(
        ..., 
        min_length=10, 
        max_length=2000,
        description="Free-text description of what happened and what coverage is being asked about"
    )
    policy_type: Optional[Literal["auto", "health", "home", "other"]] = Field(
        default=None, 
        description="Type of policy this claim falls under, if known"
    )
    policy_number: Optional[str] = Field(default=None, max_length=50)
    claimant_name: Optional[str] = Field(default=None, max_length=200)
    date_of_loss: Optional[str] = Field(
        default=None, 
        description="Date of loss, if known (any readable format)"
    )
    claimed_amount: Optional[float] = Field(
        default=None, 
        ge=0, 
        description="Dollar amount being claimed, if known"
    )

    @field_validator("claim_query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim_query cannot be blank")
        return sanitize_text(v.strip())


class RelevanceGrade(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Decision(BaseModel):
    verdict: Literal["approve", "deny", "escalate"]
    reasoning: str
    cited_clause_ids: List[str]


class HallucinationGrade(BaseModel):
    grounded: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AuditRecord(BaseModel):
    claim_id: str
    claim_query: str
    verdict: Literal["approve", "deny", "escalate"]
    reasoning: str
    relevance_confidence: Optional[float]
    hallucination_confidence: Optional[float]
    cited_clause_ids: List[str]
    retrieved_clause_ids: List[str]
    search_queries_tried: List[str]
    retry_count: int
    web_sources_consulted: List[dict] = []
    requires_human_review: bool
    escalation_priority: Optional[str] = None
    model_name_grader: str
    model_name_decision: str
    prompt_version: str
    timestamp: str


class ClaimState(TypedDict):
    claim_id: str
    raw_claim_input: dict
    claim_input: Optional[dict]
    validation_error: Optional[str]
    claim_query: str
    search_query: str
    query_history: List[str]
    vectorstore: object
    retrieved_docs: List[Document]
    relevance_grade: Optional[RelevanceGrade]
    retry_count: int
    max_retries: int
    relevance_threshold: float
    decision: Optional[Decision]
    hallucination_grade: Optional[HallucinationGrade]
    decision_retry_count: int
    hallucination_threshold: float
    audit_record: Optional[dict]
    enable_web_fallback: bool
    tavily_client: object
    web_search_results: List[dict]


# ---------- System prompts ----------
BUILD_QUERY_SYSTEM_PROMPT = """You are preparing the first search query against an insurance policy database for a newly submitted claim.

Policy type (if known): {policy_type}
Claim scenario: {claim_query}

Write ONE concise search query using terms likely to appear in policy documents
(coverage, exclusion, section names) rather than conversational language.
Return ONLY the query text, nothing else."""

RELEVANCE_GRADER_SYSTEM_PROMPT = """You are a meticulous insurance claims analyst evaluating whether retrieved policy text actually addresses a specific claim scenario.
You MUST output your evaluation using the provided tool schema.

Score how well the retrieved clauses cover the claim scenario on a 0.0-1.0 scale:
- 1.0: The clauses directly and specifically resolve the claim scenario
- 0.5: The clauses are topically related but don't clearly resolve the scenario
- 0.0: The clauses are unrelated to the claim scenario

Claim scenario: {claim_query}

Retrieved policy clauses:
{retrieved_context}"""

REWRITE_SYSTEM_PROMPT = """You are refining a search query against an insurance policy database.
The previous search attempt did not retrieve clauses that clearly resolve the claim scenario.

Original claim scenario: {claim_query}
Previous search query: {search_query}
Why the previous retrieval was weak: {grade_reasoning}

Write ONE improved search query, phrased using terms likely to appear in policy documents.
Return ONLY the query text, nothing else."""

DECISION_SYSTEM_PROMPT = """You are a senior insurance claims adjudicator. Decide this claim using ONLY the retrieved policy clauses below.
You MUST output your evaluation using the provided tool schema.

Claim scenario: {claim_query}

Retrieved policy clauses:
{retrieved_context}

Rules:
- APPROVE only if a specific clause explicitly grants coverage.
- DENY only if a specific clause explicitly excludes coverage.
- ESCALATE if clauses are ambiguous, conflicting, or do not cover the scenario.
- Be conservative: when in doubt, escalate rather than guess."""

HALLUCINATION_GRADER_SYSTEM_PROMPT = """You are a fact-checking auditor. Verify a claims decision is fully grounded in the retrieved policy text.
You MUST output your evaluation using the provided tool schema.

Retrieved policy clauses:
{retrieved_context}

Decision made: {verdict}
Decision reasoning: {decision_reasoning}
Clauses cited: {cited_clause_ids}"""

AUTHORITATIVE_INSURANCE_DOMAINS = ["naic.org", "content.naic.org"]


# ---------- Ingestion ----------
def build_vectorstore_from_files(file_paths: List[str]) -> FAISS:
    raw_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path)
        docs = loader.load()
        # Clean the text to prevent ASCII encoding issues down the pipeline
        for doc in docs:
            doc.page_content = sanitize_text(doc.page_content)
        raw_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)
    for i, chunk in enumerate(chunks):
        source = os.path.basename(chunk.metadata.get("source", "unknown"))
        chunk.metadata["clause_id"] = f"{source}::chunk_{i}"
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def format_docs_for_grading(docs):
    return "\n\n".join(f"[{d.metadata['clause_id']}]\n{d.page_content}" for d in docs)


def _escalation_priority(claim_input: Optional[dict]) -> str:
    if claim_input and claim_input.get("claimed_amount") is not None and claim_input["claimed_amount"] >= 25000:
        return "high"
    return "normal"


# ---------- Graph builder ----------
def build_claims_graph(groq_api_key: str):
    build_query_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["build_query"], api_key=groq_api_key)
    grader_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["grade_relevance"], api_key=groq_api_key)
    rewrite_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["rewrite_query"], api_key=groq_api_key)
    grounding_llm = ChatGroq(model=GRADER_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["grounding_check"], api_key=groq_api_key)
    strong_llm = ChatGroq(model=DECISION_MODEL, temperature=0, max_tokens=TOKEN_LIMITS["decision"], api_key=groq_api_key)

    build_query_chain = ChatPromptTemplate.from_messages([("system", BUILD_QUERY_SYSTEM_PROMPT)]) | build_query_llm

    # Method function_calling explicitly enforces the model uses the tool
    relevance_grader_chain = (
        ChatPromptTemplate.from_messages([("system", RELEVANCE_GRADER_SYSTEM_PROMPT)])
        | grader_llm.with_structured_output(RelevanceGrade, method="function_calling")
    )
    rewrite_chain = ChatPromptTemplate.from_messages([("system", REWRITE_SYSTEM_PROMPT)]) | rewrite_llm
    decision_chain = (
        ChatPromptTemplate.from_messages([("system", DECISION_SYSTEM_PROMPT)])
        | strong_llm.with_structured_output(Decision, method="function_calling")
    )
    hallucination_chain = (
        ChatPromptTemplate.from_messages([("system", HALLUCINATION_GRADER_SYSTEM_PROMPT)])
        | grounding_llm.with_structured_output(HallucinationGrade, method="function_calling")
    )

    # ----- Nodes -----
    def validate_claim_node(state):
        try:
            # Sanitize dictionary values
            clean_input = {k: sanitize_text(str(v)) if isinstance(v, str) else v for k, v in state["raw_claim_input"].items()}
            claim_input = ClaimInput(**clean_input)
            return {
                "claim_input": claim_input.model_dump(),
                "claim_query": claim_input.claim_query,
                "validation_error": None,
            }
        except ValidationError as e:
            return {"claim_input": None, "validation_error": str(e)}

    def invalid_input_node(state):
        reasoning = f"Claim input failed validation and cannot be processed automatically: {state['validation_error']}"
        return {"decision": Decision(verdict="escalate", reasoning=reasoning, cited_clause_ids=[])}

    def build_query_node(state):
        claim_input = state["claim_input"] or {}
        query = build_query_chain.invoke({
            "policy_type": claim_input.get("policy_type") or "unknown",
            "claim_query": state["claim_query"],
        }).content.strip()
        return {"search_query": query, "query_history": [query]}

    def retrieve_node(state):
        docs = state["vectorstore"].similarity_search(state["search_query"], k=3)
        return {"retrieved_docs": docs}

    def grade_relevance_node(state):
        context = format_docs_for_grading(state["retrieved_docs"])
        grade = relevance_grader_chain.invoke({"claim_query": state["claim_query"], "retrieved_context": context})
        return {"relevance_grade": grade}

    def rewrite_query_node(state):
        new_query = rewrite_chain.invoke({
            "claim_query": state["claim_query"], 
            "search_query": state["search_query"],
            "grade_reasoning": state["relevance_grade"].reasoning,
        }).content.strip()
        return {
            "search_query": new_query, 
            "query_history": state["query_history"] + [new_query],
            "retry_count": state["retry_count"] + 1
        }

    def web_regulation_fallback_node(state):
        if not state["enable_web_fallback"] or state["tavily_client"] is None:
            return {"web_search_results": []}
        
        query = f"insurance regulation coverage requirements: {state['claim_query']}"
        try:
            response = state["tavily_client"].search(
                query=query, 
                max_results=3, 
                search_depth="basic",
                include_domains=AUTHORITATIVE_INSURANCE_DOMAINS,
            )
            results = [
                {
                    "title": sanitize_text(r["title"]), 
                    "url": r["url"], 
                    "snippet": sanitize_text(r["content"][:300])
                }
                for r in response.get("results", [])
            ]
        except Exception:
            results = []
        return {"web_search_results": results}

    def decide_node(state):
        context = format_docs_for_grading(state["retrieved_docs"])
        decision = decision_chain.invoke({"claim_query": state["claim_query"], "retrieved_context": context})
        return {"decision": decision}

    def check_grounding_node(state):
        context = format_docs_for_grading(state["retrieved_docs"])
        grade = hallucination_chain.invoke({
            "retrieved_context": context, 
            "verdict": state["decision"].verdict,
            "decision_reasoning": state["decision"].reasoning,
            "cited_clause_ids": state["decision"].cited_clause_ids,
        })
        return {"hallucination_grade": grade}

    def increment_decision_retry_node(state):
        return {"decision_retry_count": state["decision_retry_count"] + 1}

    def escalate_insufficient_evidence_node(state):
        reasoning = (f"No policy clauses met the relevance threshold ({state['relevance_threshold']}) "
                     f"after {state['retry_count']} retrieval attempt(s). "
                     f"Last grader reasoning: {state['relevance_grade'].reasoning}")
        if state["web_search_results"]:
            sources = "; ".join(f"{r['title']} ({r['url']})" for r in state["web_search_results"])
            reasoning += f"\n\nSupplementary regulatory research for the human reviewer: {sources}"
        else:
            reasoning += "\n\nNo supplementary regulatory sources found."
        return {"decision": Decision(verdict="escalate", reasoning=reasoning, cited_clause_ids=[])}

    def escalate_ungrounded_node(state):
        reasoning = (f"Automated decision ('{state['decision'].verdict}') failed grounding verification "
                     f"(confidence {state['hallucination_grade'].confidence:.2f}). Escalating for manual review. "
                     f"Original reasoning: {state['decision'].reasoning}")
        return {"decision": Decision(verdict="escalate", reasoning=reasoning,
                                      cited_clause_ids=state["decision"].cited_clause_ids)}

    def finalize_node(state):
        requires_human_review = state["decision"].verdict == "escalate"
        priority = _escalation_priority(state.get("claim_input")) if requires_human_review else None
        
        record = AuditRecord(
            claim_id=state["claim_id"], 
            claim_query=state["claim_query"],
            verdict=state["decision"].verdict, 
            reasoning=state["decision"].reasoning,
            relevance_confidence=state["relevance_grade"].confidence if state["relevance_grade"] else None,
            hallucination_confidence=state["hallucination_grade"].confidence if state["hallucination_grade"] else None,
            cited_clause_ids=state["decision"].cited_clause_ids,
            retrieved_clause_ids=[d.metadata["clause_id"] for d in state["retrieved_docs"]],
            search_queries_tried=state["query_history"], 
            retry_count=state["retry_count"],
            web_sources_consulted=state["web_search_results"],
            requires_human_review=requires_human_review,
            escalation_priority=priority,
            model_name_grader=GRADER_MODEL,
            model_name_decision=DECISION_MODEL,
            prompt_version=PROMPT_VERSION,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        with open(AUDIT_LOG_PATH, "a") as f:
            f.write(record.model_dump_json() + "\n")
            
        if requires_human_review:
            with open(HUMAN_REVIEW_QUEUE_PATH, "a") as f:
                f.write(record.model_dump_json() + "\n")
                
        return {"audit_record": record.model_dump()}

    # ----- Routing -----
    def route_after_validation(state):
        return "invalid_input" if state["validation_error"] else "build_query"

    def route_after_grade(state):
        grade = state["relevance_grade"]
        if grade.confidence >= state["relevance_threshold"]:
            return "decide"
        if state["retry_count"] < state["max_retries"]:
            return "rewrite_query"
        return "web_fallback"

    def route_after_hallucination(state):
        h = state["hallucination_grade"]
        if h.grounded and h.confidence >= state["hallucination_threshold"]:
            return "finalize"
        if state["decision_retry_count"] < 1:
            return "retry_decision"
        return "escalate_ungrounded"

    # ----- Graph assembly -----
    builder = StateGraph(ClaimState)
    for name, fn in [
        ("validate_claim", validate_claim_node),
        ("invalid_input", invalid_input_node),
        ("build_query", build_query_node),
        ("retrieve", retrieve_node),
        ("grade_relevance", grade_relevance_node),
        ("rewrite_query", rewrite_query_node),
        ("web_fallback", web_regulation_fallback_node),
        ("decide", decide_node),
        ("check_grounding", check_grounding_node),
        ("increment_decision_retry", increment_decision_retry_node),
        ("escalate_insufficient_evidence", escalate_insufficient_evidence_node),
        ("escalate_ungrounded", escalate_ungrounded_node),
        ("finalize", finalize_node),
    ]:
        builder.add_node(name, fn)

    builder.add_edge(START, "validate_claim")
    builder.add_conditional_edges("validate_claim", route_after_validation,
        {"build_query": "build_query", "invalid_input": "invalid_input"})
    builder.add_edge("invalid_input", "finalize")
    builder.add_edge("build_query", "retrieve")
    builder.add_edge("retrieve", "grade_relevance")
    builder.add_conditional_edges("grade_relevance", route_after_grade,
        {"rewrite_query": "rewrite_query", "decide": "decide", "web_fallback": "web_fallback"})
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("web_fallback", "escalate_insufficient_evidence")
    builder.add_edge("decide", "check_grounding")
    builder.add_conditional_edges("check_grounding", route_after_hallucination,
        {"finalize": "finalize", "retry_decision": "increment_decision_retry", "escalate_ungrounded": "escalate_ungrounded"})
    builder.add_edge("increment_decision_retry", "decide")
    builder.add_edge("escalate_insufficient_evidence", "finalize")
    builder.add_edge("escalate_ungrounded", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


def adjudicate(graph, claim_id, claim_query, vectorstore, tavily_client,
               relevance_threshold=0.7, hallucination_threshold=0.7,
               max_retries=2, enable_web_fallback=True,
               policy_type=None, policy_number=None, claimant_name=None,
               date_of_loss=None, claimed_amount=None):

    max_retries = min(max_retries, MAX_RETRIEVAL_RETRIES_CAP)

    raw_claim_input = {
        "claim_query": claim_query,
        "policy_type": policy_type,
        "policy_number": policy_number,
        "claimant_name": claimant_name,
        "date_of_loss": date_of_loss,
        "claimed_amount": claimed_amount,
    }

    return graph.invoke({
        "claim_id": claim_id,
        "raw_claim_input": raw_claim_input,
        "claim_input": None,
        "validation_error": None,
        "claim_query": claim_query,
        "search_query": claim_query,
        "query_history": [],
        "vectorstore": vectorstore,
        "retrieved_docs": [],
        "relevance_grade": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "relevance_threshold": relevance_threshold,
        "decision": None,
        "hallucination_grade": None,
        "decision_retry_count": 0,
        "hallucination_threshold": hallucination_threshold,
        "audit_record": None,
        "enable_web_fallback": enable_web_fallback,
        "tavily_client": tavily_client,
        "web_search_results": [],
    })

import streamlit as st
import os, uuid
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

#from insurance_agent import (
    #build_claims_graph, build_vectorstore_from_files, adjudicate,
    #AUDIT_LOG_PATH, HUMAN_REVIEW_QUEUE_PATH,
#)

# ----------------- EMAIL NOTIFICATION FEATURE -----------------
def send_claim_email(record):
    """
    Sends a beautifully formatted HTML email with the claim decision.
    Uses sample credentials for demonstration.
    """
    sender_email = "pawasepramod@gmail.com"
    receiver_email = "pramodap2023@gmail.com"
    app_password = "rgct ynpt sduz ceng"
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Claim Adjudication Result: {record['claim_id']} - {record['verdict'].upper()}"
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    # Dynamic styling based on verdict
    color = "#28a745" if record['verdict'] == "approve" else "#dc3545" if record['verdict'] == "deny" else "#ffc107"
    
    # HTML Email Body with good fonts and layout
    html = f"""
    <html>
      <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; line-height: 1.6; padding: 20px; background-color: #f4f7f6;">
        <div style="max-width: 650px; margin: 0 auto; background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
          <div style="background-color: #f8f9fa; padding: 25px; border-bottom: 2px solid {color};">
            <h2 style="margin: 0; color: #2c3e50; font-size: 24px;">🗂️ Claim Guard Notification</h2>
          </div>
          <div style="padding: 30px;">
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Claim ID:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee; text-align: right;">{record['claim_id']}</td>
                </tr>
                <tr>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee;"><strong>Verdict:</strong></td>
                    <td style="padding: 10px 0; border-bottom: 1px solid #eee; text-align: right;">
                        <span style="color: {color}; font-weight: 800; font-size: 16px; text-transform: uppercase;">{record['verdict']}</span>
                    </td>
                </tr>
            </table>
            
            <h3 style="color: #2c3e50; margin-top: 30px; font-size: 18px;">📝 Claim Query</h3>
            <p style="background-color: #f1f3f5; padding: 15px; border-radius: 6px; font-size: 14px; border-left: 4px solid #ced4da;">
                {record['claim_query']}
            </p>
            
            <h3 style="color: #2c3e50; margin-top: 30px; font-size: 18px;">⚖️ AI Reasoning</h3>
            <p style="font-size: 14px; color: #555; background-color: #fffaf0; padding: 15px; border-radius: 6px; border: 1px solid #f0e6d2;">
                {record['reasoning']}
            </p>
          </div>
          <div style="background-color: #f8f9fa; padding: 15px 20px; text-align: center; font-size: 12px; color: #888; border-top: 1px solid #eee;">
            This is an automated message generated by the Claim Guard AI Agent. Please do not reply directly to this email.
          </div>
        </div>
      </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))
    
    # Send Email (Mocked for safety since sample credentials are used)
    try:
        # To make this live, uncomment below and use a real SMTP server:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        # Use the app_password here, NOT your regular gmail password
        server.login(sender_email, app_password) 
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        return True, f"Email successfully sent to {receiver_email}"
    except Exception as e:
        return False, str(e)


# ----------------- MAIN APP UI -----------------
st.set_page_config(page_title="Claims Adjudication Agent", page_icon="🗂️", layout="wide")
st.title("🗂️ Claim Guard AI Agent")
st.caption("Self-correcting retrieval, grounded decisions, and human escalation when evidence is weak.")

with st.sidebar:
    st.header("🔑 API Keys")
    groq_key = st.text_input("Groq API Key", type="password")
    tavily_key = st.text_input("Tavily API Key (optional)", type="password")

    st.header("⚙️ Settings")
    relevance_threshold = st.slider("Relevance threshold", 0.0, 1.0, 0.7, 0.05)
    hallucination_threshold = st.slider("Grounding threshold", 0.0, 1.0, 0.7, 0.05)
    max_retries = st.slider("Max retrieval retries", 0, 2, 2,
                            help="Hard-capped at 2 by the agent regardless of this setting.")
    enable_web_fallback = st.checkbox("Enable web regulation fallback", value=bool(tavily_key))

    st.header("📄 Policy Documents")
    uploaded_files = st.file_uploader("Upload policy docs (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    build_index = st.button("Build / Rebuild Index")

    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "rb") as f:
            st.download_button("⬇️ Download audit log (JSON)", f, file_name="audit_log.jsonl")
    if os.path.exists(HUMAN_REVIEW_QUEUE_PATH):
        with open(HUMAN_REVIEW_QUEUE_PATH, "rb") as f:
            st.download_button("⬇️ Download human review queue (JSON)", f, file_name="human_review_queue.jsonl")

# Initialize session states securely
for key, default in [("vectorstore", None), ("graph", None), ("history", [])]:
    if key not in st.session_state:
        st.session_state[key] = default

if build_index:
    if not uploaded_files:
        st.sidebar.error("Upload at least one document first.")
    else:
        os.makedirs("uploaded_docs", exist_ok=True)
        paths = []
        for f in uploaded_files:
            path = os.path.join("uploaded_docs", f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)
        
        with st.spinner("Chunking and embedding documents (Cleaning ASCII data)..."):
            try:
                st.session_state.vectorstore = build_vectorstore_from_files(paths)
                st.sidebar.success(f"Indexed {len(paths)} document(s).")
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")

if groq_key and st.session_state.graph is None:
    st.session_state.graph = build_claims_graph(groq_key)

tavily_client = None
if tavily_key:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=tavily_key)

st.subheader("Submit a claim")
claim_query = st.text_area("Describe the claim scenario", height=100,
                           placeholder="e.g. Customer's basement flooded because a pipe burst under the sink...")

with st.expander("➕ Additional claim details (optional)"):
    col1, col2 = st.columns(2)
    with col1:
        policy_type = st.selectbox("Policy type", [None, "auto", "health", "home", "other"], index=0)
        policy_number = st.text_input("Policy number")
        claimant_name = st.text_input("Claimant name")
    with col2:
        date_of_loss = st.text_input("Date of loss")
        claimed_amount = st.number_input("Claimed amount ($)", min_value=0.0, value=0.0, step=100.0)

send_email_notification = st.checkbox("📧 Send Email Notification on Decision", value=True)
submit = st.button("Adjudicate Claim", type="primary")

if submit:
    if not groq_key:
        st.error("Enter your Groq API key in the sidebar.")
    elif not claim_query.strip():
        st.error("Describe a claim scenario first.")
    elif st.session_state.vectorstore is None:
        st.error("Upload and index at least one policy document in the sidebar first.")
    else:
        claim_id = str(uuid.uuid4())[:8]
        with st.spinner("Validating, retrieving evidence, grading, and adjudicating..."):
            try:
                result = adjudicate(
                    st.session_state.graph, claim_id, claim_query,
                    st.session_state.vectorstore, tavily_client,
                    relevance_threshold, hallucination_threshold,
                    max_retries, enable_web_fallback,
                    policy_type=policy_type or None,
                    policy_number=policy_number or None,
                    claimant_name=claimant_name or None,
                    date_of_loss=date_of_loss or None,
                    claimed_amount=claimed_amount if claimed_amount > 0 else None,
                )
                
                decision = result["decision"]
                color = {"approve": "green", "deny": "red", "escalate": "orange"}[decision.verdict]
                st.markdown(f"### Verdict: :{color}[{decision.verdict.upper()}]")
                
                st.write(decision.reasoning)
                
                if decision.cited_clause_ids:
                    st.caption("Cited clauses: " + ", ".join(decision.cited_clause_ids))
                    
                if result["audit_record"].get("requires_human_review"):
                    st.warning(f"🧑‍⚖️ Routed to human adjuster review "
                               f"(priority: {result['audit_record'].get('escalation_priority')}).")
                               
                with st.expander("Full audit record"):
                    st.json(result["audit_record"])
                    
                # Store output to history properly
                st.session_state.history.append(result["audit_record"])
                
                # Execute Email logic if enabled
                if send_email_notification:
                    success, msg = send_claim_email(result["audit_record"])
                    if success:
                        st.toast(f"✅ {msg}", icon="📧")
                    else:
                        st.toast(f"❌ Failed to send email: {msg}", icon="🚨")
                
            except Exception as e:
                st.error("⚠️ The agent hit an error and could not complete automated adjudication.")
                st.write(f"Details: {e}")
                st.warning(f"Recommendation: escalate claim `{claim_id}` to a human reviewer manually.")

# ----------------- SESSION HISTORY & CSV EXPORT -----------------
if st.session_state.history:
    st.subheader("📋 Session claim history")
    
    # Map the session history to a Pandas DataFrame for full data visibility
    df_history = pd.DataFrame([
        {
            "Claim ID": r["claim_id"], 
            "Verdict": r["verdict"].upper(),
            "Human Review Needed": r.get("requires_human_review", False),
            "Claim Query": r["claim_query"], # Full query instead of truncated
            "Reasoning": r["reasoning"]      # New: Added Full reasoning
        }
        for r in reversed(st.session_state.history) # Latest first
    ])
    
    # Display the full dataframe in Streamlit
    st.dataframe(df_history, use_container_width=True)
    
    # Add a clean CSV download button
    csv_data = df_history.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download History as CSV",
        data=csv_data,
        file_name="session_claim_history.csv",
        mime="text/csv",
        type="secondary"
    )				   
