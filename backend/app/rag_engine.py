import os
import re
import json
import warnings
warnings.filterwarnings('ignore')

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> reasoning tokens emitted by chain-of-thought models (Qwen, etc)."""
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return stripped.strip()

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env'), override=True)

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB persistent client
CHROMA_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'chroma_db')

SYSTEM_ARCHITECTURE_SUMMARY = (
    "SYSTEM ARCHITECTURE & PLATFORM OVERVIEW:\n"
    "Name: Settlement Reconciliation & Forecasting Copilot (Razorpay AI Buildathon Track 04)\n"
    "Purpose: Enterprise AI solution automating bank-to-ledger reconciliation, confidence scoring, RAG Q&A, forward cash forecasting, and exception management.\n"
    "Core Components:\n"
    "1. Rule Engine (Tiers 1-3): Automated heuristic matching on exact reference, MDR fee variance (<=3.5%), date gap, and fuzzy string similarity (64.89% match rate).\n"
    "2. Random Forest Scorer (Tier 4): Evaluates ambiguous unmatched records with 5-Fold Stratified Cross-Validation.\n"
    "3. ChromaDB Vector Store & RAG Copilot: Persistent vector search repository powering grounded AI Q&A.\n"
    "4. Monte Carlo Forecaster: 10,000 simulations projecting 30-day cash positions (P10, P50, P90).\n"
    "5. Honest Exception Queue: Surfaces low-confidence unresolved items (<0.70) for mandatory Finance Controller review."
)

def get_chroma_collection():
    os.makedirs(CHROMA_DATA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DATA_DIR)
    
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(
        name="settlement_records",
        embedding_function=emb_fn
    )
    return collection

def format_record_for_embedding(row: pd.Series) -> str:
    """Formats a reconciliation record into a semantic text block for vector indexing."""
    ledger_id = row.get('ledger_txn_id', 'N/A')
    order_id = row.get('merchant_order_id', 'N/A')
    bank_id = row.get('bank_txn_id', 'N/A')
    bank_nar = row.get('bank_narration', 'N/A')
    status = row.get('status', row.get('ml_prediction', 'UNMATCHED'))
    tier = row.get('match_tier', 'N/A')
    net_amt = row.get('net_amount', 0.0)
    bank_amt = row.get('bank_amount', 0.0)
    amt_diff = row.get('amount_diff', 0.0)
    date_gap = row.get('date_gap_days', 0)
    ref_sim = row.get('ref_similarity', 0.0)
    ml_conf = row.get('ml_confidence_score', row.get('confidence_score', 0.0))
    
    text = (
        f"Reconciliation Record Summary:\n"
        f"- Merchant Order ID: {order_id}\n"
        f"- Ledger Txn ID: {ledger_id} | Bank Txn ID: {bank_id}\n"
        f"- Match Tier: {tier} | Status: {status}\n"
        f"- Confidence Score: {ml_conf}\n"
        f"- Financial Amounts: Ledger Net = INR {net_amt}, Bank Credit = INR {bank_amt}, Variance = INR {amt_diff}\n"
        f"- Date Settlement Gap: {date_gap} days\n"
        f"- Reference String Similarity: {ref_sim}\n"
        f"- Bank Narration: {bank_nar}\n"
    )
    return text

def index_reconciliation_records(df_c1: pd.DataFrame, df_ml_exceptions: pd.DataFrame):
    """Indexes reconciliation results into ChromaDB collection."""
    collection = get_chroma_collection()
    
    documents = []
    metadatas = []
    ids = []
    
    ml_dict = df_ml_exceptions.set_index('ledger_txn_id').to_dict('index') if not df_ml_exceptions.empty else {}
    
    for idx, row in df_c1.iterrows():
        ledger_id = str(row['ledger_txn_id'])
        order_id = str(row['merchant_order_id'])
        
        row_dict = row.to_dict()
        if ledger_id in ml_dict:
            row_dict.update(ml_dict[ledger_id])
            
        doc_text = format_record_for_embedding(pd.Series(row_dict))
        
        documents.append(doc_text)
        metadatas.append({
            "ledger_txn_id": ledger_id,
            "merchant_order_id": order_id,
            "match_tier": str(row_dict.get('match_tier', '')),
            "status": str(row_dict.get('status', ''))
        })
        ids.append(f"doc_{ledger_id}_{idx}")
        
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    return len(documents)

def answer_settlement_query(user_query: str, groq_api_key: str = None) -> dict:
    """
    Retrieves context from ChromaDB and generates a strictly grounded, structured response using Groq LLM or deterministic fallback.
    """
    collection = get_chroma_collection()
    
    query_lower = user_query.lower()
    
    # Check if this is a general system architecture query
    is_system_query = any(k in query_lower for k in [
        'system about', 'what is the system', 'what is this system', 'about the system',
        'what does this system do', 'what is this copilot', 'explain system',
        'how does reconciliation work', 'overview of system', 'what is this platform',
        'what is this project', 'how does this work', 'system overview'
    ])
    
    # 1. Direct Order/Ledger ID Lookup
    orders = re.findall(r'ORD_\w+', user_query, re.IGNORECASE)
    ledgers = re.findall(r'LEDGER_\w+', user_query, re.IGNORECASE)
    
    retrieved_docs = []
    retrieved_meta = []
    
    if orders or ledgers:
        target_id = (orders[0] if orders else ledgers[0]).upper()
        try:
            where_clause = {"merchant_order_id": target_id} if orders else {"ledger_txn_id": target_id}
            direct_res = collection.get(where=where_clause)
            if direct_res and direct_res.get('documents') and len(direct_res['documents']) > 0:
                retrieved_docs = direct_res['documents']
                retrieved_meta = direct_res['metadatas']
        except Exception:
            pass
            
    if not retrieved_docs:
        # Semantic vector search
        results = collection.query(
            query_texts=[user_query],
            n_results=5
        )
        retrieved_docs = results['documents'][0] if results['documents'] else []
        retrieved_meta = results['metadatas'][0] if results['metadatas'] else []
    
    if is_system_query:
        context_str = SYSTEM_ARCHITECTURE_SUMMARY + "\n---\nSAMPLE RETRIEVED RECORDS:\n" + ("\n---\n".join(retrieved_docs[:2]) if retrieved_docs else "N/A")
    else:
        context_str = "\n---\n".join(retrieved_docs) if retrieved_docs else "No records found."
    
    prompt = (
        "You are an enterprise AI Finance Assistant for Razorpay settlement reconciliation.\n"
        "STRICT GROUNDING DIRECTIVES:\n"
        "1. Use ONLY information explicitly present in the retrieved context below.\n"
        "2. NEVER invent transaction IDs, order IDs, bank IDs, amounts, dates, confidence scores, reasons, or financial conclusions.\n"
        "3. Do NOT infer information that is not explicitly present in the retrieved context.\n"
        "4. If the retrieved context is insufficient to answer the query and it is NOT a general system overview question, respond ONLY with:\n"
        "   Insufficient evidence in the available reconciliation records to answer reliably.\n"
        "5. For general system overview questions, explain the platform components using the provided System Architecture Summary.\n"
        "6. Do NOT use unsupported claims such as 'safe', 'approved', 'compliant', 'audit-confirmed', or 'financially secure'.\n"
        "7. For numerical values, preserve exact values from the retrieved records.\n\n"
        "REQUIRED STRUCTURED OUTPUT FORMAT:\n"
        "For system overview or platform questions, format as:\n"
        "DATA SUMMARY\n"
        "- Component 1: Rule Engine (64.89% automated match rate)\n"
        "- Component 2: Random Forest ML Scorer for Tier 4 ambiguous cases\n"
        "- Component 3: ChromaDB Vector Store & Grounded RAG Copilot\n"
        "- Component 4: Monte Carlo Cash Forecaster (10,000 simulations)\n"
        "- Component 5: Honest Exception Queue (<0.70 confidence items)\n\n"
        "KEY FINDING\n"
        "[One or two concise sentences explaining what the system accomplishes.]\n\n"
        "EVIDENCE\n"
        "[Summarize core system modules and record coverage]\n\n"
        "For transaction or reconciliation specific questions, format as:\n"
        "RECONCILIATION EXCEPTION / RECONCILIATION SUMMARY\n\n"
        "Transaction Details:\n"
        "- Order ID: [exact value or N/A]\n"
        "- Ledger ID: [exact value or N/A]\n"
        "- Status: [exact value or N/A]\n"
        "- Match Tier: [exact value or N/A]\n"
        "- ML Confidence: [exact value or N/A]\n\n"
        "Why it was flagged / Why it reconciled:\n"
        "- [specific evidence from retrieved records]\n\n"
        "Evidence:\n"
        "- Amount difference: [exact value or N/A]\n"
        "- Reference similarity: [exact value or N/A]\n"
        "- Exact reference match: [True/False based on ref similarity]\n"
        "- Date gap: [exact value or N/A]\n"
        "- Other relevant retrieved fields: [exact values or N/A]\n\n"
        "Recommended Action:\n"
        "- [Only if explicitly supported by reconciliation context. Otherwise: Manual review recommended based on the available reconciliation status.]\n\n"
        f"USER QUERY: {user_query}\n\n"
        f"RETRIEVED CONTEXT:\n{context_str}\n"
    )
    
    answer = ""
    llm_used = "Settlement Copilot (Vector Fallback)"
    
    if groq_api_key or os.getenv("GROQ_API_KEY"):
        try:
            import groq
            key = groq_api_key or os.getenv("GROQ_API_KEY")
            client = groq.Groq(api_key=key)
            
            # Ultra-fast low latency Groq model priority: llama-3.3-70b-versatile -> llama-3.1-8b-instant -> qwen/qwen3.6-27b
            for model_candidate in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.6-27b", "mixtral-8x7b-32768"]:
                try:
                    completion = client.chat.completions.create(
                        model=model_candidate,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1
                    )
                    raw_content = completion.choices[0].message.content
                    answer = _strip_think_tags(raw_content)
                    if answer:
                        llm_used = "AI Finance Assistant"
                        break
                except Exception as model_err:
                    continue
            if not answer:
                answer = _generate_fallback_answer(retrieved_docs, user_query, is_system_query)
        except Exception as e:
            answer = _generate_fallback_answer(retrieved_docs, user_query, is_system_query)
    else:
        answer = _generate_fallback_answer(retrieved_docs, user_query, is_system_query)
        
    return {
        "user_query": user_query,
        "retrieved_context": retrieved_docs,
        "retrieved_metadata": retrieved_meta,
        "answer": answer,
        "llm_engine": llm_used
    }

def _generate_fallback_answer(retrieved_docs: list, user_query: str, is_system_query: bool = False) -> str:
    if is_system_query:
        return (
            "DATA SUMMARY\n"
            "- Component 1: Rule Engine (64.89% automated match rate across Tiers 1-3)\n"
            "- Component 2: Random Forest ML Scorer for Tier 4 ambiguous cases\n"
            "- Component 3: ChromaDB Vector Store & Grounded RAG Copilot\n"
            "- Component 4: Monte Carlo Cash Forecaster (10,000 simulations)\n"
            "- Component 5: Honest Exception Queue (<0.70 confidence items)\n\n"
            "KEY FINDING\n"
            "The system is an enterprise AI Finance Controller Copilot that automates bank-to-ledger reconciliation, predicts match confidence, forecasts 30-day cash flow, and flags exception items for mandatory review.\n\n"
            "EVIDENCE\n"
            "- Pipeline handles 225 batch settlement records with automated rule matching, Random Forest scoring, ChromaDB indexing, and Monte Carlo cash forecasting."
        )
        
    if not retrieved_docs:
        return (
            "INSUFFICIENT EVIDENCE\n\n"
            "Insufficient evidence in the available reconciliation records to answer reliably."
        )
    
    top_doc = retrieved_docs[0]
    
    order_id = "N/A"
    ledger_id = "N/A"
    status = "N/A"
    tier = "N/A"
    conf = "N/A"
    amt_diff = "N/A"
    ref_sim = "N/A"
    date_gap = "N/A"
    
    for line in top_doc.split('\n'):
        if 'Merchant Order ID:' in line:
            order_id = line.split('Merchant Order ID:')[1].strip()
        elif 'Ledger Txn ID:' in line:
            parts = line.split('|')
            ledger_id = parts[0].replace('- Ledger Txn ID:', '').strip()
        elif 'Match Tier:' in line:
            parts = line.split('|')
            tier = parts[0].replace('- Match Tier:', '').strip()
            if len(parts) > 1 and 'Status:' in parts[1]:
                status = parts[1].replace('Status:', '').strip()
        elif 'Confidence Score:' in line:
            conf = line.split('Confidence Score:')[1].strip()
        elif 'Variance = INR' in line:
            amt_diff = line.split('Variance = INR')[1].strip()
        elif 'Date Settlement Gap:' in line:
            date_gap = line.split('Date Settlement Gap:')[1].strip()
        elif 'Reference String Similarity:' in line:
            ref_sim = line.split('Reference String Similarity:')[1].strip()

    is_dataset_summary = any(w in user_query.lower() for w in ['summary', 'overview', 'dataset', 'batch statistics', 'all records', 'reconciliation summary'])
    has_order_or_ledger = bool(re.search(r'(ORD_\w+|LEDGER_\w+)', user_query, re.IGNORECASE))
    
    if is_dataset_summary:
        return (
            "DATA SUMMARY\n"
            f"- Records analyzed: {len(retrieved_docs)}\n"
            f"- Reconciled: Status values in retrieved records ({status})\n"
            f"- Unresolved: Records matching query search\n"
            f"- Match tier(s): {tier}\n"
            f"- Relevant variance: INR {amt_diff}\n\n"
            "KEY FINDING\n"
            f"Retrieved {len(retrieved_docs)} settlement record(s) matching your query from ChromaDB vector storage.\n\n"
            "EVIDENCE\n"
            f"- Ledger ID: {ledger_id}\n"
            f"- Order ID: {order_id}"
        )
    elif has_order_or_ledger or (order_id != "N/A" and order_id in user_query.upper()):
        if "UNMATCHED" in status or "TIER_4" in tier:
            why_text = f"Record was flagged as UNMATCHED ({tier}) because exact reference match was False (ref similarity: {ref_sim}) and date settlement gap was {date_gap} day(s), despite amount difference being INR {amt_diff}."
        else:
            why_text = f"Record reconciled under {tier} with status {status}."

        return (
            "RECONCILIATION EXCEPTION\n\n"
            "Transaction Details:\n"
            f"- Order ID: {order_id}\n"
            f"- Ledger ID: {ledger_id}\n"
            f"- Status: {status}\n"
            f"- Match Tier: {tier}\n"
            f"- ML Confidence: {conf}\n\n"
            "Why it was flagged / Why it reconciled:\n"
            f"- {why_text}\n\n"
            "Evidence:\n"
            f"- Amount difference: INR {amt_diff}\n"
            f"- Reference similarity: {ref_sim}\n"
            f"- Exact reference match: {ref_sim == '1.0'}\n"
            f"- Date gap: {date_gap} days\n\n"
            "Recommended Action:\n"
            "Manual review recommended based on the available reconciliation status."
        )
    else:
        return (
            "INSUFFICIENT EVIDENCE\n\n"
            "Insufficient evidence in the available reconciliation records to answer reliably."
        )
