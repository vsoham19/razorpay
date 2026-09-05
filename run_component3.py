import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'app')))

from rag_engine import index_reconciliation_records, answer_settlement_query

def main():
    print("=" * 60)
    print(" COMPONENT 3: SETTLEMENT Q&A AGENT (RAG)")
    print("=" * 60)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    results_c1_csv = os.path.join(data_dir, 'matched_results_component1.csv')
    ml_exceptions_csv = os.path.join(data_dir, 'ml_scored_exceptions.csv')
    
    df_c1 = pd.read_csv(results_c1_csv)
    df_ml = pd.read_csv(ml_exceptions_csv) if os.path.exists(ml_exceptions_csv) else pd.DataFrame()
    
    # 1. Indexing records into ChromaDB
    print("\n[1] Indexing financial records into ChromaDB vector repository...")
    num_indexed = index_reconciliation_records(df_c1, df_ml)
    print(f" Successfully indexed {num_indexed} reconciliation documents into ChromaDB.")
    
    # 2. Test Q&A Queries
    test_queries = [
        "Why did transaction ORD_20260825_1185 fail to reconcile?",
        "Show me transactions with fee adjustments or MDR deductions",
        "Which transactions are in the honest exception list?"
    ]
    
    print("\n[2] Executing Natural Language Retrieval & Q&A Queries...")
    for q in test_queries:
        print("\n" + "=" * 50)
        print(f"QUERY: '{q}'")
        print("=" * 50)
        
        result = answer_settlement_query(q)
        print(f"Engine Used: {result['llm_engine']}")
        print("\nANSWER:")
        print(result['answer'])
        
if __name__ == "__main__":
    main()
