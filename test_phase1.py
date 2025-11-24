import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.rag.retrieval import HybridRetriever
from src.rag.generation import AnswerGenerator

def test_phase1():
    print("Testing Phase 1...")
    
    # 1. Retrieval
    print("\n--- Retrieval ---")
    retriever = HybridRetriever("wiki_rag")
    query = "What is a Generative Pre-trained Transformer?"
    
    try:
        docs = retriever.search(query, k=3)
        print(f"Retrieved {len(docs)} documents.")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1}: {doc.page_content[:100]}...")
    except Exception as e:
        print(f"Retrieval failed: {e}")
        docs = []

    if not docs:
        print("No documents retrieved. Skipping generation.")
        return

    # 2. Generation
    print("\n--- Generation ---")
    # Use 'opensource' provider (requires TGI/vLLM running, or set base_url to a public API if available)
    # For testing, we might need to mock or use a real endpoint.
    # I'll try to use the default 'opensource' provider which expects localhost:8000
    # If that fails, I'll fallback to 'gemini' just to prove the code works, 
    # but the requirement is Open Source LLM.
    
    # Let's assume the user has a local LLM running or we can't really test this part fully without it.
    # I will try to use the 'opensource' provider.
    
    generator = AnswerGenerator(provider="opensource")
    
    context = "\n\n".join([doc.page_content for doc in docs])
    
    try:
        response = generator.generate(query, context)
        print("\nGenerated Response:")
        print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"Generation failed: {e}")

if __name__ == "__main__":
    test_phase1()
