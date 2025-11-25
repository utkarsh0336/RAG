import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

from src.rag.retrieval import HybridRetriever

def debug_retrieval():
    print("Initializing HybridRetriever for 'wiki_rag'...")
    try:
        retriever = HybridRetriever("wiki_rag")
        if retriever.client:
            print("Client initialized successfully.")
            try:
                collections = retriever.client.get_collections()
                print(f"Available collections: {collections}")
            except Exception as e:
                print(f"Failed to list collections: {e}")
        else:
            print("Client failed to initialize.")
            return

        query = "Alan Turing"
        print(f"Testing search for query: '{query}'")
        results = retriever.search(query)
        
        if results:
            print(f"Successfully retrieved {len(results)} documents.")
            for i, doc in enumerate(results[:3]):
                print(f"Doc {i+1}: {doc.page_content[:100]}...")
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"An error occurred during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_retrieval()
