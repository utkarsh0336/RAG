import sys
import os
import traceback
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

from src.agents.graph import RAGGraph

def reproduce():
    print("Initializing RAG Graph...")
    try:
        graph = RAGGraph()
        question = "What datasets or benchmarks are commonly used for sentiment classification?"
        print(f"Running query: {question}")
        result = graph.run(question)
        print("Query completed successfully.")
        print("Final Answer:", result.get("final_answer"))
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    reproduce()
