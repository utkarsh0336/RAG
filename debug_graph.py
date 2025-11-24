import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.agents.graph import RAGGraph

def debug_run():
    print("Initializing Graph...")
    try:
        graph = RAGGraph()
    except Exception as e:
        print(f"Failed to init graph: {e}")
        return

    question = "What is the latest iPhone model?"
    print(f"\nRunning query: {question}")
    
    try:
        # Run the graph
        result = graph.run(question)
        
        print("\n--- RESULT DEBUG ---")
        print(f"Keys in result: {result.keys()}")
        
        if "initial_answer" in result:
            print(f"\nInitial Answer: {result['initial_answer'][:100]}...")
            
        if "validation_report" in result:
            print(f"\nValidation Report: {result['validation_report']}")
            with open("debug_validation.txt", "w", encoding="utf-8") as f:
                f.write(str(result['validation_report']))
            
        if "new_info" in result:
            print(f"\nNew Info: {result['new_info'][:200]}...")
            with open("debug_new_info.txt", "w", encoding="utf-8") as f:
                f.write(result['new_info'])
        else:
            print("\nNew Info: NOT FOUND (Execution might have been skipped)")
            
        if "final_answer" in result:
            print(f"\nFinal Answer: {result['final_answer']}")
        else:
            print("\nFinal Answer: NOT FOUND")
            
    except Exception as e:
        print(f"Graph execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_run()
