import sys
import os
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

load_dotenv()

from src.agents.tools import web_search_tool

def debug_search():
    query = "Alan Turing criticism"
    print(f"Testing web search for query: '{query}'")
    
    try:
        result = web_search_tool.invoke(query)
        print("\nSearch Results:")
        print(result)
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    debug_search()
