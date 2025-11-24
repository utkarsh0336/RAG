from src.agents.tools import web_search_tool
import os

# Ensure no API keys are set to force fallback
if "TAVILY_API_KEY" in os.environ:
    del os.environ["TAVILY_API_KEY"]
if "SERPER_API_KEY" in os.environ:
    del os.environ["SERPER_API_KEY"]

print("Testing web_search_tool with DuckDuckGo fallback...")
try:
    result = web_search_tool.invoke("Sigmoid Function definition")
    print(f"Result: {result[:200]}...")
except Exception as e:
    print(f"Error: {e}")
