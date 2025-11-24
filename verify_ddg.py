try:
    import duckduckgo_search
    print(f"duckduckgo_search version: {duckduckgo_search.__version__}")
    from duckduckgo_search import DDGS
    print("DDGS imported successfully")
    
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
    res = search.invoke("test")
    print(f"Search result: {res[:50]}...")
except Exception as e:
    print(f"Error: {e}")
