from duckduckgo_search import DDGS

print("Testing DDGS directly...")
try:
    with DDGS() as ddgs:
        results = list(ddgs.text("Sigmoid Function", max_results=3))
        print(f"Results type: {type(results)}")
        print(f"Results count: {len(results)}")
        print(f"First result: {results[0] if results else 'None'}")
except Exception as e:
    print(f"Error: {e}")
