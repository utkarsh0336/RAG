from src.agents.validation import ValidationAgent
import os
from dotenv import load_dotenv

load_dotenv()

agent = ValidationAgent()

question = "Who is the current Prime Minister of the UK?"
context = "The Prime Minister of the UK is Rishi Sunak."
answer = "The current Prime Minister of the UK is Rishi Sunak."

print("Testing Validation Agent...")
try:
    report = agent.validate(question, context, answer)
    print(f"Report: {report}")
    
    if report.get("is_outdated"):
        print("SUCCESS: Identified potentially outdated info.")
    else:
        print("NOTE: Did not flag as outdated (might be correct depending on LLM knowledge cutoff).")
        
    print(f"Reasoning: {report.get('reasoning')}")
except Exception as e:
    print(f"Error: {e}")
