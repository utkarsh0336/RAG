import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.agents.validation import ValidationAgent

def test_phase2():
    print("Testing Phase 2: Validation Agent...")
    
    agent = ValidationAgent(provider="gemini")
    
    # Scenario 1: Good Answer
    print("\n--- Scenario 1: Complete Answer ---")
    question = "What is the capital of France?"
    context = "Paris is the capital and most populous city of France."
    answer = "The capital of France is Paris."
    
    report = agent.validate(question, context, answer)
    print(json.dumps(report, indent=2))
    
    # Scenario 2: Incomplete/Gaps
    print("\n--- Scenario 2: Incomplete Answer ---")
    question = "Who is the CEO of OpenAI and when was it founded?"
    context = "OpenAI was founded in December 2015."
    answer = "OpenAI was founded in December 2015."
    
    report = agent.validate(question, context, answer)
    print(json.dumps(report, indent=2))
    
    # Scenario 3: Hallucination/Inconsistency
    print("\n--- Scenario 3: Hallucination ---")
    question = "What is the speed of light?"
    context = "The speed of light in vacuum is exactly 299,792,458 meters per second."
    answer = "The speed of light is 300,000 km/s, which is exactly 500,000 miles per hour."
    
    report = agent.validate(question, context, answer)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    test_phase2()
