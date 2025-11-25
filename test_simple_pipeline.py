import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
load_dotenv()

# Test just the retrieval and generation without validation/execution
from src.rag.retrieval import MultiSourceRetriever
from src.rag.generation import AnswerGenerator

def test_simple_pipeline():
    print("Testing simple retrieval + generation...")
    
    # Test retrieval
    retriever = MultiSourceRetriever()
    question = "What circumstances led to Alan Turing's death?"
    print(f"\nQuery: {question}")
    
    docs = retriever.retrieve(question)
    print(f"\n✓ Retrieved {len(docs)} documents")
    
    if docs:
        for i, doc in enumerate(docs[:2]):
            print(f"\nDoc {i+1} preview: {doc.page_content[:150]}...")
        
        # Test generation
        context = "\n\n".join([d.page_content for d in docs])
        generator = AnswerGenerator()
        result = generator.generate(question, context)
        
        print(f"\n✓ Generated answer:")
        print(result.get('answer', 'No answer'))
    else:
        print("\n✗ No documents retrieved - VectorDB may still be locked")

if __name__ == "__main__":
    test_simple_pipeline()
