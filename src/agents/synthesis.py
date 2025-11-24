from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.generation import LLMClient

class SynthesisAgent:
    def __init__(self):
        self.llm = LLMClient().llm
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are the final synthesizer in a RAG pipeline.
        
        Original Question: {question}
        
        Initial Answer (Draft): {initial_answer}
        
        Validation Issues: {validation_report}
        
        New Information Found: {new_info}
        
        Your task:
        1. Synthesize a FINAL, comprehensive answer that addresses the original question.
        2. Incorporate the new information to fix the gaps/inconsistencies identified.
        3. Resolve any conflicts between sources (prioritize recent web/paper info over old data).
        4. Provide CITATIONS for every key claim. Format: [Source: URL/Title].
        
        Final Answer:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def synthesize(self, question: str, initial_answer: str, validation_report: str, new_info: str) -> str:
        return self.chain.invoke({
            "question": question,
            "initial_answer": initial_answer,
            "validation_report": validation_report,
            "new_info": new_info
        })
