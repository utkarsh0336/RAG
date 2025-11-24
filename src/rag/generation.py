import os
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LLMClient:
    def __init__(self, model_name: str = "llama3", temperature: float = 0.0):
        # Connect to local Ollama instance
        # Default to localhost since we're running outside Docker
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=base_url
        )

class AnswerGenerator:
    def __init__(self):
        self.llm_client = LLMClient()
        self.llm = self.llm_client.llm
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert AI assistant. Answer the user's question based ONLY on the provided context.
        If the context does not contain enough information to answer the question fully, state what is missing.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """)
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def generate(self, question: str, context: str) -> str:
        return self.chain.invoke({"question": question, "context": context})
