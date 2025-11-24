import os
import google.generativeai as genai
from langchain_core.runnables import RunnableSerializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.llms import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from typing import Any, Dict, Optional
import json

class SimpleLLM(RunnableSerializable):
    """Simple wrapper for Google Gemini that works with LangChain."""
    
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.0
    _model: Any = None
    
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.0, **kwargs):
        super().__init__(model_name=model_name, temperature=temperature, **kwargs)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
    
    def invoke(self, input_data: Any, config: Dict = None) -> str:
        """Handle invoke from LangChain chains."""
        # Extract the actual prompt text
        if isinstance(input_data, str):
            prompt = input_data
        elif isinstance(input_data, dict):
            # Convert dict to string representation
            prompt = str(input_data)
        else:
            prompt = str(input_data)
        
        response = self._model.generate_content(prompt)
        return response.text

class LLMClient:
    def __init__(self, provider: str = "gemini", model_name: str = "gemini-2.0-flash", temperature: float = 0.7, base_url: Optional[str] = None):
        if provider == "gemini":
            self.llm = SimpleLLM(model_name=model_name, temperature=temperature)
        elif provider == "opensource":
            # Assumes OpenAI-compatible endpoint (vLLM, TGI, etc.)
            # If base_url is not provided, default to localhost
            base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
            api_key = os.getenv("LLM_API_KEY", "EMPTY")
            self.llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

class AnswerGenerator:
    def __init__(self, provider: str = "opensource"):
        self.llm_client = LLMClient(provider=provider, model_name="meta-llama/Meta-Llama-3-70B-Instruct") # Default to a strong open model
        self.llm = self.llm_client.llm
        
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert AI assistant. Answer the user's question based on the provided context.
        
        Context:
        {context}
        
        Question: {question}
        
        Instructions:
        1. Answer the question clearly and naturally based ONLY on the context.
        2. Provide a confidence score (0.0 to 1.0) reflecting how well the context supports your answer.
        3. Output your response in JSON format with keys: "answer" and "confidence".
        
        JSON Output:
        """)
        
        self.chain = self.prompt | self.llm | JsonOutputParser()
        
    def generate(self, question: str, context: str) -> Dict[str, Any]:
        try:
            return self.chain.invoke({"question": question, "context": context})
        except Exception as e:
            print(f"Generation failed: {e}")
            return {"answer": "Failed to generate answer.", "confidence": 0.0}

