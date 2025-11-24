import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model_name = "gemini-2.0-flash"
print(f"Testing {model_name}...")
try:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content("Hello, are you working?")
    print(f"SUCCESS! Response: {response.text}")
except Exception as e:
    print(f"FAILED: {e}")
