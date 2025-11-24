import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import streamlit as st
from src.agents.graph import RAGGraph

st.set_page_config(page_title="Self-Correcting RAG", layout="wide")

st.title("🤖 Self-Correcting RAG System")
st.markdown("""
This system uses a multi-agent approach to answer complex questions by:
1. Retrieving relevant information from Wikipedia and ArXiv
2. Generating an initial answer
3. Self-critiquing and identifying gaps
4. Gathering additional information
5. Synthesizing a comprehensive final answer
""")

# Initialize the RAG graph
if "graph" not in st.session_state:
    try:
        st.session_state.graph = RAGGraph()
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        st.stop()

# User input
question = st.text_input("Ask a question:", "What are transformers in AI and how do they work?")

if st.button("Run Query"):
    if question:
        with st.spinner("Processing your question..."):
            try:
                result = st.session_state.graph.run(question)
                
                # Display final answer with ChatGPT-style streaming
                st.subheader("✨ Answer")
                final_answer = result.get("final_answer", "No answer generated")
                
                if final_answer != "No answer generated":
                    # Stream the answer character by character for ChatGPT-like UX
                    answer_placeholder = st.empty()
                    displayed_text = ""
                    
                    import time
                    for char in final_answer:
                        displayed_text += char
                        answer_placeholder.markdown(displayed_text + "▌")  # Blinking cursor effect
                        time.sleep(0.01)  # Small delay for smooth streaming
                    
                    # Final display without cursor
                    answer_placeholder.markdown(final_answer)
                else:
                    st.error("I couldn't generate an answer. Please try rephrasing your question.")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question")
