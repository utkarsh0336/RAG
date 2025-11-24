import torch
import streamlit as st
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.graph import RAGGraph

st.set_page_config(page_title="Self-Correcting RAG", layout="wide")

st.title("Self-Correcting RAG System")
st.markdown("""
This system uses an agentic workflow to answer complex questions.
1. **Retrieve**: Fetches data from VectorDB.
2. **Generate**: Drafts an initial answer.
3. **Validate**: Critiques the answer for gaps/hallucinations.
4. **Execute**: autonomously gathers missing info (Web/SQL/ArXiv).
5. **Synthesize**: Combines everything into a final verified answer.
""")

# Initialize Graph
if "graph" not in st.session_state:
    st.session_state.graph = RAGGraph()

question = st.text_input("Ask a complex question:", "What are the parameter counts of Llama 2 and GPT-4, and how do they compare in architecture?")

if st.button("Run Query"):
    with st.spinner("Running Agentic Workflow..."):
        try:
            # Run the graph
            final_state = st.session_state.graph.run(question)
            
            # Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Trace")
                with st.expander("1. Retrieval", expanded=False):
                    st.write(final_state.get("context", "No context retrieved"))
                    
                with st.expander("2. Initial Answer", expanded=False):
                    st.write(final_state.get("initial_answer", "No initial answer"))
                    
                with st.expander("3. Validation Report", expanded=True):
                    st.json(final_state.get("validation_report", {}))
                    
                with st.expander("4. Execution (New Info)", expanded=True):
                    st.write(final_state.get("new_info", "No new info gathered"))
            
            with col2:
                st.subheader("Final Answer")
                st.success(final_state.get("final_answer", "Processing failed"))
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
