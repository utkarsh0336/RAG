# 🚀 Quick Start Guide - Self-Correcting RAG System

## Prerequisites (One-time Setup)

### Required
- **Python 3.11** - [Download](https://www.python.org/downloads/)
- **Git** - [Download](https://git-scm.com/downloads)

### Optional (for better performance)
- **Redis** - For caching (system works without it)
- **Qdrant Server** - For vector DB (can use local files instead)

---

## Step 1: Install Python Dependencies

```powershell
# Navigate to project directory
cd c:\Users\hp\Desktop\RAG

# Create virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 2: Set Up Environment Variables

The `.env` file is already configured with defaults. You only need to add:

1. **Google Gemini API Key** (for LLM):
   - Get free key from: https://aistudio.google.com/app/apikey
   - Update `.env` line 8:
     ```
     GOOGLE_API_KEY=your_actual_api_key_here
     ```

2. **Optional Web Search APIs** (for better results):
   - Tavily: https://tavily.com/ (get free key)
   - Serper: https://serper.dev/ (get free key)

---

## Step 3: Initialize Databases

Run these commands **once** to populate the databases:

```powershell
# 1. Create SQL database (AI model metadata)
python scripts/setup_sql.py

# 2. Ingest Wikipedia content (takes ~2-3 minutes)
python scripts/ingest_wiki.py

# 3. (Optional) Ingest ArXiv papers
python scripts/ingest_arxiv.py
```

**Output you should see:**
```
Database created at data/ai_models.db with 10 records.
Collection 'wiki_rag' created.
Processing: Generative pre-trained transformer (Depth 0)
...
Ingestion complete!
```

---

## Step 4: Run the Application

```powershell
streamlit run src/ui/app.py
```

**The app will open at:** http://localhost:8501

---

## 🎯 Quick Test

Once the UI loads, try this question:

```
Who is Alan Turing and what is the Turing Test?
```

You should see:
1. ✅ System retrieves from Wikipedia database
2. ✅ Generates initial answer
3. ✅ Validates the answer
4. ✅ Returns final synthesized answer with citations

---

## 📁 Project Structure

```
RAG/
├── src/
│   ├── agents/       # Multi-agent system (Validation, Execution, Synthesis)
│   ├── rag/          # Core RAG components (Retrieval, Generation)
│   ├── cache/        # Redis caching (optional)
│   └── ui/           # Streamlit interface
├── scripts/          # Setup scripts
├── data/             # SQL database
├── qdrant_data/      # Local vector database
└── logs/             # Query logs
```

---

## 🔧 Troubleshooting

### Issue: "Redis cache disabled"
**Solution**: This is normal! The system works without Redis, just slower on repeat queries.
To enable caching:
```powershell
# Install Redis (if you want caching)
choco install redis-64
Start-Service Redis
```

### Issue: "Collection not found"
**Solution**: Run the ingestion script again:
```powershell
python scripts/ingest_wiki.py
```

### Issue: "GOOGLE_API_KEY not set"
**Solution**: 
1. Get free key from https://aistudio.google.com/app/apikey
2. Add to `.env` file (line 8)
3. Restart Streamlit

### Issue: Import errors
**Solution**: Make sure you're in the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 🎨 What the System Does

### Phase 1: Primary Retrieval
- Searches local Wikipedia vector database
- Uses hybrid search (dense + sparse vectors)
- Re-ranks results with CrossEncoder

### Phase 2: Validation
- AI agent critiques the initial answer
- Identifies gaps, outdated info, inconsistencies
- Generates search queries to fill gaps

### Phase 3: Execution (if needed)
- Web search tool (DuckDuckGo/Tavily/Serper)
- ArXiv research paper lookup
- SQL database queries for model metadata

### Phase 4: Synthesis
- Combines all sources
- Resolves contradictions
- Generates final answer with inline citations

---

## 💡 Usage Tips

1. **First run is slower** - Models need to download
2. **Subsequent runs are faster** - With Redis caching
3. **Check the "Chain of Thought" tab** - See the full pipeline execution
4. **Enable "Show Full Logs"** - For detailed debugging

---

## 📊 Example Questions to Try

```
1. What are transformers in AI and how do they work?
2. Compare GPT-3 and GPT-4 parameters (uses SQL database)
3. What is the latest research on large language models? (uses ArXiv)
4. Explain the Turing Test with historical context
```

---

## 🛑 Stopping the Application

Press `Ctrl+C` in the terminal running Streamlit

---

## ⚙️ Advanced Configuration

### Use Different LLM Provider
Edit `src/rag/generation.py` to switch between:
- Google Gemini (default)
- Open source LLMs (vLLM/TGI)

### Adjust Ingestion Depth
Edit `scripts/ingest_wiki.py`:
```python
MAX_DEPTH = 2  # Increase for more Wikipedia articles
MAX_PAGES = 50  # Increase for more content
```

### Change Cache TTL
Edit `src/cache/redis_cache.py`:
```python
self.ANSWER_TTL = 7200  # 2 hours instead of 1 hour
```

---

## 📝 Summary

**Minimum to run:**
1. `pip install -r requirements.txt`
2. Add `GOOGLE_API_KEY` to `.env`
3. `python scripts/setup_sql.py`
4. `python scripts/ingest_wiki.py`
5. `streamlit run src/ui/app.py`

**That's it!** 🎉
