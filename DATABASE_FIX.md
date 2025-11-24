# Quick Fix for Database Setup

## Problem
The model fails to fetch data because the databases don't exist yet.

## Solution: Run these commands

### 1. Create SQL Database ✅ (Already done)
```powershell
python scripts/setup_sql.py
```

### 2. Create Qdrant Vector Database (LOCAL, no Docker needed)
Set environment variable first:
```powershell
$env:QDRANT_PATH="./qdrant_data"
python scripts/ingest_wiki.py
```

OR edit `scripts/ingest_wiki.py` line 29 to:
```python
qdrant = QdrantHandler(path="./qdrant_data")
```

Then run:
```powershell
python scripts/ingest_wiki.py
```

### 3. Also update `src/rag/retrieval.py`
Line 15 should be:
```python
self.qdrant = QdrantHandler(path=path or "./qdrant_data")
```

This ensures the RAG pipeline also uses local Qdrant.

## Why This Happened
- Qdrant server (Docker) isn't running
- Code defaulted to connecting to `http://localhost:6333`
- Local fallback wasn't enabled by default

## Quickest Solution
Run this single command to set up everything:
```powershell
$env:QDRANT_PATH="./qdrant_data"; python scripts/ingest_wiki.py
```

After ingestion completes (takes ~2-3 minutes), restart Streamlit:
```powershell
streamlit run src/ui/app.py
```
