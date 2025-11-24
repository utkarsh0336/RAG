# Fix for ONNX Runtime DLL Error on Windows

## Error
```
ImportError: DLL load failed while importing onnxruntime_pybind11_state: 
A dynamic link library (DLL) initialization routine failed.
```

## Quick Fixes (Try in order)

### Solution 1: Downgrade ONNX Runtime (Recommended)
```powershell
pip uninstall onnxruntime -y
pip install onnxruntime==1.16.3
```

### Solution 2: Install Visual C++ Redistributables
Download and install:
https://aka.ms/vs/17/release/vc_redist.x64.exe

Then restart your terminal.

### Solution 3: Use CPU-only version
```powershell
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime==1.16.3
```

### Solution 4: Switch to torch backend (if using fastembed)
Modify code to avoid ONNX:
- Use `SentenceTransformer` only (already CPU-friendly)
- Temporarily disable sparse embeddings

## Temporary Workaround
If you need to proceed immediately, comment out sparse embedding code:

In `src/rag/retrieval.py`:
```python
# self.sparse_model = SparseTextEmbedding(...)
```

In `scripts/ingest_wiki.py`:
```python
# sparse_model = SparseTextEmbedding(...)
# sparse_embeddings = list(sparse_model.embed(documents))
# Pass None or [] for sparse_embeddings
```

## Recommended: Solution 1
Run this command:
```powershell
pip uninstall onnxruntime -y && pip install onnxruntime==1.16.3
```
