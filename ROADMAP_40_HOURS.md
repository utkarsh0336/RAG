# 40-Hour Intensive Learning Roadmap
## Master the Self-Correcting RAG System in 8 Days

**Total Time:** 40 hours (8 days × 5 hours/day OR 2 weeks × 20 hours/week)

**Focus:** Hands-on understanding over theory. Learn by doing.

---

## 📅 Day 1: Foundations (5 hours)

### Morning Session (3 hours)

**Hour 1: LLM Basics**
- ⏱️ Watch: [Andrej Karpathy - Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1 hour)
- 📝 Key takeaways: How LLMs work, context windows, temperature

**Hour 2: RAG Fundamentals**
- ⏱️ Watch: [RAG from Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) by LangChain (30 min)
- 📖 Read: [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/) (30 min)
- 📝 Understand: Retrieve → Augment → Generate pipeline

**Hour 3: Hands-On - Your First RAG**
```python
# Install
pip install langchain openai chromadb

# Build tiny RAG
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

# 1. Create embeddings
docs = ["Alan Turing invented computers", "Python is a programming language"]
embeddings = OpenAIEmbeddings()
db = Chroma.from_texts(docs, embeddings)

# 2. Retrieve
results = db.similarity_search("Who invented computers?")

# 3. Generate
llm = ChatOpenAI()
answer = llm.invoke(f"Context: {results[0]}\nQuestion: Who invented computers?")
print(answer)
```

### Afternoon Session (2 hours)

**Hour 4: Vector Embeddings Deep Dive**
- 📖 Read: [Understanding Vector Embeddings](https://www.deeplearning.ai/the-batch/how-vector-similarity-search-works/) (30 min)
- 💻 Hands-On: (30 min)
  ```python
  from sentence_transformers import SentenceTransformer
  import numpy as np
  
  model = SentenceTransformer('all-MiniLM-L6-v2')
  
  # Embed sentences
  sentences = ["I love pizza", "Pizza is delicious", "Cars are fast"]
  embeddings = model.encode(sentences)
  
  # Calculate similarity
  from sklearn.metrics.pairwise import cosine_similarity
  print(cosine_similarity([embeddings[0]], [embeddings[1]]))  # High
  print(cosine_similarity([embeddings[0]], [embeddings[2]]))  # Low
  ```
- 📂 Review: `src/rag/qdrant_handler.py` (30 min)

**Hour 5: Project Setup**
- 💻 Clone repo, run `docker-compose up`
- 🔧 Ingest sample data: `python scripts/ingest_wiki.py`
- 🌐 Access UI: http://localhost:8501
- ✅ Test 3 queries, observe outputs

**✅ Day 1 Checkpoint:** Understand basic RAG pipeline and embeddings

---

## 📅 Day 2: LangChain & LangGraph (5 hours)

### Morning Session (3 hours)

**Hour 1: LangChain Quickstart**
- 📖 Read: [LangChain Getting Started](https://python.langchain.com/docs/get_started/introduction) (30 min)
- 💻 Tutorial: Follow [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) (30 min)

**Hour 2-3: LangGraph Core Concepts**
- 📹 Watch: [LangGraph Tutorial](https://www.youtube.com/watch?v=9BPCV5TYPmg) (45 min)
- 📖 Read: [LangGraph Docs - Introduction](https://langchain-ai.github.io/langgraph/tutorials/introduction/) (45 min)
- 💻 Build Simple Graph: (30 min)
  ```python
  from langgraph.graph import StateGraph, END
  from typing import TypedDict
  
  class State(TypedDict):
      question: str
      answer: str
  
  def retrieve(state):
      return {"answer": f"Retrieved info for {state['question']}"}
  
  def generate(state):
      return {"answer": f"Final answer: {state['answer']}"}
  
  graph = StateGraph(State)
  graph.add_node("retrieve", retrieve)
  graph.add_node("generate", generate)
  graph.set_entry_point("retrieve")
  graph.add_edge("retrieve", "generate")
  graph.add_edge("generate", END)
  
  app = graph.compile()
  result = app.invoke({"question": "What is AI?"})
  print(result)
  ```

### Afternoon Session (2 hours)

**Hour 4: Study Project Graph**
- 📂 **DEEP DIVE:** `src/agents/graph.py` (2 hours)
  - Line-by-line understanding
  - Draw state diagram on paper:
    ```
    retrieve → generate → validate
                             ↓
                      [check_validation]
                        ↓           ↓
                   accepted      needs_work
                      ↓              ↓
                     END         execute → synthesize → END
    ```
  - Understand each node function
  - Trace one query through entire graph

**Hour 5: Implement Your Own**
- 💻 Create `my_simple_graph.py`:
  - 3 nodes: search → validate → answer
  - Conditional edge based on validation score
  - Test with sample question

**✅ Day 2 Checkpoint:** Understand LangGraph workflow and state management

---

## 📅 Day 3: Multi-Agent Systems (5 hours)

### Morning Session (3 hours)

**Hour 1: Agent Concepts**
- 📖 Read: [ReAct Paper Summary](https://arxiv.org/abs/2210.03629) (30 min)
- 📹 Watch: [Multi-Agent Systems](https://www.youtube.com/watch?v=DWUdGhRrv2c) (30 min)

**Hour 2-3: Study Project Agents**
- 📂 `src/agents/validation.py` (1 hour)
  - How validation works
  - Gap identification
  - Score calculation
  - Test: Run `python verify_validation.py`

- 📂 `src/agents/execution.py` (30 min)
  - Tool calling
  - Web search integration
  - ArXiv search

- 📂 `src/agents/synthesis.py` (30 min)
  - Multi-source combination
  - Citation generation

### Afternoon Session (2 hours)

**Hour 4: Build Validation Agent**
- 💻 Create `my_validator.py`:
  ```python
  from langchain.chat_models import ChatOpenAI
  
  class SimpleValidator:
      def __init__(self):
          self.llm = ChatOpenAI(temperature=0.2)
      
      def validate(self, question, answer):
          prompt = f"""
          Question: {question}
          Answer: {answer}
          
          Is this answer complete? Score 0-1.
          Output: {{"score": 0.8, "gaps": ["missing detail"]}}
          """
          result = self.llm.invoke(prompt)
          return result
  
  # Test it
  validator = SimpleValidator()
  result = validator.validate("What is AI?", "AI is artificial intelligence")
  print(result)
  ```

**Hour 5: Tool Integration**
- 📂 Study: `src/agents/tools.py`
- 💻 Test web search tool independently
- 💻 Add custom tool (e.g., Wikipedia search)

**✅ Day 3 Checkpoint:** Understand self-correction and tool use

---

## 📅 Day 4: Retrieval & Re-ranking (5 hours)

### Morning Session (3 hours)

**Hour 1: Vector Databases**
- 📹 Watch: [Vector Databases Explained](https://www.youtube.com/watch?v=dN0lsF2cvm4) (20 min)
- 📖 Read: [Qdrant Quick Start](https://qdrant.tech/documentation/quick-start/) (40 min)
- 🌐 Explore: Qdrant UI at http://localhost:6333/dashboard

**Hour 2: Bi-encoder vs Cross-encoder**
- 📖 Read: [Cross-Encoders Tutorial](https://www.sbert.net/examples/applications/cross-encoder/README.html) (30 min)
- 📂 Study: `crossencoder_justification.md` (15 min)
- 💻 Hands-On: (15 min)
  ```python
  from sentence_transformers import CrossEncoder
  
  model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  
  query = "What is machine learning?"
  docs = [
      "ML is a subset of AI",
      "Pizza is delicious",
      "Machine learning algorithms learn patterns"
  ]
  
  scores = model.predict([(query, doc) for doc in docs])
  print(scores)  # See which doc scores highest
  ```

**Hour 3: Two-Stage Retrieval**
- 📂 **DEEP DIVE:** `src/rag/retrieval.py` (1 hour)
  - Understand `MultiSourceRetriever`
  - Dense retrieval (gets 20 docs)
  - CrossEncoder re-ranks to top 10
  - Trace through code with debugger

### Afternoon Session (2 hours)

**Hour 4-5: Implement Your Own Retriever**
- 💻 Create `my_retriever.py`:
  ```python
  from qdrant_client import QdrantClient
  from sentence_transformers import SentenceTransformer, CrossEncoder
  
  class TwoStageRetriever:
      def __init__(self):
          self.client = QdrantClient("localhost", port=6333)
          self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
          self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
      
      def search(self, query, k=5):
          # Stage 1: Dense retrieval (20 docs)
          query_vector = self.encoder.encode(query)
          results = self.client.search(
              collection_name="wiki_rag",
              query_vector=query_vector,
              limit=20
          )
          
          # Stage 2: Re-rank to top k
          pairs = [(query, doc.payload['text']) for doc in results]
          scores = self.reranker.predict(pairs)
          
          # Sort by score and take top k
          ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
          return [doc for doc, score in ranked[:k]]
  
  # Test
  retriever = TwoStageRetriever()
  results = retriever.search("Who invented computers?")
  for r in results:
      print(r.payload['text'][:100])
  ```

**✅ Day 4 Checkpoint:** Understand two-stage retrieval and re-ranking

---

## 📅 Day 5: Caching & Performance (5 hours)

### Morning Session (3 hours)

**Hour 1: Redis Crash Course**
- 📹 Watch: [Redis in 100 Seconds](https://www.youtube.com/watch?v=G1rOthIU-uo) (2 min)
- 📹 Watch: [Redis Crash Course](https://www.youtube.com/watch?v=jgpVdJB2sKQ) (30 min)
- 💻 Hands-On: (30 min)
  ```bash
  # Install Redis
  docker run -d -p 6379:6379 redis
  
  # Use redis-cli
  docker exec -it <container> redis-cli
  
  # Try commands
  SET mykey "Hello"
  GET mykey
  SETEX tempkey 10 "Expires in 10s"
  TTL tempkey
  ```

**Hour 2-3: Caching Strategies**
- 📖 Read: [Caching Best Practices](https://redis.com/blog/cache-invalidation-strategies/) (30 min)
- 📂 **DEEP DIVE:** `src/cache/redis_cache.py` (1.5 hours)
  - Two-tier architecture
  - Tier 1: Answer caching (1hr TTL)
  - Tier 2: Vector caching (24hr TTL)
  - Different TTLs for web-sourced data
  - Test each method

### Afternoon Session (2 hours)

**Hour 4-5: Implement Your Cache**
- 💻 Create `my_cache.py`:
  ```python
  import redis
  import json
  from datetime import timedelta
  
  class SimpleCache:
      def __init__(self):
          self.client = redis.Redis(host='localhost', port=6379, db=0)
      
      def set_answer(self, question, answer, ttl=3600):
          key = f"answer:{question}"
          self.client.setex(key, ttl, answer)
      
      def get_answer(self, question):
          key = f"answer:{question}"
          result = self.client.get(key)
          return result.decode() if result else None
      
      def cache_stats(self):
          # Get all answer keys
          keys = self.client.keys("answer:*")
          return {
              "total_cached": len(keys),
              "cache_size": sum(self.client.memory_usage(k) for k in keys)
          }
  
  # Test
  cache = SimpleCache()
  cache.set_answer("What is AI?", "Artificial Intelligence", ttl=60)
  print(cache.get_answer("What is AI?"))  # Cache hit
  print(cache.cache_stats())
  ```

- 💻 Measure performance:
  - Query without cache: `time python run_query.py`
  - Query with cache: `time python run_query.py` (2nd run)
  - Compare execution times

**✅ Day 5 Checkpoint:** Understand caching and performance optimization

---

## 📅 Day 6: Observability & Evaluation (5 hours)

### Morning Session (3 hours)

**Hour 1: Logging Fundamentals**
- 📖 Read: [Python Logging Guide](https://realpython.com/python-logging/) (30 min)
- 📂 Study: `src/observability/tracker.py` (30 min)
  - JSON logging format
  - Event types: retrieval, generation, validation, tool_call, synthesis
  - Metric collection

**Hour 2: Implement Basic Tracker**
- 💻 Create `my_tracker.py`:
  ```python
  import json
  import time
  from datetime import datetime
  
  class QueryTracker:
      def __init__(self):
          self.events = []
          self.start_time = None
      
      def start(self, question):
          self.start_time = time.time()
          self.events.append({
              "type": "start",
              "question": question,
              "timestamp": datetime.now().isoformat()
          })
      
      def log_retrieval(self, docs, scores):
          self.events.append({
              "type": "retrieval",
              "num_docs": len(docs),
              "top_score": max(scores),
              "timestamp": datetime.now().isoformat()
          })
      
      def log_generation(self, answer, tokens):
          self.events.append({
              "type": "generation",
              "answer_length": len(answer),
              "tokens": tokens,
              "timestamp": datetime.now().isoformat()
          })
      
      def finish(self):
          duration = time.time() - self.start_time
          self.events.append({
              "type": "end",
              "duration_seconds": duration,
              "timestamp": datetime.now().isoformat()
          })
          return self.events
  
  # Test
  tracker = QueryTracker()
  tracker.start("What is AI?")
  tracker.log_retrieval(["doc1", "doc2"], [0.9, 0.7])
  tracker.log_generation("AI is artificial intelligence", 500)
  events = tracker.finish()
  
  # Save to JSON
  with open('logs/test_run.json', 'w') as f:
      json.dump(events, f, indent=2)
  ```

**Hour 3: Evaluation Metrics**
- 📖 Read: [RAGAS: Automated Evaluation](https://docs.ragas.io/en/stable/concepts/metrics/index.html) (30 min)
- 📂 Study: `src/evaluation/metrics.py` (30 min)
  - Faithfulness (hallucination detection)
  - Relevance (answers the question)
  - Citation accuracy

### Afternoon Session (2 hours)

**Hour 4-5: Run Evaluation Pipeline**
- 📂 Study: `src/evaluation/dataset.py` (30 min)
- 💻 Run evaluation:
  ```bash
  python run_evaluation.py --sample-size 3
  ```
- 📊 Analyze results in `evaluation_reports/`
- 📝 Understand each metric:
  - What makes an answer faithful?
  - What makes it relevant?
  - How are citations validated?

**✅ Day 6 Checkpoint:** Understand observability and evaluation

---

## 📅 Day 7: Docker & Deployment (5 hours)

### Morning Session (3 hours)

**Hour 1: Docker Basics**
- 📹 Watch: [Docker in 100 Seconds](https://www.youtube.com/watch?v=Gjnup-PuquQ) (2 min)
- 📹 Watch: [Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo) (First 1 hour)

**Hour 2: Dockerfile Deep Dive**
- 📂 Study: `Dockerfile` (30 min)
  - Multi-stage build
  - Layer caching
  - Health checks
  - Non-root user
- 💻 Build image:
  ```bash
  docker build -t rag-app .
  docker images  # See image size
  ```

**Hour 3: docker-compose**
- 📖 Read: [docker-compose Overview](https://docs.docker.com/compose/) (30 min)
- 📂 **DEEP DIVE:** `docker-compose.yml` (30 min)
  - 3 services: app, qdrant, redis
  - Networks and volumes
  - Health checks
  - Environment variables
  - Depends_on conditions

### Afternoon Session (2 hours)

**Hour 4-5: Deploy & Test**
- 💻 Full deployment:
  ```bash
  # Build and start all services
  docker-compose up --build
  
  # In another terminal, check status
  docker-compose ps
  
  # View logs
  docker-compose logs app
  docker-compose logs qdrant
  docker-compose logs redis
  
  # Test the system
  curl http://localhost:8501
  
  # Ingest data into containerized system
  docker-compose exec app python scripts/ingest_wiki.py
  
  # Test query via UI
  # Go to http://localhost:8501
  
  # Shutdown
  docker-compose down
  
  # With volume cleanup
  docker-compose down -v
  ```

- 📝 Document:
  - How to start system
  - How to stop system
  - How to view logs
  - How to rebuild

**✅ Day 7 Checkpoint:** Can deploy full system with Docker

---

## 📅 Day 8: Integration & Mastery (5 hours)

### Morning Session (3 hours)

**Hour 1: End-to-End Trace**
- 💻 Run query with full observability:
  ```bash
  # Start system
  docker-compose up -d
  
  # Submit complex query via UI
  "Which AI models released after 2020 have more than 100B parameters?"
  
  # Watch logs in real-time
  docker-compose logs -f app
  
  # Examine JSON log
  cat logs/run_*.json | jq .
  ```

- 📝 Trace the flow:
  1. Cache check (miss)
  2. Vector retrieval (20 docs)
  3. CrossEncoder re-rank (10 docs)
  4. Initial generation
  5. Validation (gap found)
  6. Tool execution (web search)
  7. Synthesis
  8. Cache store

**Hour 2: Performance Analysis**
- 📊 Collect metrics:
  ```python
  import json
  from pathlib import Path
  
  # Analyze all logs
  logs = Path('logs').glob('run_*.json')
  
  total_time = []
  cache_hits = 0
  total_tokens = 0
  
  for log_file in logs:
      with open(log_file) as f:
          data = json.load(f)
          total_time.append(data.get('total_time', 0))
          if data.get('cache_hit'):
              cache_hits += 1
          total_tokens += data.get('total_tokens', 0)
  
  print(f"Avg response time: {sum(total_time)/len(total_time):.2f}s")
  print(f"Cache hit rate: {cache_hits/len(list(logs)):.2%}")
  print(f"Avg tokens: {total_tokens/len(list(logs)):.0f}")
  ```

**Hour 3: System Optimization**
- 🔧 Experiment:
  - Change `k` in retrieval (5 vs 10 docs)
  - Adjust CrossEncoder threshold
  - Tune cache TTLs
  - Modify temperature
- 📊 Measure impact on metrics

### Afternoon Session (2 hours)

**Hour 4: Build Something New**
- 💻 Choose one enhancement:

**Option A: Add Conversational Memory**
```python
class ConversationalRAG:
    def __init__(self):
        self.graph = RAGGraph()
        self.history = []
    
    def chat(self, question):
        # Add history to context
        context = "\n".join(self.history[-3:])  # Last 3 turns
        full_question = f"Context: {context}\nQuestion: {question}"
        
        result = self.graph.run(full_question)
        self.history.append(f"Q: {question}\nA: {result['final_answer']}")
        
        return result
```

**Option B: Add Query Expansion**
```python
def expand_query(original_query):
    llm = ChatOpenAI()
    prompt = f"Generate 3 alternative phrasings of: {original_query}"
    variations = llm.invoke(prompt)
    return [original_query] + variations.split('\n')

# Use all variations for retrieval
```

**Option C: Add Feedback Loop**
```python
def collect_feedback(question, answer, helpful: bool):
    with open('feedback.jsonl', 'a') as f:
        json.dump({
            "question": question,
            "answer": answer,
            "helpful": helpful,
            "timestamp": datetime.now().isoformat()
        }, f)
        f.write('\n')
```

**Hour 5: Documentation & Presentation**
- 📝 Create project presentation:
  - System architecture diagram
  - Key innovations (self-correction, caching, re-ranking)
  - Performance metrics
  - Demo screenshots
  - What you learned
  - Future improvements

**✅ Day 8 Checkpoint:** Full system mastery!

---

## 📊 Final Assessment (Self-Check)

After 40 hours, you should be able to:

✅ **Explain** how RAG works end-to-end  
✅ **Describe** the difference between bi-encoder and cross-encoder  
✅ **Implement** a basic multi-agent workflow with LangGraph  
✅ **Set up** vector database and perform similarity search  
✅ **Design** a two-tier caching strategy  
✅ **Debug** the system using logs and traces  
✅ **Deploy** the full stack with Docker  
✅ **Evaluate** RAG output quality  
✅ **Optimize** for performance (latency, cost)  
✅ **Modify** and extend the codebase  

---

## 🎯 Critical Files Mastery Order

Study these files in this exact order:

1. **Day 1:** `src/rag/qdrant_handler.py` - Vector DB basics
2. **Day 2:** `src/agents/graph.py` - Complete workflow ⭐
3. **Day 3:** `src/agents/validation.py` - Self-correction
4. **Day 4:** `src/rag/retrieval.py` - Two-stage retrieval
5. **Day 5:** `src/cache/redis_cache.py` - Caching
6. **Day 6:** `src/observability/tracker.py` - Logging
7. **Day 7:** `Dockerfile` + `docker-compose.yml` - Deployment
8. **Day 8:** Review all + build extension

---

## 💡 Time Management Tips

**Daily Structure (5 hours):**
- 🕐 Hours 1-2: Watch videos + read docs (input)
- 🕑 Hours 3-4: Hands-on coding (practice)
- 🕔 Hour 5: Review project code (application)

**Focus Blocks:**
- Use Pomodoro: 25 min focus + 5 min break
- Take notes in Markdown
- Run code as you read it
- Test everything yourself

**Avoid:**
- Don't just read - code along!
- Don't skip hands-on exercises
- Don't get stuck on theory (learn pragmatically)

---

## 🚀 Post-40 Hours

After completing this roadmap:

**Week 3 (Optional extras):**
- Read research papers
- Contribute to LangChain/LangGraph
- Build your own RAG for different domain
- Deploy to cloud (AWS/GCP)

**Continuous Learning:**
- Follow r/LocalLLaMA
- Join LangChain Discord
- Watch new LLM research
- Build side projects

---

## 📚 Resource Pack (Bookmark These)

**Videos (Total: ~6 hours)**
- Andrej Karpathy LLM intro (1h)
- LangGraph tutorial (45m)
- RAG from scratch (30m)
- Docker tutorial (1h)
- Redis crash course (30m)

**Docs (Reference)**
- LangChain docs
- LangGraph docs
- Qdrant docs
- Redis docs

**Tools**
- VS Code with Python extension
- Docker Desktop
- Qdrant UI (localhost:6333)
- Redis Insight (GUI for Redis)

---

## ✅ Daily Checklist Template

Use this for each day:

**Day X: [Topic]**
- [ ] Watched tutorial video(s)
- [ ] Read documentation
- [ ] Completed hands-on exercise
- [ ] Reviewed project file
- [ ] Built something from scratch
- [ ] Tested and debugged
- [ ] Documented learnings
- [ ] Ready for next day

---

**Good luck with your intensive 40-hour sprint! 🚀**

Remember: **Understanding > Memorization**. Focus on grasping concepts deeply rather than rushing through material.
