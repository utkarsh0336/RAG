# Complete Learning Guide: Self-Correcting RAG System

A structured learning path to master every component of this project, from fundamentals to advanced concepts.

---

## 📚 Learning Path Overview

**Estimated Time:** 4-6 weeks (2-3 hours/day)

**Skill Levels:**
- 🟢 Beginner: Start here if new to the concept
- 🟡 Intermediate: Core project components
- 🔴 Advanced: Optimization and production concepts

---

## Phase 1: Foundational Knowledge (Week 1)

### 1.1 Python Advanced Concepts 🟡

**What to Learn:**
- Type hints and TypedDict
- Decorators and context managers
- Asynchronous programming (async/await)
- Error handling and logging

**Resources:**
- **Book:** "Fluent Python" by Luciano Ramalho (Chapters 5, 7, 18, 19)
- **Video:** [Python AsyncIO Complete Tutorial](https://www.youtube.com/watch?v=t5Bo1Je9EmE) (1 hour)
- **Practice:** Rewrite simple scripts using type hints and async

**Project Files to Review:**
- `src/agents/graph.py` - See TypedDict usage
- `src/cache/redis_cache.py` - Async operations
- `src/observability/tracker.py` - Logging patterns

---

### 1.2 Large Language Models Fundamentals 🟢

**What to Learn:**
- How LLMs work (transformers architecture basics)
- Tokenization and context windows
- Temperature, top-p sampling
- Prompt engineering principles
- System vs user vs assistant roles

**Resources:**
- **Article:** [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- **Video:** [Andrej Karpathy - Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1 hour)
- **Course:** [DeepLearning.AI - ChatGPT Prompt Engineering](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
- **Paper:** "Attention Is All You Need" (Vaswani et al., 2017) - Skim sections 1-3

**Project Files to Review:**
- `src/rag/generation.py` - LLM prompts and temperature settings
- `src/agents/validation.py` - Structured prompting for validation
- `src/agents/synthesis.py` - Multi-turn conversation patterns

**Hands-On:**
- Experiment with different prompts in ChatGPT/Claude
- Try different temperature values (0.0, 0.5, 1.0)
- Understand tokens: Use OpenAI's tokenizer tool

---

### 1.3 RAG (Retrieval-Augmented Generation) Basics 🟢

**What to Learn:**
- What is RAG and why it's useful
- Difference between RAG and fine-tuning
- Basic RAG pipeline: Retrieve → Augment → Generate
- Chunking strategies
- Embedding models

**Resources:**
- **Paper:** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- **Video:** [RAG from Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) by LangChain (30 min)
- **Article:** [What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/) (Pinecone)
- **Tutorial:** [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

**Project Files to Review:**
- `src/rag/retrieval.py` - Complete retrieval implementation
- `scripts/ingest_wiki.py` - Document chunking and ingestion

---

## Phase 2: Core Technologies (Week 2)

### 2.1 Vector Databases & Embeddings 🟡

**What to Learn:**
- What are embeddings (vector representations)
- Cosine similarity, dot product
- Vector databases (Qdrant, Pinecone, Weaviate)
- Approximate Nearest Neighbor (ANN) search
- HNSW algorithm basics

**Resources:**
- **Article:** [Understanding Vector Embeddings](https://www.deeplearning.ai/the-batch/how-vector-similarity-search-works/)
- **Video:** [Vector Databases Explained](https://www.youtube.com/watch?v=dN0lsF2cvm4) (20 min)
- **Docs:** [Qdrant Documentation](https://qdrant.tech/documentation/)
- **Tutorial:** [Sentence Transformers](https://www.sbert.net/) - Embedding library we use

**Project Files to Review:**
- `src/rag/qdrant_handler.py` - Vector DB operations
- `src/rag/retrieval.py` - Embedding and search

**Hands-On:**
- Install `sentence-transformers`
- Embed a few sentences and compute cosine similarity
- Explore Qdrant UI at `http://localhost:6333/dashboard`

---

### 2.2 LangChain & LangGraph 🟡

**What to Learn:**
- LangChain architecture (models, prompts, chains)
- LangGraph for agentic workflows
- State management in graphs
- Conditional edges and routing

**Resources:**
- **Docs:** [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- **Tutorial:** [LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- **Video:** [LangGraph Tutorial](https://www.youtube.com/watch?v=9BPCV5TYPmg) (45 min)
- **Course:** [DeepLearning.AI - Building Multi-Agent Systems](https://www.deeplearning.ai/short-courses/)

**Project Files to Review:**
- `src/agents/graph.py` - **CRITICAL FILE** - Complete LangGraph implementation
- Study the state flow: retrieve → generate → validate → execute → synthesize

**Hands-On:**
- Build a simple LangGraph with 2-3 nodes
- Implement conditional routing
- Visualize your graph with LangGraph's drawing tools

---

### 2.3 Redis & Caching Strategies 🟡

**What to Learn:**
- Redis data structures (strings, hashes)
- TTL (Time To Live) and expiration
- Cache invalidation strategies
- LRU (Least Recently Used) eviction
- Multi-tier caching

**Resources:**
- **Docs:** [Redis Documentation](https://redis.io/docs/)
- **Video:** [Redis Crash Course](https://www.youtube.com/watch?v=jgpVdJB2sKQ) (30 min)
- **Article:** [Caching Best Practices](https://redis.com/blog/cache-invalidation-strategies/)

**Project Files to Review:**
- `src/cache/redis_cache.py` - Two-tier caching implementation

**Hands-On:**
- Install Redis locally
- Use `redis-cli` to set/get keys with TTL
- Understand `SETEX`, `GET`, `DEL` commands

---

## Phase 3: Advanced RAG Concepts (Week 3)

### 3.1 Multi-Agent Systems 🔴

**What to Learn:**
- Agent roles and responsibilities
- Inter-agent communication
- State sharing between agents
- Tool use and function calling
- Self-correction patterns

**Resources:**
- **Paper:** "AutoGPT: An Autonomous GPT-4 Experiment" (Survey paper)
- **Article:** [Multi-Agent Systems with LangGraph](https://blog.langchain.dev/langgraph-multi-agent-workflows/)
- **Video:** [Building Autonomous Agents](https://www.youtube.com/watch?v=DWUdGhRrv2c) (1 hour)

**Project Files to Review:**
- `src/agents/validation.py` - Self-critique agent
- `src/agents/execution.py` - Tool-using agent
- `src/agents/synthesis.py` - Information synthesis

**Key Concepts in Our Project:**
1. **Validation Agent:** Self-corrects by critiquing initial answers
2. **Execution Agent:** Uses tools (web, ArXiv, SQL) autonomously
3. **Synthesis Agent:** Combines multi-source information

---

### 3.2 Information Retrieval & Re-ranking 🔴

**What to Learn:**
- Bi-encoder vs Cross-encoder architectures
- Two-stage retrieval (retrieve then re-rank)
- BM25 algorithm (keyword search)
- Hybrid search (dense + sparse)
- Evaluation metrics: MRR, NDCG, Recall@k

**Resources:**
- **Paper:** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers et al., 2019)
- **Tutorial:** [Cross-Encoders for Re-ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- **Article:** [Understanding Re-ranking in Search](https://medium.com/@sahin.samia/a-comprehensive-guide-to-re-ranking-in-information-retrieval-c1b3b7f0aa35)

**Project Files to Review:**
- `src/rag/retrieval.py` - Two-stage retrieval with CrossEncoder
- `crossencoder_justification.md` - Technical reasoning

**Hands-On:**
- Compare scores: bi-encoder vs cross-encoder for same query
- Measure retrieval time difference
- Understand the accuracy vs speed tradeoff

---

### 3.3 Tool Use & Function Calling 🟡

**What to Learn:**
- How LLMs can call external functions
- Tool schemas and descriptions
- Parameter extraction from natural language
- Error handling in tool calls

**Resources:**
- **Docs:** [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- **Tutorial:** [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- **Article:** [Building LLM-Powered Tools](https://www.anthropic.com/index/tool-use)

**Project Files to Review:**
- `src/agents/tools.py` - Tool definitions (web search, ArXiv, SQL)
- `src/agents/execution.py` - Tool orchestration

---

## Phase 4: Production & Optimization (Week 4)

### 4.1 Docker & Containerization 🟡

**What to Learn:**
- Dockerfile basics (FROM, RUN, COPY, CMD)
- Multi-stage builds
- docker-compose orchestration
- Volumes and networks
- Health checks

**Resources:**
- **Course:** [Docker for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo) (3 hours)
- **Docs:** [Docker Documentation](https://docs.docker.com/get-started/)
- **Tutorial:** [docker-compose Tutorial](https://docs.docker.com/compose/gettingstarted/)

**Project Files to Review:**
- `Dockerfile` - Production-ready container
- `docker-compose.yml` - Multi-service orchestration

**Hands-On:**
- Build the Docker image: `docker build -t rag-app .`
- Run with compose: `docker-compose up`
- Inspect running containers: `docker ps`

---

### 4.2 Observability & MLOps 🔴

**What to Learn:**
- Logging strategies (structured logging)
- Metrics collection (latency, tokens, cache hit rate)
- Tracing distributed systems
- LangSmith for LLM observability
- Evaluation frameworks

**Resources:**
- **Course:** [MLOps Fundamentals](https://www.deeplearning.ai/courses/mlops-specialization/)
- **Docs:** [LangSmith Documentation](https://docs.smith.langchain.com/)
- **Article:** [Observability for LLM Applications](https://www.langchain.com/blog/observability-for-llm-applications)

**Project Files to Review:**
- `src/observability/tracker.py` - Custom tracking system
- `src/evaluation/metrics.py` - Evaluation metrics
- `run_evaluation.py` - Automated evaluation pipeline

---

### 4.3 Evaluation & Metrics 🔴

**What to Learn:**
- Answer faithfulness (hallucination detection)
- Answer relevance
- Citation accuracy
- LLM-as-judge evaluation
- Creating evaluation datasets

**Resources:**
- **Paper:** "RAGAS: Automated Evaluation of RAG" (Es et al., 2023)
- **Library:** [RAGAS Documentation](https://docs.ragas.io/)
- **Article:** [Evaluating RAG Systems](https://www.anthropic.com/index/evaluating-ai-systems)

**Project Files to Review:**
- `src/evaluation/dataset.py` - Evaluation questions
- `src/evaluation/metrics.py` - Metric implementations

---

## Phase 5: Deep Dives (Ongoing)

### 5.1 Research Papers to Read

**Foundational:**
1. "Attention Is All You Need" (Transformers) - Vaswani et al., 2017
2. "BERT: Pre-training of Deep Bidirectional Transformers" - Devlin et al., 2018
3. "Retrieval-Augmented Generation for Knowledge-Intensive Tasks" - Lewis et al., 2020

**RAG-Specific:**
4. "Self-RAG: Learning to Retrieve, Generate, and Critique" - Asai et al., 2023
5. "REALM: Retrieval-Augmented Language Model Pre-Training" - Guu et al., 2020
6. "In-Context Retrieval-Augmented Language Models" - Ram et al., 2023

**Agents:**
7. "ReAct: Synergizing Reasoning and Acting in Language Models" - Yao et al., 2022
8. "Toolformer: Language Models Can Teach Themselves to Use Tools" - Schick et al., 2023

---

### 5.2 Key Concepts Map

```
Self-Correcting RAG System
│
├── Data Ingestion
│   ├── Chunking strategies
│   ├── Embedding generation
│   └── Vector DB indexing (HNSW)
│
├── Retrieval
│   ├── Dense retrieval (bi-encoder)
│   ├── CrossEncoder re-ranking
│   └── Hybrid search
│
├── Multi-Agent Workflow (LangGraph)
│   ├── Retrieval Agent
│   ├── Generation Agent
│   ├── Validation Agent (self-correction)
│   ├── Execution Agent (tool use)
│   └── Synthesis Agent
│
├── Optimizations
│   ├── Two-tier caching (Redis)
│   ├── Re-ranking (CrossEncoder)
│   └── Token optimization
│
├── Observability
│   ├── Structured logging
│   ├── Metrics tracking
│   └── Evaluation framework
│
└── Production
    ├── Containerization (Docker)
    ├── Orchestration (docker-compose)
    └── Health monitoring
```

---

## 📋 Study Checklist

### Week 1: Foundations
- [ ] Review Python async/await
- [ ] Understand transformer architecture basics
- [ ] Complete LangChain quickstart
- [ ] Read RAG fundamentals

### Week 2: Core Tech
- [ ] Set up local Qdrant and explore UI
- [ ] Build simple LangGraph workflow
- [ ] Implement basic Redis caching
- [ ] Study sentence transformers

### Week 3: Advanced
- [ ] Understand bi-encoder vs cross-encoder
- [ ] Implement multi-agent system (simple)
- [ ] Study tool calling patterns
- [ ] Review validation/self-correction

### Week 4: Production
- [ ] Build and run Docker containers
- [ ] Study docker-compose networks
- [ ] Implement custom logging
- [ ] Create evaluation metrics

---

## 🎯 Project-Specific Deep Dives

### Critical Files to Master (in order):

1. **`src/rag/qdrant_handler.py`** (3 hours)
   - Vector DB operations
   - Collection management
   - Search implementation

2. **`src/rag/retrieval.py`** (4 hours)
   - Two-stage retrieval
   - CrossEncoder re-ranking
   - Caching integration

3. **`src/agents/graph.py`** (6 hours) ⭐ **MOST IMPORTANT**
   - Complete workflow
   - State management
   - Conditional routing
   - Agent orchestration

4. **`src/agents/validation.py`** (3 hours)
   - Self-correction logic
   - Structured output parsing
   - Gap identification

5. **`src/cache/redis_cache.py`** (2 hours)
   - Two-tier strategy
   - TTL management
   - Cache invalidation

6. **`src/observability/tracker.py`** (2 hours)
   - Event logging
   - Metrics collection
   - JSON serialization

---

## 💡 Hands-On Learning Projects

To truly master the concepts, build these mini-projects:

### Project 1: Simple RAG (Week 1-2)
Build a basic RAG system:
- Embed 10 documents
- Store in Qdrant
- Implement simple retrieval
- Generate answer with LLM

### Project 2: Self-Correcting Agent (Week 3)
- Create validation agent
- Implement self-critique
- Add tool calling
- Handle retries

### Project 3: Cached RAG (Week 4)
- Add Redis caching
- Implement TTL
- Measure cache hit rate
- Compare performance

---

## 🔗 Essential Bookmarks

**Documentation:**
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [Redis Docs](https://redis.io/docs/)
- [Sentence Transformers](https://www.sbert.net/)

**Communities:**
- [LangChain Discord](https://discord.gg/langchain)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [Hugging Face Forums](https://discuss.huggingface.co/)

---

## 🎓 Certification Paths (Optional)

- **DeepLearning.AI:** LangChain & LangGraph courses
- **AWS:** Machine Learning Specialty
- **Docker:** Docker Certified Associate

---

## 📈 Skill Progression

**After 1 Week:** Understand basic RAG pipeline
**After 2 Weeks:** Can build simple RAG system
**After 3 Weeks:** Understand multi-agent architectures
**After 4 Weeks:** Can deploy production RAG system
**After 6 Weeks:** Can optimize and evaluate RAG systems

---

## 🚀 Next Steps After Mastery

1. **Experiment with variations:**
   - Try different embedding models
   - Test various LLMs (Claude, GPT-4, Llama)
   - Implement query expansion

2. **Add features:**
   - Conversational memory
   - Multi-turn conversations
   - User feedback integration

3. **Scale:**
   - Kubernetes deployment
   - Load balancing
   - Horizontal scaling

---

## 📌 Daily Study Routine (Suggested)

**Morning (1 hour):**
- Read 1 article or watch 1 tutorial
- Take notes on key concepts

**Afternoon (1 hour):**
- Code along with tutorial
- Experiment with project code

**Evening (30 min):**
- Review project files
- Document learnings
- Plan tomorrow's focus

---

**Remember:** This is a complex system touching multiple domains. Don't rush! Take time to understand each component thoroughly before moving to the next.

Good luck with your learning journey! 🎉
