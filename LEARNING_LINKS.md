# 40-Hour Roadmap - Complete Resource Links

All links organized by day for easy access. **Bookmark this page!**

---

## 📅 DAY 1: Foundations (5 hours)

### Videos
- **[Andrej Karpathy - Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)** (1 hour)
  - Best introduction to how LLMs work
  
- **[RAG from Scratch - LangChain](https://www.youtube.com/watch?v=sVcwVQRHIc8)** (30 min)
  - Complete RAG tutorial

### Articles & Tutorials
- **[What is RAG? - Pinecone](https://www.pinecone.io/learn/retrieval-augmented-generation/)**
  - Comprehensive RAG introduction
  
- **[Understanding Vector Embeddings - DeepLearning.AI](https://www.deeplearning.ai/the-batch/how-vector-similarity-search-works/)**
  - Vector similarity explained
  
- **[The Illustrated Transformer - Jay Alammar](http://jalammar.github.io/illustrated-transformer/)**
  - Visual guide to transformers

### Documentation
- **[LangChain Getting Started](https://python.langchain.com/docs/get_started/introduction)**
- **[Sentence Transformers Documentation](https://www.sbert.net/)**

### Tools to Install
```bash
pip install langchain langchain-community sentence-transformers
pip install qdrant-client chromadb
```

---

## 📅 DAY 2: LangChain & LangGraph (5 hours)

### Videos
- **[LangGraph Tutorial - Complete Guide](https://www.youtube.com/watch?v=9BPCV5TYPmg)** (45 min)
  - Official LangGraph walkthrough
  
- **[Building Agentic Workflows](https://www.youtube.com/watch?v=VKVGXvC3fDY)** (30 min)
  - LangChain agentic patterns

### Documentation
- **[LangChain Documentation - Home](https://python.langchain.com/docs/get_started/introduction)**
- **[LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)**
- **[LangGraph Documentation](https://langchain-ai.github.io/langgraph/)**
- **[LangGraph Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)**
- **[LangGraph Concepts](https://langchain-ai.github.io/langgraph/concepts/)**

### Interactive Tutorials
- **[LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)**
- **[LangGraph How-To Guides](https://langchain-ai.github.io/langgraph/how-tos/)**

### GitHub Examples
- **[LangGraph Examples Repository](https://github.com/langchain-ai/langgraph/tree/main/examples)**

---

## 📅 DAY 3: Multi-Agent Systems (5 hours)

### Research Papers
- **[ReAct: Reasoning and Acting in LLMs (arXiv)](https://arxiv.org/abs/2210.03629)**
  - Foundation paper for agent reasoning
  
- **[Toolformer Paper (arXiv)](https://arxiv.org/abs/2302.04761)**
  - How LLMs learn to use tools

### Videos
- **[Building Autonomous Agents](https://www.youtube.com/watch?v=DWUdGhRrv2c)** (1 hour)
  - Multi-agent systems explained
  
- **[Function Calling & Tool Use](https://www.youtube.com/watch?v=0lOSvOoF2to)** (25 min)
  - OpenAI function calling tutorial

### Articles
- **[Multi-Agent Workflows with LangGraph - LangChain Blog](https://blog.langchain.dev/langgraph-multi-agent-workflows/)**
  
- **[Building LLM-Powered Tools - Anthropic](https://www.anthropic.com/news/tool-use)**

### Documentation
- **[LangChain Tools Documentation](https://python.langchain.com/docs/modules/agents/tools/)**
- **[OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)**

---

## 📅 DAY 4: Retrieval & Re-ranking (5 hours)

### Videos
- **[Vector Databases Explained](https://www.youtube.com/watch?v=dN0lsF2cvm4)** (20 min)
  - Complete overview
  
- **[Semantic Search with Embeddings](https://www.youtube.com/watch?v=QvQXu3gYjzQ)** (15 min)
  - Practical tutorial

### Research Papers
- **[Sentence-BERT Paper (arXiv)](https://arxiv.org/abs/1908.10084)**
  - Bi-encoder architecture
  
- **[Dense Passage Retrieval (arXiv)](https://arxiv.org/abs/2004.04906)**
  - Dense retrieval fundamentals

### Tutorials
- **[Cross-Encoders for Re-ranking - SBERT](https://www.sbert.net/examples/applications/cross-encoder/README.html)**
  - Official cross-encoder tutorial
  
- **[Bi-Encoders vs Cross-Encoders - SBERT](https://www.sbert.net/examples/applications/retrieve_rerank/README.html)**
  - Comparison and use cases

### Documentation
- **[Qdrant Documentation](https://qdrant.tech/documentation/)**
- **[Qdrant Quick Start](https://qdrant.tech/documentation/quick-start/)**
- **[Qdrant Python Client](https://qdrant.tech/documentation/frameworks/langchain/)**
- **[Sentence Transformers Models Hub](https://www.sbert.net/docs/pretrained_models.html)**

### Interactive
- **[Qdrant UI Dashboard](http://localhost:6333/dashboard)** (after starting Qdrant)

---

## 📅 DAY 5: Caching & Performance (5 hours)

### Videos
- **[Redis in 100 Seconds](https://www.youtube.com/watch?v=G1rOthIU-uo)** (2 min)
  - Quick overview
  
- **[Redis Crash Course](https://www.youtube.com/watch?v=jgpVdJB2sKQ)** (30 min)
  - Complete tutorial
  
- **[Redis for Python Developers](https://www.youtube.com/watch?v=WQ61RL1GpEE)** (20 min)

### Articles
- **[Caching Best Practices - Redis](https://redis.com/blog/cache-invalidation-strategies/)**
- **[Two-Tier Caching Strategy](https://redis.com/blog/redis-caching-strategies/)**
- **[Redis LRU Eviction Explained](https://redis.io/docs/reference/eviction/)**

### Documentation
- **[Redis Documentation](https://redis.io/docs/)**
- **[Redis Commands Reference](https://redis.io/commands/)**
- **[redis-py Documentation](https://redis-py.readthedocs.io/en/stable/)**
- **[Redis Python Tutorial](https://realpython.com/python-redis/)**

### Tools
- **[Redis Insight (GUI)](https://redis.com/redis-enterprise/redis-insight/)** - Download
- **[redis-cli Tutorial](https://redis.io/docs/ui/cli/)**

---

## 📅 DAY 6: Observability & Evaluation (5 hours)

### Videos
- **[MLOps Crash Course](https://www.youtube.com/watch?v=Jy-AHot4Z2w)** (20 min)
- **[Observability for ML Systems](https://www.youtube.com/watch?v=96ODkTAMl4A)** (30 min)

### Articles
- **[Python Logging Guide - Real Python](https://realpython.com/python-logging/)**
- **[Structured Logging in Python](https://www.structlog.org/en/stable/)**
- **[Observability for LLM Applications - LangChain](https://blog.langchain.dev/observability-for-llm-applications/)**
- **[Evaluating RAG Systems - Anthropic](https://www.anthropic.com/index/evaluating-ai-systems)**

### Research Papers
- **[RAGAS: Automated Evaluation of RAG (arXiv)](https://arxiv.org/abs/2309.15217)**
  - RAG evaluation framework

### Documentation & Tools
- **[LangSmith Documentation](https://docs.smith.langchain.com/)**
- **[LangSmith Tracing Guide](https://docs.smith.langchain.com/tracing)**
- **[RAGAS Documentation](https://docs.ragas.io/)**
- **[RAGAS Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html)**

### Courses
- **[MLOps Specialization - DeepLearning.AI](https://www.deeplearning.ai/courses/mlops-specialization/)**
- **[Evaluating LLM Apps - DeepLearning.AI](https://www.deeplearning.ai/short-courses/evaluating-debugging-generative-ai/)**

---

## 📅 DAY 7: Docker & Deployment (5 hours)

### Videos
- **[Docker in 100 Seconds](https://www.youtube.com/watch?v=Gjnup-PuquQ)** (2 min)
  - Quick intro
  
- **[Docker Tutorial for Beginners](https://www.youtube.com/watch?v=fqMOX6JJhGo)** (3 hours - watch first hour)
  - Comprehensive tutorial
  
- **[Docker Compose Tutorial](https://www.youtube.com/watch?v=MVIcrmeV_6c)** (20 min)
  - docker-compose explained

### Documentation
- **[Docker Documentation](https://docs.docker.com/get-started/)**
- **[Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)**
- **[Docker Compose Documentation](https://docs.docker.com/compose/)**
- **[docker-compose.yml Reference](https://docs.docker.com/compose/compose-file/)**
- **[Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)**

### Interactive Tutorials
- **[Play with Docker](https://labs.play-with-docker.com/)** - Free online Docker playground
- **[Docker Getting Started Tutorial](https://docs.docker.com/get-started/)**

### Articles
- **[Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)**
- **[Multi-stage Builds](https://docs.docker.com/build/building/multi-stage/)**
- **[Health Checks in Docker](https://docs.docker.com/engine/reference/builder/#healthcheck)**

---

## 📅 DAY 8: Integration & Mastery (5 hours)

### Advanced Topics

#### Performance Optimization
- **[Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)**
- **[Async Python Tutorial](https://realpython.com/async-io-python/)**

#### Production Deployment
- **[Deploying Docker to AWS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html)**
- **[Kubernetes Tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)**

#### Advanced RAG
- **[Advanced RAG Techniques - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/index_structs/)**
- **[Query Transformations](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)**
- **[Hypothetical Document Embeddings (HyDE)](https://python.langchain.com/docs/how_to/hyde/)**

### Inspiration & Examples
- **[LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)**
- **[LangGraph Examples Repository](https://github.com/langchain-ai/langgraph/tree/main/examples)**
- **[RAG Projects on GitHub](https://github.com/topics/rag)**

---

## 🎓 Courses (Optional Deep Dives)

### Free Courses
1. **[ChatGPT Prompt Engineering - DeepLearning.AI](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)**
2. **[LangChain for LLM App Development - DeepLearning.AI](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)**
3. **[Building Systems with ChatGPT API - DeepLearning.AI](https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/)**
4. **[Vector Databases - DeepLearning.AI](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)**
5. **[Building Multi-Agent Systems - DeepLearning.AI](https://www.deeplearning.ai/short-courses/)**

### Paid Courses (Optional)
- **[Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)**
- **[MLOps Specialization - Coursera](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops)**

---

## 📚 Essential Books (Reference)

1. **"Designing Data-Intensive Applications" by Martin Kleppmann**
   - [Book Website](https://dataintensive.net/)
   
2. **"Fluent Python" by Luciano Ramalho**
   - [O'Reilly Store](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)

3. **"Building LLM Applications" by Valentina Alto**
   - [Manning Store](https://www.manning.com/books/building-llm-applications)

---

## 🔗 Communities & Forums

### Discord Servers
- **[LangChain Discord](https://discord.gg/langchain)** - Official community
- **[OpenAI Developers Discord](https://discord.gg/openai)**
- **[Hugging Face Discord](https://huggingface.co/join/discord)**

### Reddit
- **[r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)** - Local LLM community
- **[r/MachineLearning](https://reddit.com/r/MachineLearning)** - ML research
- **[r/LangChain](https://reddit.com/r/LangChain)** - LangChain specific

### Forums
- **[Hugging Face Forums](https://discuss.huggingface.co/)**
- **[LangChain Discussions](https://github.com/langchain-ai/langchain/discussions)**
- **[Stack Overflow - langchain tag](https://stackoverflow.com/questions/tagged/langchain)**

---

## 🛠️ Tools & Platforms

### Development Tools
- **[VS Code](https://code.visualstudio.com/)** - Free IDE
- **[PyCharm](https://www.jetbrains.com/pycharm/)** - Python IDE
- **[Cursor](https://cursor.sh/)** - AI-powered IDE

### Vector Databases
- **[Qdrant](https://qdrant.tech/)**
- **[Pinecone](https://www.pinecone.io/)**
- **[Weaviate](https://weaviate.io/)**
- **[Chroma](https://www.trychroma.com/)**

### LLM Platforms
- **[OpenAI Platform](https://platform.openai.com/)**
- **[Anthropic Claude](https://www.anthropic.com/)**
- **[Google AI Studio](https://aistudio.google.com/)**
- **[Hugging Face](https://huggingface.co/)**

### Observability
- **[LangSmith](https://www.langchain.com/langsmith)**
- **[Weights & Biases](https://wandb.ai/)**
- **[MLflow](https://mlflow.org/)**

---

## 📖 Research Papers (Advanced Reading)

### Foundational Papers
1. **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (Transformers)
2. **[BERT: Pre-training](https://arxiv.org/abs/1810.04805)**
3. **[GPT-3 Paper](https://arxiv.org/abs/2005.14165)**

### RAG-Specific
4. **[RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)**
5. **[Self-RAG](https://arxiv.org/abs/2310.11511)**
6. **[REALM: Retrieval-Augmented LM](https://arxiv.org/abs/2002.08909)**
7. **[In-Context RAG](https://arxiv.org/abs/2302.00083)**

### Agent Papers
8. **[ReAct: Reasoning + Acting](https://arxiv.org/abs/2210.03629)**
9. **[Toolformer](https://arxiv.org/abs/2302.04761)**
10. **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)**

### Evaluation
11. **[RAGAS: Automated RAG Evaluation](https://arxiv.org/abs/2309.15217)**

---

## 🎥 YouTube Channels to Follow

1. **[LangChain Official](https://www.youtube.com/@LangChain)**
2. **[Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)**
3. **[DeepLearning.AI](https://www.youtube.com/@Deeplearningai)**
4. **[sentdex](https://www.youtube.com/@sentdex)** - Python tutorials
5. **[ArjanCodes](https://www.youtube.com/@ArjanCodes)** - Software design
6. **[TechWithTim](https://www.youtube.com/@TechWithTim)** - Python projects

---

## 📰 Newsletters & Blogs

### Newsletters
- **[The Batch (DeepLearning.AI)](https://www.deeplearning.ai/the-batch/)**
- **[LangChain Blog](https://blog.langchain.dev/)**
- **[Hugging Face Newsletter](https://huggingface.co/blog)**

### Blogs to Follow
- **[OpenAI Blog](https://openai.com/blog/)**
- **[Anthropic Blog](https://www.anthropic.com/news)**
- **[Google AI Blog](https://ai.googleblog.com/)**
- **[Pinecone Blog](https://www.pinecone.io/blog/)**

---

## 🔍 Search Resources

### Code Search
- **[GitHub Code Search](https://github.com/search?type=code)** - Find RAG implementations
- **[Sourcegraph](https://sourcegraph.com/)** - Search across repos

### Paper Search
- **[arXiv.org](https://arxiv.org/)** - Research papers
- **[Papers with Code](https://paperswithcode.com/)** - Papers + implementations
- **[Semantic Scholar](https://www.semanticscholar.org/)** - AI-powered search

---

## 🎯 Cheat Sheets

- **[LangChain Cheat Sheet](https://python.langchain.com/docs/get_started/quickstart)**
- **[Docker Cheat Sheet](https://docs.docker.com/get-started/docker_cheatsheet.pdf)**
- **[Redis Commands Cheat Sheet](https://redis.io/docs/manual/cli/)**
- **[Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)**

---

## 📱 Mobile Apps (Learn on the go)

- **[Brilliant](https://brilliant.org/)** - Interactive CS & Math
- **[SoloLearn](https://www.sololearn.com/)** - Python coding
- **[Anki](https://apps.ankiweb.net/)** - Flashcards for concepts

---

## ✅ Quick Reference - Day-by-Day Links

**Day 1:** Start → [Karpathy LLM Video](https://www.youtube.com/watch?v=zjkBMFhNj_g)  
**Day 2:** Start → [LangGraph Tutorial](https://www.youtube.com/watch?v=9BPCV5TYPmg)  
**Day 3:** Read → [ReAct Paper](https://arxiv.org/abs/2210.03629)  
**Day 4:** Tutorial → [Cross-Encoder Guide](https://www.sbert.net/examples/applications/cross-encoder/README.html)  
**Day 5:** Watch → [Redis Crash Course](https://www.youtube.com/watch?v=jgpVdJB2sKQ)  
**Day 6:** Docs → [RAGAS Documentation](https://docs.ragas.io/)  
**Day 7:** Watch → [Docker Tutorial](https://www.youtube.com/watch?v=fqMOX6JJhGo)  
**Day 8:** Build → [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)  

---

**🎯 Pro Tip:** Bookmark this file and check off links as you complete them!

**🚀 Start NOW:** Click the first link → [Andrej Karpathy - Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g)

Good luck with your 40-hour journey! 🎉
