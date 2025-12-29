# Advanced RAG Techniques Documentation

This document provides an overview of advanced Retrieval-Augmented Generation (RAG) techniques that can enhance the FusionRAG system.

## Table of Contents
1. [CRAG (Corrective Retrieval Augmented Generation)](#crag)
2. [LATS (Language Agent Tree Search)](#lats)
3. [Agentic RAG](#agentic-rag)
4. [Self-RAG](#self-rag)
5. [Adaptive RAG](#adaptive-rag)
6. [HyDE (Hypothetical Document Embedding)](#hyde)

---

## CRAG (Corrective Retrieval Augmented Generation)

### Overview
CRAG is an advanced RAG technique that focuses on making LLMs more reliable by adding a corrective layer that evaluates and improves retrieval quality before generation.

### How It Works
1. **Query and Retrieval**: Initial retrieval of top K documents from the knowledge base
2. **Retrieval Evaluation**: A lightweight evaluator scores document relevance (Correct/Incorrect/Ambiguous)
3. **Decision Pathways**:
   - **Correct**: Apply "decompose-then-recompose" to filter and rank knowledge strips
   - **Incorrect**: Discard documents and trigger web search for fresh information
   - **Ambiguous**: Combine internal refinement with external search
4. **Knowledge Refinement**: Filter, rerank, and validate information before LLM generation

### Key Benefits
- Reduces hallucinations through active validation
- Handles outdated or irrelevant information
- Self-correcting with web search fallback
- Plug-and-play with existing RAG systems

### Implementation with LlamaIndex
```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Implement retrieval evaluator
class RetrievalEvaluator:
    def evaluate(self, query, documents):
        # Score each document for relevance
        # Return: "correct", "incorrect", or "ambiguous"
        pass

# Add web search fallback for incorrect retrievals
# Implement knowledge strip decomposition for correct retrievals
```

### References
- [DataCamp CRAG Tutorial](https://www.datacamp.com/tutorial/corrective-rag-crag)
- [Meilisearch CRAG Guide](https://www.meilisearch.com/blog/corrective-rag)

---

## LATS (Language Agent Tree Search)

### Overview
LATS combines LLM capabilities with Monte Carlo Tree Search (MCTS) to enable systematic planning, reasoning, and multi-path exploration for complex problem-solving.

### How It Works
1. **Tree Search Structure**: Tasks represented as nodes, edges as exploratory actions
2. **Core Cycle**:
   - **Selection**: Choose next node using Upper Confidence Bound score
   - **Expansion**: Generate multiple candidate solutions using LLM
   - **Evaluation/Reflection**: Score new nodes for quality
   - **Backpropagation**: Update node values up the tree
   - **Iteration**: Repeat until solution or budget exhausted

### Integration with RAG
- **Multi-path Reasoning**: Explore multiple synthesis strategies over retrieved documents
- **Iterative Feedback**: Incorporate validation scores to prefer better solutions
- **Strategic Query Planning**: Decide when to search further vs synthesize

### Use Cases
- Multi-hop question answering
- Complex programming tasks
- Strategic planning problems
- Algorithm design

### Implementation with LlamaIndex
```python
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner

# Initialize LATS agent
agent_worker = LATSAgentWorker.from_tools(
    tools=[query_engine_tool],
    llm=llm,
    num_expansions=4,
    max_rollouts=16
)

agent = AgentRunner(agent_worker)
response = agent.chat("Complex multi-step query")
```

### References
- [LlamaIndex LATS Documentation](https://llamahub.ai/l/agent/llama-index-agent-lats)
- [Analytics Vidhya LATS Tutorial](https://www.analyticsvidhya.com/blog/2025/01/lats-advanced-ai-agent-with-llamaindex/)
- [arXiv Paper](https://arxiv.org/abs/2310.04406)

---

## Agentic RAG

### Overview
Agentic RAG incorporates autonomous AI agents that can plan, reason, and dynamically orchestrate retrieval workflows, going beyond simple retrieve-and-generate patterns.

### Key Differences from Traditional RAG

| Aspect | Traditional RAG | Agentic RAG |
|--------|----------------|-------------|
| Process | Sequential: retrieve → generate | Dynamic: plan → retrieve → validate → re-search → generate |
| Reasoning | Limited to immediate query | Multi-step reasoning with context |
| Adaptability | Fixed pipeline | Adaptive strategy selection |
| Memory | Stateless | Maintains context across interactions |
| Sources | Single retrieval | Multiple sources and APIs |

### Capabilities
- **Planning**: Break complex queries into sub-tasks
- **Dynamic Strategy**: Choose optimal retrieval methods
- **Validation**: Verify and cross-check information
- **Multi-source**: Combine knowledge from diverse sources
- **Context Management**: Maintain conversation history and learning

### Use Cases
- Advanced customer support
- Medical/healthcare multi-database queries
- Financial analysis across live APIs
- Educational personalization
- Complex research tasks

### Implementation Approach
```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool

# Create tools for different capabilities
query_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Search documents for information"
)

# Initialize agent with tools and memory
agent = ReActAgent.from_tools(
    tools=[query_tool, web_search_tool],
    llm=llm,
    memory=ChatMemoryBuffer.from_defaults(),
    verbose=True
)
```

### References
- [IBM Agentic RAG Guide](https://www.ibm.com/think/topics/agentic-rag)
- [DataCamp Agentic RAG Tutorial](https://www.datacamp.com/blog/agentic-rag)
- [Analytics Vidhya Comparison](https://www.analyticsvidhya.com/blog/2024/11/rag-vs-agentic-rag/)

---

## Self-RAG

### Overview
Self-RAG systems autonomously improve their retrieval and generation pipelines using in-context feedback, allowing the LLM to critique and refine results iteratively.

### How It Works
1. **Initial Retrieval**: Retrieve documents based on query
2. **Self-Critique**: LLM evaluates retrieval quality
3. **Reranking**: Dynamically reorder or filter documents
4. **Supplemental Retrieval**: Request additional context if needed
5. **Generation**: Produce final output with refined context

### Key Benefits
- Reduces hallucinations through self-validation
- Adapts to user intent in multi-turn conversations
- Iteratively improves document quality

### Implementation Pattern
```python
class SelfRAGEngine:
    def retrieve_and_critique(self, query):
        # Initial retrieval
        docs = self.retriever.retrieve(query)
        
        # Self-critique
        critique_prompt = f"Evaluate relevance of these documents to query: {query}"
        scores = self.llm.evaluate(critique_prompt, docs)
        
        # Decide: accept, rerank, or re-retrieve
        if max(scores) < threshold:
            docs = self.web_search(query)  # Fallback
        else:
            docs = self.rerank(docs, scores)
        
        return docs
```

### References
- [GitHub All-RAG-Techniques](https://github.com/mbergo/all-rag-techniques)
- [LlamaIndex Advanced RAG Guide](https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)

---

## Adaptive RAG

### Overview
Adaptive RAG dynamically optimizes the retrieval and generation pipeline based on query complexity, user history, and retrieval confidence.

### Key Concepts
1. **Dynamic Chunk Sizing**: Adjust chunk size based on query specificity
2. **Context Window Optimization**: Balance relevance with LLM context limits
3. **Adaptive Retrieval Strategy**: Switch between semantic, keyword, or hybrid retrieval

### Benefits
- Increases efficiency for both simple and complex queries
- Minimizes retrieval of irrelevant information
- Optimizes cost and latency
- Balances accuracy with resource usage

### Implementation Approach
```python
class AdaptiveRetriever:
    def retrieve(self, query):
        # Analyze query complexity
        complexity = self.analyze_complexity(query)
        
        if complexity == "simple":
            # Use keyword search with larger chunks
            return self.keyword_retriever.retrieve(query, top_k=2)
        elif complexity == "moderate":
            # Use hybrid retrieval
            return self.hybrid_retriever.retrieve(query, top_k=4)
        else:  # complex
            # Use semantic search with smaller chunks and more results
            return self.semantic_retriever.retrieve(query, top_k=8)
```

### References
- [Analytics Vidhya Advanced RAG](https://www.analyticsvidhya.com/blog/2025/04/advanced-rag-techniques/)
- [Designveloper RAG Guide](https://www.designveloper.com/blog/advanced-rag/)

---

## HyDE (Hypothetical Document Embedding)

### Overview
HyDE generates a hypothetical answer to a query using an LLM, then retrieves documents similar to this synthesized answer, uncovering context that traditional search might miss.

### How It Works
1. **Generate Hypothetical Answer**: LLM creates an answer to the query
2. **Embed Hypothetical Answer**: Create vector embedding of the generated answer
3. **Retrieve Similar Documents**: Search for documents similar to the hypothetical answer
4. **Generate Final Answer**: Use retrieved documents to produce grounded response

### Benefits
- Finds deeply relevant or "hidden" context
- Effective for novel or poorly worded questions
- Boosts coverage for ambiguous queries
- Excellent for multi-hop information needs

### Implementation with LlamaIndex
```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# Create HyDE query transformation
hyde_transform = HyDEQueryTransform(
    llm=llm,
    include_original=True
)

# Wrap query engine with HyDE transform
hyde_query_engine = TransformQueryEngine(
    query_engine=base_query_engine,
    query_transform=hyde_transform
)

response = hyde_query_engine.query("Your question here")
```

### References
- [Field Guide to AI HyDE Tutorial](https://fieldguidetoai.com/guides/advanced-rag-techniques)
- [DataCamp Advanced RAG](https://www.datacamp.com/blog/rag-advanced)

---

## Comparison Matrix

| Technique | Complexity | Best For | Primary Benefit |
|-----------|-----------|----------|----------------|
| CRAG | Medium | Handling unreliable retrievals | Self-correction and validation |
| LATS | High | Complex multi-step reasoning | Systematic planning and exploration |
| Agentic RAG | High | Dynamic, multi-source queries | Autonomous reasoning and orchestration |
| Self-RAG | Medium | Iterative refinement | Reduced hallucinations |
| Adaptive RAG | Medium | Variable query complexity | Efficiency and optimization |
| HyDE | Low | Ambiguous or novel queries | Better semantic matching |

---

## Implementation Priority for FusionRAG

Based on the current FusionRAG architecture, recommended implementation order:

1. **HyDE** (Low complexity, high impact for current setup)
2. **Self-RAG** (Natural extension of existing reranking)
3. **Adaptive RAG** (Optimize existing retrieval)
4. **CRAG** (Add validation layer)
5. **Agentic RAG** (Major architectural change)
6. **LATS** (Requires significant infrastructure)

---

## Additional Resources

### LlamaIndex Documentation
- [Building Advanced RAG](https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)
- [Query Transformations](https://docs.llamaindex.ai/en/stable/module_guides/querying/query_transforms/)
- [Agents](https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/)

### Research Papers
- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [LATS Paper](https://arxiv.org/abs/2310.04406)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)

### Community Resources
- [GitHub: All RAG Techniques](https://github.com/mbergo/all-rag-techniques)
- [LlamaIndex Hub](https://llamahub.ai/)
- [RAG Evaluation Frameworks](https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/evaluation)

---

*Last Updated: December 2025*
*For FusionRAG Project: https://github.com/Aariz1001/FusionRAG*
