# FusionRAG Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FusionRAG Application                       │
│                         (Streamlit Frontend)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
            ┌───────▼────────┐            ┌────────▼────────┐
            │  Ollama Local  │            │  OpenRouter API │
            │    Provider    │            │    Provider     │
            └───────┬────────┘            └────────┬────────┘
                    │                               │
                    │   ┌──────────────────────┐   │
                    └───►   LLM Interface      ◄───┘
                        │  (LlamaIndex Core)   │
                        └──────────┬───────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
┌───────▼────────┐      ┌─────────▼─────────┐      ┌────────▼────────┐
│   Document     │      │    Retrieval      │      │  Token/Cost     │
│   Processing   │      │    System         │      │   Tracking      │
└───────┬────────┘      └─────────┬─────────┘      └────────┬────────┘
        │                          │                          │
        │                          │                          │
┌───────▼────────┐      ┌─────────▼─────────┐      ┌────────▼────────┐
│ PDF Parsing    │      │ Query Fusion      │      │ tiktoken        │
│ Text Splitting │      │ Vector Search     │      │ Cost Calculator │
│ Embeddings     │      │ BM25 Search       │      │ Context Monitor │
└───────┬────────┘      │ Reranking         │      └────────┬────────┘
        │               └─────────┬─────────┘               │
        │                         │                         │
        │               ┌─────────▼─────────┐              │
        │               │  FAISS Vector DB  │              │
        │               └───────────────────┘              │
        │                                                  │
        └──────────────────────┬───────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Response Generation │
                    │  with Streaming      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼──────────┐
                    │   User Interface     │
                    │  • Chat Display      │
                    │  • Debug Expander    │
                    │  • Usage Stats       │
                    └──────────────────────┘
```

## Component Flow

### 1. Document Upload & Processing
```
User uploads PDFs
    ↓
PyMuPDFReader parses PDFs
    ↓
Text split into chunks (1024 tokens)
    ↓
Embeddings generated (Ollama: snowflake-arctic-embed2)
    ↓
Stored in FAISS vector store
```

### 2. Query Processing
```
User enters query
    ↓
Token counting (tiktoken/fallback)
    ↓
Query transformation (Ollama: phi3.5 / OpenRouter: gemini-flash)
    ↓
Query Fusion (generates 6 sub-queries)
    ↓
Parallel retrieval:
  • Vector search (semantic)
  • BM25 search (keyword)
    ↓
Reciprocal rank fusion
    ↓
Reranking (CrossEncoder)
    ↓
Top 4 documents selected
```

### 3. Response Generation
```
Retrieved documents + Query
    ↓
LLM Generation:
  • Ollama: Local model
  • OpenRouter: API model
    ↓
Streaming response chunks
    ↓
Token counting & cost calculation
    ↓
Display to user
```

## Data Flow

### OpenRouter Provider Flow
```
┌──────────────────────────────────────────────────────────────┐
│ 1. Startup                                                    │
│    • Load API key from .env                                  │
│    • Validate key status                                     │
│    • Fetch available models                                  │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. Model Selection                                           │
│    • User searches models                                    │
│    • Filter 400+ models by name/provider                     │
│    • Select model with pricing info                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. Document Processing                                       │
│    • Use selected model for LLM tasks                        │
│    • Initialize OpenAILike adapter                           │
│    • Process and index documents                             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. Chat Interaction                                          │
│    • Count input tokens                                      │
│    • Send request to OpenRouter API                          │
│    • Stream response chunks                                  │
│    • Parse usage data from final chunk                       │
│    • Update token counts and costs                           │
│    • Display context window fillup                           │
└──────────────────────────────────────────────────────────────┘
```

### Ollama Provider Flow
```
┌──────────────────────────────────────────────────────────────┐
│ 1. Startup                                                    │
│    • Connect to Ollama service                               │
│    • List installed models                                   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. Model Selection                                           │
│    • User selects from installed models                      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. Document Processing                                       │
│    • Use selected model for LLM tasks                        │
│    • Initialize Ollama adapter                               │
│    • Process and index documents                             │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. Chat Interaction                                          │
│    • Estimate input tokens                                   │
│    • Use chat engine with Ollama                             │
│    • Stream response chunks                                  │
│    • Estimate response tokens                                │
│    • Update estimated counts                                 │
└──────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Dual Provider Architecture
- **Abstraction**: Common interface for both providers
- **Flexibility**: Users choose based on privacy/cost needs
- **Fallback**: Graceful degradation when one provider unavailable

### 2. Token Tracking Strategy
- **OpenRouter**: Real usage data from API response
- **Ollama**: Character-based estimation (1 token ≈ 4 chars)
- **tiktoken**: Used when available, fallback when offline

### 3. Cost Management
- **Real-time**: Calculate cost per request
- **Cumulative**: Track session totals
- **Transparent**: Show all metrics to user
- **Control**: Reset button for new sessions

### 4. Retrieval Pipeline
- **Hybrid**: Vector + BM25 for comprehensive coverage
- **Fusion**: Combine multiple retrieval strategies
- **Reranking**: Second-stage relevance scoring
- **Adaptive**: Can be extended with CRAG, LATS, etc.

### 5. User Experience
- **Progressive Disclosure**: Advanced info in collapsible expander
- **Visual Feedback**: Progress bars, token counts, status icons
- **Error Handling**: Friendly messages, graceful fallbacks
- **Persistence**: API keys saved, models cached

## Technology Stack

### Core Framework
- **Streamlit**: Web UI framework
- **LlamaIndex**: RAG orchestration
- **Python 3.8+**: Runtime

### LLM Providers
- **Ollama**: Local inference
- **OpenRouter**: Cloud API gateway

### Retrieval
- **FAISS**: Vector similarity search
- **BM25Okapi**: Keyword search
- **CrossEncoder**: Reranking

### Embeddings
- **snowflake-arctic-embed2**: Ollama embedding model

### Utilities
- **tiktoken**: Token counting
- **python-dotenv**: Environment management
- **requests**: HTTP client
- **PyMuPDF**: PDF parsing
- **NLTK**: Text processing

## Security Considerations

### API Key Protection
```
User enters key
    ↓
Stored in .env file (gitignored)
    ↓
Loaded at startup
    ↓
Password-masked in UI
    ↓
Never logged or committed
```

### Data Privacy
- **Local Mode**: All data stays local with Ollama
- **API Mode**: Data sent to OpenRouter (check their privacy policy)
- **Documents**: Processed locally, only embeddings stored

## Performance Optimizations

1. **Caching**: Model list cached in session state
2. **Streaming**: Responses stream for faster perceived performance
3. **Batch Processing**: Documents processed in chunks
4. **Lazy Loading**: Models loaded on demand
5. **Async Operations**: Query fusion runs in parallel

## Extension Points

### Easy to Add:
1. New LLM providers (Anthropic, Cohere, etc.)
2. Different embedding models
3. Additional document types
4. Custom reranking strategies
5. Advanced RAG techniques (CRAG, LATS, etc.)

### Configuration Options:
- System prompts (lines 326-405 in main.py)
- Chunk size (line 444: chunk_size=1024)
- Top-K retrieval (line 683: similarity_top_k=4)
- Query fusion count (line 691: num_queries=6)
- Reranking weights (line 492: 0.3 BM25 + 0.7 semantic)

## Monitoring & Debugging

### Available Metrics:
1. Token counts (input/output/total)
2. Cost per request and cumulative
3. Context window usage
4. Retrieved document scores
5. Query transformation details
6. Reranking process

### Debug Information:
- Original vs transformed query
- Retrieved documents with scores
- Reranking decisions
- Performance metrics

---

**Architecture Version**: 1.0  
**Last Updated**: December 2025  
**Maintainer**: FusionRAG Team
