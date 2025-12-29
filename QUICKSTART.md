# FusionRAG Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- (Optional) Ollama installed for local models
- (Optional) OpenRouter API key for cloud models

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aariz1001/FusionRAG.git
cd FusionRAG

# 2. Install dependencies
pip install -r requirements.txt
```

### Option A: Using Ollama (Local, Private, Free)

```bash
# 3a. Install Ollama (if not already installed)
# Visit https://ollama.ai/ and follow installation instructions

# 4a. Pull required models
ollama pull mistral              # Main LLM
ollama pull phi3.5               # Query transformation
ollama pull snowflake-arctic-embed2  # Embeddings

# 5a. Start the app
streamlit run main.py
```

### Option B: Using OpenRouter (Cloud, 400+ Models)

```bash
# 3b. Get API key from https://openrouter.ai/keys

# 4b. Create .env file
cp .env.example .env
# Edit .env and add: OPENROUTER_API_KEY=your_key_here

# 5b. Start the app
streamlit run main.py
```

## 📖 Basic Usage

### 1. Select Provider & Model
- Choose **Ollama** for local models or **OpenRouter** for API models
- For OpenRouter: Enter and validate your API key
- Search and select your preferred model

### 2. Upload Documents
- Click "Upload PDF files" and select 1-5 PDFs
- Click "🚀 Process files" to index documents
- Wait for "✓ Documents processed successfully!" message

### 3. Start Chatting
- Type your question in the chat input
- Press Enter or click send
- View the response and retrieved sources

### 4. Monitor Usage (OpenRouter)
- Check sidebar for token counts and costs
- View context window fillup
- Click "🔄 Reset Statistics" to clear

## 🎯 Key Features

### Token Tracking
- Real-time input/output token counts
- Automatic cost calculation for API usage
- Context window visualization

### Model Search
- Filter 400+ models by provider or name
- View pricing information
- Quick model switching

### Debug Information
- Expand "Sources Cited & Debug Info" to see:
  - Original vs. transformed query
  - Retrieved documents with scores
  - Reranking process details
- Stays collapsed after follow-up queries (fixed!)

## 💡 Tips & Tricks

### Getting Better Answers
1. **Be Specific**: Ask detailed questions for better retrieval
2. **Use Follow-ups**: Build on previous answers in conversation
3. **Check Sources**: Expand debug info to verify retrieved context

### Optimizing Costs
1. **Use Free Models**: OpenRouter offers many free models
2. **Monitor Tokens**: Keep an eye on usage statistics
3. **Reset Stats**: Clear counts between sessions

### Choosing Models
- **For Speed**: Use smaller models like `gemini-flash` or `llama-3-8b`
- **For Quality**: Use `gpt-4`, `claude-3`, or `gemini-pro`
- **For Cost**: Filter by free models in OpenRouter

## 🔧 Troubleshooting

### "No Ollama models found"
```bash
# Check if Ollama is running
ollama serve

# List installed models
ollama list

# Pull a model if needed
ollama pull mistral
```

### "API key is invalid"
1. Verify key at https://openrouter.ai/keys
2. Check for extra spaces in `.env` file
3. Ensure account has credits

### "Documents processed successfully but no chat"
- Refresh the page
- Re-upload documents
- Check browser console for errors

### Token counts seem off
- Token estimates use character-based approximation when offline
- OpenRouter provides exact counts via API
- Counts are cumulative across chat session

## 📚 Advanced Features

### Custom System Prompts
Edit the system prompts in `main.py` (lines 326-405) to customize behavior for your use case.

### Adding More Document Types
Extend the file uploader in `main.py` (line 664) to support `.txt`, `.docx`, etc.

### Implementing Advanced RAG Techniques
See [ADVANCED_RAG_TECHNIQUES.md](ADVANCED_RAG_TECHNIQUES.md) for:
- CRAG (Corrective RAG)
- LATS (Language Agent Tree Search)
- Agentic RAG
- Self-RAG
- Adaptive RAG
- HyDE

## 🆘 Getting Help

- **Issues**: https://github.com/Aariz1001/FusionRAG/issues
- **Discussions**: Check existing issues for solutions
- **Documentation**: See README.md and ADVANCED_RAG_TECHNIQUES.md

## 📊 Example Usage

### Academic Research
```
1. Upload: Research papers (PDFs)
2. Model: claude-3-opus or gpt-4 (high quality)
3. Query: "Summarize the key findings across all papers"
```

### Technical Documentation
```
1. Upload: API docs, user manuals (PDFs)
2. Model: gemini-pro or llama-3-70b (good balance)
3. Query: "How do I implement authentication?"
```

### Legal Documents
```
1. Upload: Contracts, policies (PDFs)
2. Model: claude-3 or gpt-4 (best accuracy)
3. Query: "What are the termination clauses?"
```

## 🎓 Learning Resources

- **LlamaIndex**: https://docs.llamaindex.ai/
- **OpenRouter**: https://openrouter.ai/docs
- **Ollama**: https://github.com/ollama/ollama
- **RAG Techniques**: See ADVANCED_RAG_TECHNIQUES.md

---

**Ready to start?** Run `streamlit run main.py` and begin chatting with your documents! 🚀
