# Changelog

## [Unreleased] - 2025-12-29

### Added

#### OpenRouter API Integration
- ✨ Support for 400+ models via OpenRouter API
- 🔑 API key management with .env file storage
- ✅ API key validation and status checking
- 🔍 Model search and filtering functionality
- 🔄 Automatic model list fetching and caching
- 🔀 Provider selection toggle (Ollama/OpenRouter)
- 📡 Streaming response support for OpenRouter

#### Token Counting & Cost Tracking
- 📊 Real-time token counting for prompts and responses
- 💰 Automatic cost calculation based on model pricing
- 📈 Context window visualization with progress bar
- 📉 Cumulative usage tracking across session
- 🔢 Separate input/output token metrics
- 🔄 Reset statistics functionality
- 💱 Per-request cost breakdown
- 🎯 Token estimation fallback for offline mode

#### Documentation
- 📚 ADVANCED_RAG_TECHNIQUES.md - Comprehensive guide to 6 advanced RAG techniques
  - CRAG (Corrective Retrieval Augmented Generation)
  - LATS (Language Agent Tree Search)
  - Agentic RAG
  - Self-RAG
  - Adaptive RAG
  - HyDE (Hypothetical Document Embedding)
- 🚀 QUICKSTART.md - 5-minute setup guide
- 📖 IMPLEMENTATION_SUMMARY.md - Complete implementation details
- 🏗️ ARCHITECTURE.md - System architecture documentation
- 📝 Enhanced README.md with comprehensive usage guide

#### Security & Configuration
- 🔒 .env file for secure API key storage
- 🙈 .gitignore to protect sensitive files
- 📄 .env.example template
- 🛡️ Password-masked API key input

### Fixed
- 🐛 Expander auto-expansion issue - "Sources Cited & Debug Info" now stays collapsed after follow-up queries
- 🔧 Improved error handling for offline scenarios
- 🌐 Added fallback token counting when tiktoken unavailable

### Changed
- ♻️ Major refactor of main.py (950+ lines)
- 🎨 Enhanced UI with better organization and visual feedback
- 📦 Updated requirements.txt with new dependencies
- 🔄 Provider-agnostic LLM interface in load_data()
- 🎯 Improved query transformation with provider support

### Dependencies Added
```
python-dotenv==1.0.0
requests==2.31.0
tiktoken==0.5.2
```

### Files Created/Modified
- `main.py` - Complete overhaul with new features (41KB)
- `requirements.txt` - Updated dependencies
- `README.md` - Comprehensive rewrite (8.2KB)
- `ADVANCED_RAG_TECHNIQUES.md` - New documentation (13KB)
- `QUICKSTART.md` - New quick start guide (4.9KB)
- `IMPLEMENTATION_SUMMARY.md` - New implementation details (9.0KB)
- `ARCHITECTURE.md` - New architecture documentation (10.6KB)
- `CHANGELOG.md` - This file
- `.env.example` - API key template (119B)
- `.gitignore` - Security configuration (439B)

### Technical Details

#### New Functions (10)
1. `get_openrouter_models()` - Fetch models from OpenRouter API
2. `search_openrouter_models()` - Filter models by search term
3. `save_api_key_to_env()` - Persist API key to .env
4. `load_api_key_from_env()` - Load API key from .env
5. `validate_openrouter_api_key()` - Validate API key with test request
6. `count_tokens()` - Count tokens with offline fallback
7. `calculate_cost()` - Calculate usage cost from token counts
8. `get_model_context_length()` - Get model's context window size
9. `get_model_pricing()` - Get model pricing information

#### Modified Functions (3)
1. `transform_query()` - Support for both Ollama and OpenRouter
2. `load_data()` - Provider-agnostic LLM initialization
3. `main()` - Complete UI redesign with new features

### UI Improvements
- 🎨 Restructured sidebar with clear sections
- 📊 Token usage statistics panel
- 🔍 Model search interface
- ⚡ Status indicators and feedback messages
- 📈 Visual progress bars
- 🎯 Better organization with emojis

### Performance
- ⚡ Model list caching in session state
- 🚀 Streaming responses for faster UX
- 🔄 Async operations where possible
- 💾 Efficient token counting with caching

### Testing
- ✅ Syntax validation passed
- ✅ AST parsing successful (16 functions)
- ✅ Logic tests for helper functions
- ✅ Token counting fallback verified
- ⏳ Manual testing required in live environment

### Known Limitations
1. Token counting for Ollama is estimated (no native API)
2. Model list limited to first 50 for UI performance
3. Cost tracking only for OpenRouter (Ollama is free/local)
4. Requires internet for OpenRouter and tiktoken encoding files

### Breaking Changes
None - All changes are additive and backward compatible with Ollama-only usage.

---

## How to Upgrade

### From Previous Version
```bash
# Pull latest changes
git pull

# Install new dependencies
pip install -r requirements.txt

# (Optional) Set up OpenRouter
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Configuration
No configuration changes required for existing Ollama users. OpenRouter is an optional addition.

---

**Contributors**: GitHub Copilot AI Agent
**Date**: December 29, 2025
