# FusionRAG: Advanced RAG Document Querying Chatbot

[![GitHub Stars](https://img.shields.io/github/stars/Aariz1001/FusionRAG.svg)](https://github.com/Aariz1001/FusionRAG/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Aariz1001/FusionRAG.svg)](https://github.com/Aariz1001/FusionRAG/issues)
[![GitHub License](https://img.shields.io/github/license/Aariz1001/FusionRAG)](https://github.com/Aariz1001/FusionRAG/blob/main/LICENSE) <!--- Update this link when you add LICENSE -->

FusionRAG is a sophisticated Retrieval-Augmented Generation (RAG) chatbot designed for accurate and efficient document querying.  It leverages a powerful combination of technologies and advanced techniques to significantly improve retrieval accuracy over baseline RAG implementations.  It boasts an up to 60% improvement in document retrieval accuracy by employing Query Fusion and Prompt Engineering techniques.

## Key Technologies

*   **LlamaIndex:** Used as the core framework for building the RAG pipeline, managing data ingestion, indexing, and querying.
*   **Ollama:**  Facilitates easy and efficient deployment of large language models (LLMs) for local inference.  This allows you to run the chatbot without relying on external API services.
*   **OpenRouter:** Provides API access to 400+ models from various providers (OpenAI, Anthropic, Google, Meta, etc.) through a single API key.
*   **Streamlit:** Powers the interactive and user-friendly web-based user interface (UI) for interacting with the chatbot.

## Advanced Techniques

*   **Query Fusion:**  Expands the initial user query by generating multiple, related queries.  This helps to capture different aspects of the user's information need and improve the chances of retrieving relevant documents.
*   **Prompt Engineering:**  Carefully crafted prompts are used to guide the LLM's response generation, ensuring that the answers are concise, accurate, and relevant to the user's query and the retrieved context.
*   **Hybrid Retrieval:** Combines vector-based semantic search with BM25 keyword search for optimal document retrieval.
*   **Reranking:** Uses cross-encoder models to rerank retrieved documents for improved relevance.

For more advanced RAG techniques like CRAG, LATS, and Agentic RAG, see [ADVANCED_RAG_TECHNIQUES.md](ADVANCED_RAG_TECHNIQUES.md).

## Features

*   **Dual Provider Support:** Choose between local Ollama models or cloud-based OpenRouter API models
*   **Token Counting & Cost Tracking:** Real-time token usage monitoring with cost calculation for API models
*   **Context Window Visualization:** Progress bar showing context window fillup
*   **Model Search:** Search and filter through 400+ models available via OpenRouter
*   **API Key Management:** Securely store and validate OpenRouter API keys in `.env` file
*   **Accurate Document Retrieval:**  Significantly improved document retrieval accuracy compared to standard RAG systems
*   **Interactive Web UI:**  Intuitive Streamlit-based web interface for easy interaction
*   **Collapsible Debug Info:** View query transformation and retrieval details without auto-expansion

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Aariz1001/FusionRAG.git
    cd FusionRAG
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Provider (Choose One or Both):**

    ### Option A: Using Ollama (Local Models)
    
    *   Download and install Ollama from: [https://ollama.ai/](https://ollama.ai/)
    *   Pull the desired LLM model:

        ```bash
        ollama pull mistral  # Or any other model
        ollama pull phi3.5   # For query transformation
        ollama pull snowflake-arctic-embed2  # For embeddings
        ```

    ### Option B: Using OpenRouter (API Models)
    
    *   Sign up at [https://openrouter.ai/](https://openrouter.ai/)
    *   Get your API key from [https://openrouter.ai/keys](https://openrouter.ai/keys)
    *   Create a `.env` file in the project root:

        ```bash
        cp .env.example .env
        ```
    
    *   Edit `.env` and add your API key:

        ```
        OPENROUTER_API_KEY=your_actual_api_key_here
        ```

## Usage

1.  **Start the Streamlit App:**

    ```bash
    streamlit run main.py
    ```

2.  **Open your browser:** Navigate to `http://localhost:8501`

3.  **Select Provider:**
    *   Choose **Ollama** for local models (ensure Ollama is running)
    *   Choose **OpenRouter** for API models (enter and validate your API key)

4.  **Select Model:**
    *   For **Ollama**: Select from installed models
    *   For **OpenRouter**: Search and select from 400+ available models

5.  **Upload Documents:** 
    *   Upload 1-5 PDF files
    *   Click "🚀 Process files" to index documents

6.  **Chat:** Enter your questions in the chat input

7.  **Monitor Usage:** 
    *   View token counts and costs in the sidebar
    *   Check context window fillup (for OpenRouter)
    *   Expand "Sources Cited & Debug Info" to see retrieval details

## Features Walkthrough

### Token Counting & Cost Tracking

The sidebar displays real-time usage statistics:
- **Input Tokens:** Cumulative tokens from all prompts
- **Output Tokens:** Cumulative tokens from all responses  
- **Total Tokens:** Sum of input and output
- **Total Cost:** Calculated cost for OpenRouter API usage
- **Context Window:** Visual progress bar showing token usage vs. model's context length

### Model Search (OpenRouter)

Use the search box to filter models by:
- Provider (e.g., "openai", "anthropic", "google")
- Model name (e.g., "gpt-4", "claude", "gemini")
- Capabilities (e.g., "vision", "function-calling")

### API Key Management

1. Enter your OpenRouter API key in the sidebar
2. Click **💾 Save Key** to persist it in `.env` file
3. Click **🔄 Validate** to test the key
4. The app will auto-load the key from `.env` on next startup

### Collapsible Debug Info

The "Sources Cited & Debug Info" expander shows:
- Original vs. transformed query
- Retrieved documents with scores
- Reranking process
- Performance metrics

**Fixed Issue:** The expander now stays collapsed after follow-up queries, preventing auto-expansion.

## Advanced RAG Techniques

See [ADVANCED_RAG_TECHNIQUES.md](ADVANCED_RAG_TECHNIQUES.md) for detailed documentation on:
- **CRAG** (Corrective Retrieval Augmented Generation)
- **LATS** (Language Agent Tree Search)
- **Agentic RAG** (Autonomous AI agents)
- **Self-RAG** (Self-correcting retrieval)
- **Adaptive RAG** (Dynamic optimization)
- **HyDE** (Hypothetical Document Embedding)

## Project Structure

```
FusionRAG/
├── main.py                          # Main application file
├── requirements.txt                  # Python dependencies
├── .env.example                      # Example environment file
├── .gitignore                        # Git ignore rules
├── ADVANCED_RAG_TECHNIQUES.md        # Advanced RAG documentation
└── README.md                         # This file
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1.  Fork the repository
2.  Create a new branch for your feature or bug fix
3.  Make your changes and commit them with clear messages
4.  Submit a pull request

## License

Apache 2.0 License: [LICENSE](LICENSE)

## Acknowledgements

*   LlamaIndex: [https://www.llamaindex.ai/](https://www.llamaindex.ai/)
*   Ollama: [https://ollama.ai/](https://ollama.ai/)
*   OpenRouter: [https://openrouter.ai/](https://openrouter.ai/)
*   Streamlit: [https://streamlit.io/](https://streamlit.io/)

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Verify models are installed: `ollama list`

### OpenRouter API Issues
- Verify API key is valid at [https://openrouter.ai/keys](https://openrouter.ai/keys)
- Check your account has credits
- Ensure `.env` file is in the project root

### Token Counting
- Token counts are estimates when tiktoken is offline
- OpenRouter provides exact token counts in API responses
- Ollama uses character-based estimation (~1 token per 4 characters)

## Support

For issues and questions:
- Create an issue: [GitHub Issues](https://github.com/Aariz1001/FusionRAG/issues)
- Check existing issues for solutions

