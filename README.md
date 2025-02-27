# FusionRAG: Advanced RAG Document Querying Chatbot

[![GitHub Stars](https://img.shields.io/github/stars/Aariz1001/FusionRAG.svg)](https://github.com/Aariz1001/FusionRAG/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Aariz1001/FusionRAG.svg)](https://github.com/Aariz1001/FusionRAG/issues)
[![GitHub License](https://img.shields.io/github/license/Aariz1001/FusionRAG)](https://github.com/Aariz1001/FusionRAG/blob/main/LICENSE) <!--- Update this link when you add LICENSE -->

FusionRAG is a sophisticated Retrieval-Augmented Generation (RAG) chatbot designed for accurate and efficient document querying.  It leverages a powerful combination of technologies and advanced techniques to significantly improve retrieval accuracy over baseline RAG implementations.  It boasts an up to 60% improvement in document retrieval accuracy by employing Query Fusion and Prompt Engineering techniques.

## Key Technologies

*   **LlamaIndex:** Used as the core framework for building the RAG pipeline, managing data ingestion, indexing, and querying.
*   **Ollama:**  Facilitates easy and efficient deployment of large language models (LLMs) for local inference.  This allows you to run the chatbot without relying on external API services.
*   **Streamlit:** Powers the interactive and user-friendly web-based user interface (UI) for interacting with the chatbot.

## Advanced Techniques

*   **Query Fusion:**  Expands the initial user query by generating multiple, related queries.  This helps to capture different aspects of the user's information need and improve the chances of retrieving relevant documents.
*   **Prompt Engineering:**  Carefully crafted prompts are used to guide the LLM's response generation, ensuring that the answers are concise, accurate, and relevant to the user's query and the retrieved context.

## Features

*   **Accurate Document Retrieval:**  Provides significantly improved document retrieval accuracy compared to standard RAG systems, thanks to the advanced techniques employed.
*   **Local Model Deployment:**  Runs the LLM locally using Ollama, ensuring privacy and eliminating the need for external API keys.
*   **Interactive Web UI:**  Offers an intuitive Streamlit-based web interface for easy interaction with the chatbot.
*   **Customizable:**  Provides flexibility to adapt the model, data sources, and UI to specific needs.

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
    *Make sure `requirements.txt` contains all project dependencies, including `llama-index`, `streamlit`, and any necessary Ollama-related packages (e.g., a Python client for Ollama if you're using one).*

3.  **Install and Configure Ollama:**

    *   Download and install Ollama from the official website: [https://ollama.ai/](https://ollama.ai/)
    *   Pull the desired LLM model (e.g., Mistral, Llama2).  See the Ollama documentation for available models and instructions:

        ```bash
        ollama pull mistral  # Or any other model you want to use.
        ```
    * *Ensure you have a model selected in the `constants.py`.*

4. **Configure data (if needed):**

    *   Place the documents you want to query into the `data` directory. The code currently handles PDF files. Modify the loading logic in `main.py` if you need to support other file types.
    * The files in the folder are iterated in `main.py`

## Usage

1.  **Start the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

2.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  Select the Ollama model that is called via the Ollama API

4.  Upload the documents and click the process documents button

5.  Enter your query in the input box and press Enter to interact with the chatbot.

## Contributing

Contributions are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear and concise commit messages.
4.  Submit a pull request.

## License

Apache 2.0 Licence: [LICENSE](LICENSE)

## Acknowledgements

*   LlamaIndex: [https://www.llamaindex.ai/](https://www.llamaindex.ai/)
*   Ollama: [https://ollama.ai/](https://ollama.ai/)
*   Streamlit: [https://streamlit.io/](https://streamlit.io/)
