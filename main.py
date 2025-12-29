# tools
import glob
from tempfile import NamedTemporaryFile
import clipboard
import tempfile
import os
import json
import requests
from dotenv import load_dotenv
# ollama
import ollama
from llama_index.vector_stores.faiss import FaissVectorStore
# llamaindex
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, CBEventType
from llama_index.core.callbacks.base import BaseCallbackHandler
import nest_asyncio
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
import faiss
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.utils import get_response_text
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from sentence_transformers import CrossEncoder
import nltk
import numpy as np
from rank_bm25 import BM25Okapi
# tiktoken for token counting
import tiktoken
# streamlit
import streamlit as st

# Load environment variables
load_dotenv()

nest_asyncio.apply()

# OpenRouter API functions
def get_openrouter_models():
    """Fetch available models from OpenRouter API.
    
    Returns:
        list: List of model dictionaries with id, name, context_length, and pricing info
    """
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
        response.raise_for_status()
        models_data = response.json().get("data", [])
        return models_data
    except Exception as e:
        st.error(f"Error fetching OpenRouter models: {e}")
        return []

def search_openrouter_models(models, search_term):
    """Search models by name or ID.
    
    Args:
        models (list): List of model dictionaries
        search_term (str): Search term to filter models
        
    Returns:
        list: Filtered list of models
    """
    if not search_term:
        return models
    
    search_lower = search_term.lower()
    filtered = [
        m for m in models 
        if search_lower in m.get("id", "").lower() 
        or search_lower in m.get("name", "").lower()
    ]
    return filtered

def save_api_key_to_env(api_key):
    """Save OpenRouter API key to .env file.
    
    Args:
        api_key (str): The API key to save
    """
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    
    # Read existing .env content
    existing_lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            existing_lines = f.readlines()
    
    # Update or add API key
    key_found = False
    for i, line in enumerate(existing_lines):
        if line.startswith('OPENROUTER_API_KEY='):
            existing_lines[i] = f'OPENROUTER_API_KEY={api_key}\n'
            key_found = True
            break
    
    if not key_found:
        existing_lines.append(f'OPENROUTER_API_KEY={api_key}\n')
    
    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(existing_lines)

def load_api_key_from_env():
    """Load OpenRouter API key from .env file.
    
    Returns:
        str: The API key if found, None otherwise
    """
    return os.getenv('OPENROUTER_API_KEY')

def validate_openrouter_api_key(api_key):
    """Validate OpenRouter API key by making a test request.
    
    Args:
        api_key (str): The API key to validate
        
    Returns:
        tuple: (bool, str) - (is_valid, message)
    """
    if not api_key or api_key == 'your_api_key_here':
        return False, "Invalid API key format"
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        # Simple test request to check auth
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            return True, "API key is valid ✓"
        elif response.status_code == 401:
            return False, "API key is invalid or expired"
        else:
            return False, f"API error: {response.status_code}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def count_tokens(text, model="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken with fallback for offline environments.
    
    Args:
        text (str): Text to count tokens for
        model (str): Model name for encoding
        
    Returns:
        int: Number of tokens (estimated if tiktoken unavailable)
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except (KeyError, Exception) as e:
        # Fallback: rough estimation (1 token ≈ 4 characters for English text)
        return len(text) // 4

def calculate_cost(prompt_tokens, completion_tokens, model_pricing):
    """Calculate cost based on token usage and model pricing.
    
    Args:
        prompt_tokens (int): Number of prompt tokens
        completion_tokens (int): Number of completion tokens
        model_pricing (dict): Pricing info with 'prompt' and 'completion' costs
        
    Returns:
        float: Total cost in USD
    """
    prompt_cost = (prompt_tokens / 1_000_000) * float(model_pricing.get('prompt', 0))
    completion_cost = (completion_tokens / 1_000_000) * float(model_pricing.get('completion', 0))
    return prompt_cost + completion_cost

def get_model_context_length(model_id, models_data):
    """Get context length for a specific model.
    
    Args:
        model_id (str): Model identifier
        models_data (list): List of model dictionaries
        
    Returns:
        int: Context length in tokens
    """
    for model in models_data:
        if model.get('id') == model_id:
            return model.get('context_length', 4096)
    return 4096  # default

def get_model_pricing(model_id, models_data):
    """Get pricing info for a specific model.
    
    Args:
        model_id (str): Model identifier
        models_data (list): List of model dictionaries
        
    Returns:
        dict: Pricing dictionary with 'prompt' and 'completion' keys
    """
    for model in models_data:
        if model.get('id') == model_id:
            pricing = model.get('pricing', {})
            return {
                'prompt': pricing.get('prompt', 0),
                'completion': pricing.get('completion', 0)
            }
    return {'prompt': 0, 'completion': 0}


def transform_query(query: str, chat_history: list) -> str:
    """Transforms the user query by incorporating chat history context.
    
    Args:
        query (str): The current user query
        chat_history (list): List of previous chat messages
        
    Returns:
        str: Enhanced query incorporating context
    """
    if not query or query.strip() == "":
        return "Please provide a valid query"

    # Extract recent chat context
    recent_context = chat_history[-4:] if len(chat_history) > 0 else []
    context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_context])

    if not st.session_state.retriever:
        return query
    
    nodes = st.session_state.retriever.retrieve(query)
    if not nodes:
        return query

    doc_context = "\n".join([node.text for node in nodes[:2]])
    
    # Create enhanced query
    enhanced_query = f"""Based on this conversation:

    Chat History:
    {context_text}

    Relevant Document Context:
    {doc_context}
    
    Transform this query: {query}
    into a detailed standalone question that captures the full context.
    If the query doesn't mention anything specifically, refer and relate back to the previous query.
    Do not include anything else but the prompt.
    Do not include an answer to the prompt. Write just the prompt."""
    
    # Get transformed query from LLM based on provider
    if st.session_state.get("provider") == "OpenRouter":
        # Use a lightweight model for query transformation
        try:
            api_key = st.session_state.get("openrouter_api_key")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "google/gemini-2.0-flash-exp:free",
                "messages": [{"role": "user", "content": enhanced_query}]
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code == 200:
                transformed = response.json()['choices'][0]['message']['content']
                return transformed.strip() if transformed else query
        except Exception as e:
            st.warning(f"Query transformation error: {e}, using original query")
            return query
    else:
        # Use Ollama
        llm = Ollama(
            model="phi3.5:latest",
            n_gpu_layers=-1,
        )
        transformed = llm.complete(enhanced_query).text
        return transformed.strip() if transformed else query

def response_generator(stream):
    """Generator that yields chunks of data from a stream response.
    Args:
        stream: The stream object from which to read data chunks.
    Yields:
        bytes: The next chunk of data from the stream response.
    """
    for chunk in stream.response_gen:
        yield chunk

        

@st.cache_resource(show_spinner=False)
def load_data(documents, model_name: str, provider: str = "Ollama", api_key: str = None) -> VectorStoreIndex:
    """Loads and indexes documents using specified LLM provider and Llamaindex.

    This function performs the following actions:

    1. LLM Initialization: Initializes LLM (Ollama or OpenRouter) using the provided model name
    2. Data Ingestion: Reads documents (PDFs) using the SimpleDirectoryReader class
    3. Text Splitting and Embedding: Splits documents and generates embeddings
    4. Service Context Creation: Creates a ServiceContext with LLM, embedding model, and system prompt
    5. VectorStore Indexing: Creates a VectorStoreIndex from processed documents

    Args:
        documents: List of uploaded documents
        model_name (str): The name of the LLM model to be used
        provider (str): Provider name ("Ollama" or "OpenRouter")
        api_key (str): API key for OpenRouter (if using OpenRouter)

    Returns:
        VectorStoreIndex: An instance of VectorStoreIndex containing the indexed documents and embeddings.
    """
    

    Settings.system_prompt = """You are an expert mathematician with extensive experience in mathematical competitions. You approach problems through systematic thinking and rigorous reasoning. When solving problems, follow these thought processes:
    ## Deep Understanding
    Take time to fully comprehend the problem before attempting a solution. Consider:
    - What is the real question being asked?
    - What are the given conditions and what do they tell us?
    - Are there any special restrictions or assumptions?
    - Which information is crucial and which is supplementary?
    ## Multi-angle Analysis
    Before solving, conduct thorough analysis:
    - What mathematical concepts and properties are involved?
    - Can you recall similar classic problems or solution methods?
    - Would diagrams or tables help visualize the problem?
    - Are there special cases that need separate consideration?
    ## Systematic Thinking
    Plan your solution path:
    - Propose multiple possible approaches
    - Analyze the feasibility and merits of each method
    - Choose the most appropriate method and explain why
    - Break complex problems into smaller, manageable steps
    ## Rigorous Proof
    During the solution process:
    - Provide solid justification for each step
    - Include detailed proofs for key conclusions
    - Pay attention to logical connections
    - Be vigilant about potential oversights
    ## Repeated Verification
    After completing your solution:
    - Verify your results satisfy all conditions
    - Check for overlooked special cases
    - Consider if the solution can be optimized or simplified
    - Review your reasoning process
    Remember:
    1. Take time to think thoroughly rather than rushing to an answer
    2. Rigorously prove each key conclusion
    3. Keep an open mind and try different approaches
    4. Summarize valuable problem-solving methods
    5. Maintain healthy skepticism and verify multiple times
    Your response should reflect deep mathematical understanding and precise logical thinking, making your solution path and reasoning clear to others.
    When you're ready, present your complete solution with:
    - Clear problem understanding
    - Detailed solution process
    - Key insights
    - Thorough verification
    Focus on clear, logical progression of ideas and thorough explanation of your mathematical reasoning. Provide answers in the same language as the user asking the question, repeat the final answer using a '\\boxed{}' without any units, you have [[8192]] tokens to complete the answer.
    """


    llm_system_prompt = """You are an advanced Retrieval-Augmented Generation (RAG) system with Query Fusion capabilities. Your task is to deliver the most accurate and comprehensive answers by:

    - Understanding the User Query
        - Carefully read and interpret the user’s request.
        - Identify key terms or concepts that may need additional context.
    
    Generating Multiple Queries

    Expand or rephrase the original query into 5 distinct but relevant sub-queries.
    Ensure these sub-queries capture different perspectives or nuances of the user’s request.
    Retrieving Relevant Information

    For each sub-query, retrieve the most pertinent documents or pieces of information from the knowledge base.
    Pay attention to overlapping information and complementary details.
    Summarizing Each Retrieval

    Extract and summarize the critical points from each set of retrieved documents.
    Avoid fabrications; only use facts or data that are present in the retrieved context.
    Synthesizing a Unified Answer

    Merge the key insights from all five summaries into a single, coherent response.
    Maintain factual accuracy and clarity.
    If multiple points conflict, present the conflict, indicate uncertainty, or choose the most credible evidence.
    Ensuring Transparency & Accuracy

    Where appropriate, reference or cite sources to enhance trust and credibility.
    If the user query is ambiguous or there is insufficient context, ask clarifying questions or state the limitations.
    Final Answer Delivery

    Provide a concise, direct, and helpful response to the user’s query.
    Refrain from including any unsupported speculation or extraneous details.
    Follow these steps meticulously to generate answers that are grounded, comprehensive, and highly accurate.
    """



    # llm initialization based on provider
    if provider == "OpenRouter" and api_key:
        Settings.llm = OpenAILike(
            model=model_name,
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            is_chat_model=True,
            system_prompt=llm_system_prompt,
            timeout=120
        )
    else:
        Settings.llm = Ollama(
            model=model_name, 
            n_gpu_layers=-1, 
            request_timeout=120,
            system_prompt=llm_system_prompt
        )

    QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
    )

    # data ingestion
    all_docs = []
    for document in documents:
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pdf') as temp_file:
            temp_file.write(document.getbuffer())
            temp_file_path = temp_file.name

        try:
            with st.spinner(text="Loading and indexing the documents. This should take 1-2 minutes."):
                parser = PyMuPDFReader()
                file_extractor = {".pdf": parser}
                docs = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor=file_extractor).load_data()
                all_docs.extend(docs)
        finally:
            if os.path.exists(temp_file_path):
                os.chmod(temp_file_path, 0o777)  # Ensure we have permission to delete
                os.unlink(temp_file_path)

    # embeddings | query container
    Settings.text_splitter = SentenceSplitter(
        chunk_size=1024,
    )
    Settings.embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed2:latest", n_gpu_layers=-1)
    #Settings.embed_model = OllamaEmbedding(model_name="milkey/gte:large-zh-f16", n_gpu_layers=-1)
    d = 1536
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # indexing db
    index = VectorStoreIndex.from_documents(
        all_docs,
        embed_model=OllamaEmbedding(model_name="snowflake-arctic-embed2:latest", n_gpu_layers=-1),
        vector_store=vector_store
    )
    return index

def compute_bm25(query, results):
    tokenizer = nltk.word_tokenize
    bm25 = BM25Okapi([tokenizer(doc.text) for doc in results])
    return bm25.get_scores(tokenizer(query))

def compute_semantic(query, results):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', device="cuda")
    pairs = [[query, result.text] for result in results]
    return cross_encoder.predict(pairs)

def rerank_results(query, initial_results):
    bm25_scores = compute_bm25(query, initial_results)
    semantic_scores = compute_semantic(query, initial_results)

    # Apply softmax normalization
    bm25_norm = np.exp(bm25_scores) / np.sum(np.exp(bm25_scores))
    semantic_norm = np.exp(semantic_scores) / np.sum(np.exp(semantic_scores))

    # Weighted combination
    final_scores = bm25_norm * 0.3 + semantic_norm * 0.7

    # Sort results by final scores
    reranked = [x for _, x in sorted(zip(final_scores, initial_results), reverse=True)]
    return reranked[:4]



def main() -> None:
    """Controls the main chat application logic using Streamlit and Ollama.

    This function serves as the primary orchestrator of a chat application with the following tasks:

    1. Page Configuration: Sets up the Streamlit page's title, icon, layout, and sidebar using st.set_page_config.
    2. Model Selection: Manages model selection using st.selectbox and stores the chosen model in Streamlit's session state.
    3. Chat History Initialization: Initializes the chat history list in session state if it doesn't exist.
    4. Data Loading and Indexing: Calls the load_data function to create a VectorStoreIndex from the provided model name.
    5. Chat Engine Initialization: Initializes the chat engine using the VectorStoreIndex instance, enabling context-aware and streaming responses.
    6. Chat History Display: Iterates through the chat history messages and presents them using Streamlit's chat message components.
    7. User Input Handling:
          - Accepts user input through st.chat_input.
          - Appends the user's input to the chat history.
          - Displays the user's message in the chat interface.
    8. Chat Assistant Response Generation:
          - Uses the chat engine to generate a response to the user's prompt.
          - Displays the assistant's response in the chat interface, employing st.write_stream for streaming responses.
          - Appends the assistant's response to the chat history.

    Args:
        docs_path (str): Path of the documents to query.
    """
    if "expander_state" not in st.session_state:
        st.session_state.expander_state = False

    st.set_page_config(page_title="Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with documents 💬")
    
    if "activate_chat" not in st.session_state:
        st.session_state.activate_chat = False

    if "expander_state" not in st.session_state:
        st.session_state.expander_state = False
    
    # Initialize token tracking
    if "total_prompt_tokens" not in st.session_state:
        st.session_state.total_prompt_tokens = 0
    if "total_completion_tokens" not in st.session_state:
        st.session_state.total_completion_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "openrouter_models" not in st.session_state:
        st.session_state.openrouter_models = []

    with st.sidebar:
        st.markdown("## 🤖 Model Configuration")
        
        # Provider selection
        if "provider" not in st.session_state:
            st.session_state["provider"] = "Ollama"
        
        provider = st.radio(
            "Select Provider",
            ["Ollama", "OpenRouter"],
            key="provider_radio",
            horizontal=True
        )
        st.session_state["provider"] = provider
        
        # OpenRouter API Key Management
        if provider == "OpenRouter":
            st.markdown("### 🔑 OpenRouter API Key")
            
            # Try to load API key from env
            if "openrouter_api_key" not in st.session_state:
                loaded_key = load_api_key_from_env()
                if loaded_key and loaded_key != 'your_api_key_here':
                    st.session_state.openrouter_api_key = loaded_key
                    is_valid, msg = validate_openrouter_api_key(loaded_key)
                    if is_valid:
                        st.success(f"API key loaded from .env file {msg}")
                    else:
                        st.warning(f"Loaded API key is invalid: {msg}")
                else:
                    st.session_state.openrouter_api_key = ""
            
            # API key input
            api_key_input = st.text_input(
                "Enter OpenRouter API Key",
                type="password",
                value=st.session_state.get("openrouter_api_key", ""),
                help="Get your API key from https://openrouter.ai/keys"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save Key", help="Save API key to .env file"):
                    if api_key_input:
                        is_valid, msg = validate_openrouter_api_key(api_key_input)
                        if is_valid:
                            save_api_key_to_env(api_key_input)
                            st.session_state.openrouter_api_key = api_key_input
                            st.success("API key saved to .env file ✓")
                        else:
                            st.error(f"Invalid API key: {msg}")
                    else:
                        st.warning("Please enter an API key")
            
            with col2:
                if st.button("🔄 Validate", help="Test API key"):
                    if api_key_input:
                        is_valid, msg = validate_openrouter_api_key(api_key_input)
                        if is_valid:
                            st.success(msg)
                            st.session_state.openrouter_api_key = api_key_input
                        else:
                            st.error(msg)
                    else:
                        st.warning("Please enter an API key")
            
            # Load and cache models
            if st.session_state.get("openrouter_api_key"):
                if not st.session_state.openrouter_models or st.button("🔄 Refresh Models"):
                    with st.spinner("Loading models..."):
                        st.session_state.openrouter_models = get_openrouter_models()
                
                if st.session_state.openrouter_models:
                    st.markdown("### 🔍 Search Models")
                    search_term = st.text_input(
                        "Search by model name or ID",
                        placeholder="e.g., gpt-4, claude, gemini",
                        help="Filter models by name or provider"
                    )
                    
                    filtered_models = search_openrouter_models(
                        st.session_state.openrouter_models,
                        search_term
                    )
                    
                    # Display count
                    st.caption(f"Found {len(filtered_models)} models")
                    
                    # Model selection
                    if filtered_models:
                        model_options = [
                            f"{m.get('id', 'unknown')} | {m.get('name', 'Unknown')} | ${float(m.get('pricing', {}).get('prompt', 0)) * 1000:.4f}/1K tokens"
                            for m in filtered_models[:50]  # Limit to first 50 for performance
                        ]
                        
                        selected_option = st.selectbox(
                            "Select Model",
                            model_options,
                            help="Choose from available models"
                        )
                        
                        # Extract model ID
                        if selected_option:
                            st.session_state["model"] = selected_option.split(" | ")[0]
                    else:
                        st.warning("No models found matching your search")
                else:
                    st.error("Could not load models from OpenRouter")
            else:
                st.warning("⚠️ Please enter and validate your OpenRouter API key to continue")
        
        else:  # Ollama
            st.markdown("### 📦 Ollama Models")
            if "model" not in st.session_state:
                st.session_state["model"] = ""
            try:
                models = [model["name"] for model in ollama.list()["models"]]
                if models:
                    st.session_state["model"] = st.selectbox("Select a model", models)
                else:
                    st.warning("No Ollama models found. Please pull a model first.")
            except Exception as e:
                st.error(f"Error connecting to Ollama: {e}")
                st.info("Make sure Ollama is running")
        
        st.markdown("---")
        st.markdown("## 📄 Document Upload")
        
        # data ingestion
        documents = st.file_uploader("Upload PDF files to query (max 5)", type=['pdf'], accept_multiple_files=True)


        # file processing                
        if st.button('🚀 Process files', help="Process uploaded documents and create index"):
            if documents and len(documents) <= 5:
                # Validate requirements based on provider
                can_process = True
                if st.session_state["provider"] == "OpenRouter":
                    if not st.session_state.get("openrouter_api_key"):
                        st.error("Please enter and validate your OpenRouter API key first")
                        can_process = False
                    if not st.session_state.get("model"):
                        st.error("Please select a model")
                        can_process = False
                else:  # Ollama
                    if not st.session_state.get("model"):
                        st.error("Please select an Ollama model")
                        can_process = False
                
                if can_process:
                    index = load_data(
                        documents, 
                        st.session_state["model"],
                        st.session_state["provider"],
                        st.session_state.get("openrouter_api_key")
                    )
                    vector_retriever = index.as_retriever(similarity_top_k=4)
                    bm25_retriever = BM25Retriever.from_defaults(
                        docstore=index.docstore, similarity_top_k=4
                    )
                    
                    retriever = QueryFusionRetriever(
                        [vector_retriever, bm25_retriever],
                        similarity_top_k=4,
                        num_queries=6,  # set this to 1 to disable query generation
                        mode="reciprocal_rerank",
                        use_async=True,
                        verbose=True,
                    )
                    st.session_state.retriever = retriever
                    st.session_state.activate_chat = True
                    st.success("✓ Documents processed successfully!")
            elif len(documents) > 5:
                st.warning("Please upload a maximum of 5 files.")
            else:
                st.warning("Please upload at least one file.")
        
        # Display token usage statistics
        if st.session_state.get("activate_chat"):
            st.markdown("---")
            st.markdown("## 📊 Usage Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Tokens", f"{st.session_state.total_prompt_tokens:,}")
                st.metric("Output Tokens", f"{st.session_state.total_completion_tokens:,}")
            with col2:
                total_tokens = st.session_state.total_prompt_tokens + st.session_state.total_completion_tokens
                st.metric("Total Tokens", f"{total_tokens:,}")
                st.metric("Total Cost", f"${st.session_state.total_cost:.6f}")
            
            # Context window fillup if using OpenRouter
            if st.session_state["provider"] == "OpenRouter" and st.session_state.openrouter_models:
                context_length = get_model_context_length(
                    st.session_state["model"],
                    st.session_state.openrouter_models
                )
                fillup_pct = (total_tokens / context_length * 100) if context_length > 0 else 0
                st.progress(min(fillup_pct / 100, 1.0))
                st.caption(f"Context window: {total_tokens:,} / {context_length:,} tokens ({fillup_pct:.1f}%)")
            
            if st.button("🔄 Reset Statistics"):
                st.session_state.total_prompt_tokens = 0
                st.session_state.total_completion_tokens = 0
                st.session_state.total_cost = 0.0
                st.rerun()

            

    if st.session_state.activate_chat == True:
        # initialize chat history                   
        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "chat_engine" not in st.session_state.keys():

            query_engine = RetrieverQueryEngine.from_args(
                retriever,
                response_mode="refine",
                alpha=0.7,
                use_async=True,
                streaming=True
            )

            chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
                chat_history=st.session_state.messages,
                #memory=memory,
                streaming=True
            )

            st.session_state.chat_engine = chat_engine

        # display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # accept user input
        if prompt := st.chat_input("How can I help you?"):
            # Count prompt tokens
            prompt_tokens = count_tokens(prompt)
            
            # add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # chat assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                message_placeholder = st.empty()

                transformed_prompt = transform_query(prompt, st.session_state.messages)

                if transformed_prompt and transformed_prompt.strip():
                    nodes = st.session_state.retriever.retrieve(transformed_prompt)

                # FIXED: Set expander_state to False before creating expander to keep it collapsed
                st.session_state.expander_state = False
                expander = st.expander("Sources Cited & Debug Info", expanded=st.session_state.expander_state)

                with expander:
                    # Query Transformation Section
                    st.write('#### Query Processing:')
                    st.markdown("**Original Query:**")
                    st.code(prompt)
                    st.markdown("**Transformed Query:**")
                    st.code(transformed_prompt.strip())
                    
                    # Retrieval Process Section
                    st.write('#### Retrieval Process:')
                    st.markdown("**Vector Retrieval:**")
                    nodes = st.session_state.retriever.retrieve(transformed_prompt)
                    st.markdown(f"- Retrieved {len(nodes)} initial nodes")
                    st.markdown("**Reranking Process:**")
                    reranked_nodes = rerank_results(transformed_prompt, nodes)
                    
                    # Detailed Results Section
                    st.write("#### Retrieved Context (Reranked):")
                    for i, node in enumerate(reranked_nodes, 1):
                        st.markdown(f"""
                        **Document {i}:**
                        - Score: {node.score:.4f}
                        - Length: {len(node.text)} characters
                        ```
                        {node.text}
                        ```
                        ---
                        """)
                    
                    # Performance Metrics
                    st.write("#### Performance Metrics:")
                    st.markdown(f"""
                    - Total nodes processed: {len(nodes)}
                    - Final nodes selected: {len(reranked_nodes)}
                    - Average score: {sum(node.score for node in reranked_nodes)/len(reranked_nodes):.4f}
                    """)

                with st.chat_message("assistant"):
                    st.write("#### Final Response: ")
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Track response generation with token counting
                    if st.session_state["provider"] == "OpenRouter":
                        # For OpenRouter, we'll make a direct API call to get usage info
                        try:
                            api_key = st.session_state.get("openrouter_api_key")
                            headers = {
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            }
                            
                            # Build context for the query
                            messages = [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": transformed_prompt}
                            ]
                            
                            data = {
                                "model": st.session_state["model"],
                                "messages": messages,
                                "stream": True
                            }
                            
                            response = requests.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                headers=headers,
                                json=data,
                                stream=True,
                                timeout=120
                            )
                            
                            for line in response.iter_lines():
                                if line:
                                    line_text = line.decode('utf-8')
                                    if line_text.startswith('data: '):
                                        json_str = line_text[6:]
                                        if json_str.strip() == '[DONE]':
                                            break
                                        try:
                                            chunk_data = json.loads(json_str)
                                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                                delta = chunk_data['choices'][0].get('delta', {})
                                                content = delta.get('content', '')
                                                if content:
                                                    full_response += content
                                                    message_placeholder.markdown(full_response + "▌")
                                            
                                            # Check for usage data in the final chunk
                                            if 'usage' in chunk_data:
                                                usage = chunk_data['usage']
                                                st.session_state.total_prompt_tokens += usage.get('prompt_tokens', 0)
                                                completion = usage.get('completion_tokens', 0)
                                                st.session_state.total_completion_tokens += completion
                                                
                                                # Calculate cost
                                                if st.session_state.openrouter_models:
                                                    pricing = get_model_pricing(
                                                        st.session_state["model"],
                                                        st.session_state.openrouter_models
                                                    )
                                                    cost = calculate_cost(
                                                        usage.get('prompt_tokens', 0),
                                                        completion,
                                                        pricing
                                                    )
                                                    st.session_state.total_cost += cost
                                        except json.JSONDecodeError:
                                            pass
                            
                            message_placeholder.markdown(full_response)
                            
                            # Display token info for this response
                            if full_response:
                                response_tokens = count_tokens(full_response)
                                st.caption(f"📊 This response: ~{response_tokens:,} tokens")
                        
                        except Exception as e:
                            st.error(f"Error generating response: {e}")
                            # Fallback to regular chat engine
                            response = st.session_state.chat_engine.stream_chat(transformed_prompt)
                            for chunk in response.response_gen:
                                full_response += chunk
                                message_placeholder.markdown(full_response + "▌")
                            message_placeholder.markdown(full_response)
                    else:
                        # Use Ollama chat engine
                        response = st.session_state.chat_engine.stream_chat(transformed_prompt)
                        for chunk in response.response_gen:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)
                        
                        # Estimate token count for Ollama
                        response_tokens = count_tokens(full_response)
                        st.session_state.total_prompt_tokens += prompt_tokens
                        st.session_state.total_completion_tokens += response_tokens
                        st.caption(f"📊 Estimated tokens: ~{response_tokens:,} (Ollama doesn't provide exact counts)")
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        st.markdown("<span style='font-size:15px;'><b>Upload a PDF to start chatting</span>", unsafe_allow_html=True)

if __name__=='__main__':
    main()