# tools
import glob
from tempfile import NamedTemporaryFile
import clipboard
import tempfile
import os
# ollama
import ollama
from llama_index.vector_stores.faiss import FaissVectorStore
# llamaindex
from llama_index.llms.ollama import Ollama
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
# ollama
import streamlit as st

nest_asyncio.apply()

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
    
    # Get transformed query from LLM
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
def load_data(documents, model_name:str) -> VectorStoreIndex:
    """Loads and indexes Streamlit documentation using Ollama and Llamaindex.

    This function takes a model name as input and performs the following actions:

    1. Ollama Initialization: Initializes an Ollama instance using the provided model name. Ollama is a library that facilitates communication with large language models (LLMs).
    2. Data Ingestion: Reads the Streamlit documentation (assumed to be a PDF file) using the SimpleDirectoryReader class.
    3. Text Splitting and Embedding: Splits the loaded documents into sentences using the SentenceSplitter class and generates embeddings for each sentence using the HuggingFaceEmbedding model.
    4. Service Context Creation: Creates a ServiceContext object that holds all the necessary components for processing the data, including the Ollama instance, embedding model, text splitter, and a system prompt for the LLM.
    5. VectorStore Indexing: Creates a VectorStoreIndex instance from the processed documents and the service context. VectorStore is a library for efficient searching of high-dimensional vectors.

    Args:
        # docs_path  (str): Path of the documents to query.
        model_name (str): The name of the LLM model to be used by Ollama.

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



    # llm
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

    with st.sidebar:
        # model selection
        if "model" not in st.session_state:
            st.session_state["model"] = ""
        models = [model["name"] for model in ollama.list()["models"]]
        st.session_state["model"] = st.selectbox("Select a model", models)
        
        # llm
        llm = Ollama(model=st.session_state["model"], n_gpu_layers=-1)

        # data ingestion
        documents = st.file_uploader("Upload PDF files to query (max 5)", type=['pdf'], accept_multiple_files=True)


        # file processing                
        if st.button('Process files'):
            if documents and len(documents) <= 5:


                index = load_data(documents, st.session_state["model"])
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
            elif len(documents) > 5:
                st.warning("Please upload a maximum of 5 files.")
            else:
                st.warning("Please upload at least one file.")

            

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
        if prompt := st.chat_input("How I can help you?"):
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
                    st.session_state["expander_state"] = False

                with st.chat_message("assistant"):
                    st.write("#### Final Response: ")
                    message_placeholder = st.empty()
                    full_response = ""
                    response = st.session_state.chat_engine.stream_chat(transformed_prompt)
                    for chunk in response.response_gen:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        st.markdown("<span style='font-size:15px;'><b>Upload a PDF to start chatting</span>", unsafe_allow_html=True)

if __name__=='__main__':
    main()