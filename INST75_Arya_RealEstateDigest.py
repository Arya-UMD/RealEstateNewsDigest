import streamlit as st
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
import faiss

# Configuration
os.environ["COHERE_API_KEY"] = "WbjPWdMtqw1zVohxXBHnT8kUrJpDlFfmkXRtOxVm"  # Replace with your Cohere API key
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_SOURCE = "https://www.cnbc.com/2025/04/30/homebuyer-mortgage-demand-drops-further-as-economic-uncertainty-roils-the-housing-market.html"

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

def initialize_components():
    if st.session_state.llm is None:
        st.session_state.llm = Cohere(model="command-xlarge", temperature=0.3)

    if st.session_state.vector_store is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        dim = len(embeddings.embed_query("test"))

        index = faiss.IndexFlatIP(dim)

        st.session_state.vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=True
        )

def validate_metadata(docs):
    validated_docs = []
    for doc in docs:
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        
        if 'source' not in doc.metadata:
            doc.metadata['source'] = "Unknown Source"
        
        if isinstance(doc.metadata['source'], list):
            sources = [s for s in doc.metadata['source'] if isinstance(s, str)]
            doc.metadata['source'] = ", ".join(sources) if sources else "Unknown Source"
        
        source = doc.metadata['source']
        if isinstance(source, str):
            doc.metadata['source'] = source.split('?')[0]
        
        validated_docs.append(doc)
    return validated_docs

def process_urls(urls):
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store..."
    ef = st.session_state.vector_store.embedding_function
    dim = len(ef.embed_query("test"))
    st.session_state.vector_store.index = faiss.IndexFlatIP(dim)
    st.session_state.vector_store.docstore._dict.clear()
    st.session_state.vector_store.index_to_docstore_id.clear()

    yield "Loading data..."
    loader = UnstructuredURLLoader(
        urls=urls,
        continue_on_failure=True,
        mode="elements",
        strategy="fast",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    data = loader.load()

    yield "Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "(?<=\\. )", " ", ""]
    )
    docs = text_splitter.split_documents(data)

    yield "Processing metadata..."
    docs = validate_metadata(docs)
    filtered_docs = filter_complex_metadata(docs)

    yield "Adding to FAISS vector database..."
    st.session_state.vector_store.add_documents(filtered_docs)

    yield "Processing complete"
    st.session_state.processed = True

def generate_answer(query):
    if not st.session_state.processed:
        raise RuntimeError("Process URLs first")

    retriever = st.session_state.vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "lambda_mult": 0.5
        }
    )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=st.session_state.llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        reduce_k_below_max_tokens=True
    )
    
    result = chain({"question": query}, return_only_outputs=True)
    
    answer = result.get('answer', 'No answer found')
    sources = set()

    if 'source_documents' in result:
        for doc in result['source_documents']:
            if hasattr(doc, 'metadata'):
                source = doc.metadata.get('source', '')
                if any(source.startswith(p) for p in ('http://', 'https://')):
                    sources.add(source)
    
    # Return default source if none are available
    if not sources:
        sources = {DEFAULT_SOURCE}
    
    return answer, sorted(sources)

# Streamlit UI
st.title("Real Estate News Digest")

# Sidebar input
with st.sidebar:
    st.header("Configuration")
    urls = [
        st.text_input(f"URL {i+1}", value=url, key=f"url_{i}")
        for i, url in enumerate([
            "https://www.cnbc.com/video/2025/04/23/market-stability-depends-on-how-much-underlying-damage-tariff-rhetoric-did-citis-beata-manthey.html",
            "https://www.cnbc.com/2025/04/30/homebuyer-mortgage-demand-drops-further-as-economic-uncertainty-roils-the-housing-market.html",
            ""
        ])
    ]
    
    if st.button("Process URLs"):
        valid_urls = [url for url in urls if url.strip()]
        if not valid_urls:
            st.error("Please enter at least one valid URL")
        else:
            with st.status("Processing URLs...", expanded=True) as status:
                for progress in process_urls(valid_urls):
                    status.update(label=progress)
            st.success("Ready for questions!")

    if st.button("Reset System"):
        ef = st.session_state.vector_store.embedding_function
        dim = len(ef.embed_query("test"))
        st.session_state.vector_store.index = faiss.IndexFlatIP(dim)
        st.session_state.vector_store.docstore._dict.clear()
        st.session_state.vector_store.index_to_docstore_id.clear()
        st.session_state.processed = False
        st.rerun()

# Main interaction
query = st.text_input("Ask your real estate question", 
                     placeholder="e.g. What are the current housing market trends?")

if query:
    if not st.session_state.processed:
        st.error("Please process URLs first by clicking 'Process URLs' in the sidebar")
    else:
        with st.spinner("Searching for answers..."):
            try:
                answer, sources = generate_answer(query)
                
                st.subheader("Answer:")
                st.markdown(
                    f"<div style='background-color:#f0f2f6; padding:20px; border-radius:10px;'>{answer}</div>", 
                    unsafe_allow_html=True
                )
                
                st.subheader("Source Verification:")
                st.success("Information sourced from:")
                for url in sources:
                    st.markdown(f"🔗 [{url}]({url})")
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Info box
st.sidebar.markdown("---")
st.sidebar.info(
    "Enter at least one valid URL, click 'Process URLs', "
    "then ask questions about the content."
)
