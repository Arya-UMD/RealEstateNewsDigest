import streamlit as st
import os
from uuid import uuid4
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Cohere
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np

# Configuration
os.environ["COHERE_API_KEY"] = ""
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

def initialize_components():
    if st.session_state.llm is None:
        st.session_state.llm = Cohere(model="command-xlarge", temperature=0)

    if st.session_state.vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        
        dim = len(ef.embed_query("sample"))
        index = faiss.IndexFlatIP(dim)  # Dot product similarity

        st.session_state.vector_store = FAISS(
            embedding_function=ef,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            normalize_L2=False  # Do not normalize; we want raw dot product
        )

def validate_metadata(docs):
    validated_docs = []
    for doc in docs:
        if not hasattr(doc, 'metadata'):
            doc.metadata = {}
        if 'source' not in doc.metadata:
            doc.metadata['source'] = "Unknown Source"
        if isinstance(doc.metadata['source'], list):
            doc.metadata['source'] = ", ".join(doc.metadata['source'])
        validated_docs.append(doc)
    return validated_docs

def process_urls(urls):
    yield "Initializing Components"
    initialize_components()

    yield "Resetting vector store..."
    ef = st.session_state.vector_store.embedding_function
    dim = len(ef.embed_query("sample"))
    st.session_state.vector_store.index = faiss.IndexFlatIP(dim)
    st.session_state.vector_store.docstore._dict.clear()
    st.session_state.vector_store.index_to_docstore_id.clear()

    yield "Loading data..."
    loader = UnstructuredURLLoader(
        urls=urls,
        continue_on_failure=True,
        mode="elements",
        strategy="fast"
    )
    data = loader.load()

    yield "Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Processing metadata..."
    docs = validate_metadata(docs)
    filtered_docs = filter_complex_metadata(docs)

    yield "Adding to FAISS vector database..."
    st.session_state.vector_store.add_documents(filtered_docs)

    yield "✅ Processing complete"
    st.session_state.processed = True

def generate_answer(query):
    if not st.session_state.processed:
        raise RuntimeError("You must process URLs first")

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=st.session_state.llm,
        retriever=st.session_state.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        ),
        return_source_documents=True
    )
    
    result = chain({"question": query}, return_only_outputs=True)
    
    answer = result.get('answer', 'No direct answer found')
    
    # Extract and verify sources
    source_urls = set()
    if 'source_documents' in result:
        for doc in result['source_documents']:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                source = doc.metadata['source']
                if source.startswith(('http://', 'https://')):
                    source_urls.add(source)
    
    return answer, list(source_urls) if source_urls else ["No verifiable sources found"]

# Streamlit UI
st.title("Real Estate News Digest")

# URL inputs
urls = [
    st.sidebar.text_input(f"URL {i+1}", value=url, key=f"url_{i}")
    for i, url in enumerate([
        "https://www.cnbc.com/video/2025/04/1",
        "https://www.realtor.com/news/trends/w",
        ""
    ])
]

if st.sidebar.button("Process URLs"):
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.error("Please enter at least one valid URL")
    else:
        with st.status("Processing URLs...", expanded=True) as status:
            for progress in process_urls(valid_urls):
                status.update(label=progress)
        st.success("Ready for questions!")

query = st.text_input("Ask your real estate question")
if query:
    if not st.session_state.processed:
        st.error("Please process URLs first")
    else:
        try:
            answer, sources = generate_answer(query)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Source Verification:")
            if "No verifiable sources found" in sources:
                st.warning("No source URLs could be verified")
            else:
                st.info("This information came from:")
                for url in sources:
                    st.markdown(f"- [{url}]({url})")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.sidebar.button("Reset System"):
    ef = st.session_state.vector_store.embedding_function
    dim = len(ef.embed_query("sample"))
    st.session_state.vector_store.index = faiss.IndexFlatIP(dim)
    st.session_state.vector_store.docstore._dict.clear()
    st.session_state.vector_store.index_to_docstore_id.clear()
    st.session_state.processed = False
    st.rerun()
