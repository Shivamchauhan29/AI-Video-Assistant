import os
import uuid
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.pipeline_logger import log_if

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : 'cpu'}
    )

def build_vector_store(transcript : str, logger=None)->Chroma:
    print("Building vector Store")
    log_if(logger, "Vector Store", "Splitting transcript for embedding...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 100
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata = {'chunk_index' : i})
        for i,chunk in enumerate(chunks)
    ]

    log_if(logger, "Vector Store", f"Embedding {len(docs)} chunk(s)...")
    embeddings = get_embeddings()

    # In-memory (no persist_directory) with a per-call collection name:
    # a shared collection name still leaks documents across calls even
    # without disk persistence, since Chroma's default in-memory backend
    # is keyed by collection name at the process level. Without this,
    # concurrent runs (different sessions/videos) corrupt or contaminate
    # each other's RAG retrieval.
    vector_store = Chroma.from_documents(
        documents= docs,
        embedding=embeddings,
        collection_name=f"{COLLECTION_NAME}_{uuid.uuid4().hex}",
    )

    log_if(logger, "Vector Store", "Vector store ready")
    return vector_store



def load_vector_store() ->Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k :int = 10):
    return vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k":k}
    )

