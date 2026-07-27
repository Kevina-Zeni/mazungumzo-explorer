import streamlit as st
import json, os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

APP_DATA_DIR = os.path.join(os.path.dirname(__file__), "app_data")
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "chunks")

SHOW_DESCRIPTION = (
    "Mazungumzo: African Scholarly Conversations is a podcast that highlights the "
    "perspectives of various stakeholders in academia and research fields across Africa "
    "through open dialogue on scholarly communication in Africa. It is hosted by Joy Owango, "
    "Executive Director of the Training Centre in Communication (TCC Africa)."
)

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

@st.cache_resource
def load_manifest():
    with open(os.path.join(APP_DATA_DIR, "manifest.json")) as f:
        return json.load(f)

@st.cache_data
def load_turns(ep_id):
    with open(os.path.join(APP_DATA_DIR, f"{ep_id}_turns.json")) as f:
        return json.load(f)

@st.cache_data
def load_topics(ep_id):
    with open(os.path.join(APP_DATA_DIR, f"{ep_id}_topics.json")) as f:
        return json.load(f)

@st.cache_data
def load_keywords(ep_id):
    with open(os.path.join(APP_DATA_DIR, f"{ep_id}_keywords.json")) as f:
        return json.load(f)

@st.cache_resource
def load_vector_store(ep_id):
    with open(os.path.join(CHUNKS_DIR, f"{ep_id}_chunks.json")) as f:
        chunks = json.load(f)
    docs = [Document(page_content=c, metadata={"source": ep_id}) for c in chunks]
    docs.append(Document(page_content=SHOW_DESCRIPTION, metadata={"source": "show_description", "type": "meta"}))
    embedding_model = load_embedding_model()
    store = Chroma.from_documents(
        documents=docs, embedding=embedding_model,
        collection_name=f"{ep_id}_runtime",
        collection_metadata={"hnsw:space": "cosine"},
    )
    return store, chunks

@st.cache_resource
def load_bm25(ep_id):
    _, chunks = load_vector_store(ep_id)
    tokenized = [c.lower().split() for c in chunks]
    return {"index": BM25Okapi(tokenized), "chunks": chunks}

def hybrid_retrieve(question, dense_store, bm25_data, k=3, rrf_k=60):
    dense_results = dense_store.similarity_search_with_score(question, k=10)
    dense_ranked = [doc.page_content for doc, score in dense_results]
    tokenized_query = question.lower().split()
    bm25_scores = bm25_data["index"].get_scores(tokenized_query)
    bm25_ranked_idx = sorted(range(len(bm25_scores)), key=lambda i: -bm25_scores[i])[:10]
    bm25_ranked = [bm25_data["chunks"][i] for i in bm25_ranked_idx]
    rrf_scores = {}
    for rank, content in enumerate(dense_ranked, start=1):
        rrf_scores[content] = rrf_scores.get(content, 0) + 1.0 / (rrf_k + rank)
    for rank, content in enumerate(bm25_ranked, start=1):
        rrf_scores[content] = rrf_scores.get(content, 0) + 1.0 / (rrf_k + rank)
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
    return [content for content, score in fused]
