import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_collection("note_articles")
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Note記事検索Bot")
query = st.text_input("知りたいことを入力してください")
if query:
    query_emb = model.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=3
    )
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        st.write(f"### {meta['filename']}")
        st.write(doc[:500])
