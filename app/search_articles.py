# search_articles.py

import os
import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
COLLECTION  = os.environ.get("CHROMA_COLLECTION", "note_articles")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(COLLECTION)
model = SentenceTransformer(EMBED_MODEL)

query = input("検索したい内容を入力してください: ").strip()
if not query:
    print("空のクエリです。終了します。")
    raise SystemExit(0)

query_emb = model.encode([query])[0]
results = collection.query(query_embeddings=[query_emb.tolist()], n_results=3)

for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    soup = BeautifulSoup(doc, "html.parser")
    text = soup.get_text(separator="\n")
    print(f"\n--- {meta.get('filename','(no name)')} ---\n{text[:600]} ...")
