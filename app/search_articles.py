import chromadb
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

# 新クライアントで永続ストレージ指定（絶対パスでも可）
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("note_articles")
model = SentenceTransformer("all-MiniLM-L6-v2")

query = input("検索したい内容を入力してください: ")
query_emb = model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_emb.tolist()],
    n_results=3
)
for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
    # HTML→テキスト
    soup = BeautifulSoup(doc, "html.parser")
    text = soup.get_text(separator="\n")
    print(f"\n--- {meta['filename']} ---\n{text[:300]} ...")
