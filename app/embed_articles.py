import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 新方式：persist_directory のみ指定
client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))

# コレクションを取得 or 作成
names = [c.name for c in client.list_collections()]
if "note_articles" in names:
    collection = client.get_collection("note_articles")
else:
    collection = client.create_collection("note_articles")

model = SentenceTransformer("all-MiniLM-L6-v2")

metadatas, embeddings, documents, ids = [], [], [], []
for path in glob.glob("articles/*.txt"):
    text = open(path, encoding="utf-8").read()
    emb = model.encode([text])[0]
    embeddings.append(emb.tolist())
    documents.append(text)
    fname = os.path.basename(path)
    metadatas.append({"filename": fname})
    ids.append(fname)

collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("全記事をChromaDBへ登録完了！")
