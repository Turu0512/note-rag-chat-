import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = client.get_or_create_collection("note_articles")

model = SentenceTransformer("all-MiniLM-L6-v2")

for filepath in glob.glob("articles/*.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    embedding = model.encode([text])[0]
    collection.add(
        ids=[filepath],
        embeddings=[embedding.tolist()],
        documents=[text],
        metadatas=[{"filename": filepath}]
    )
    print(f"登録: {filepath}")

client.persist()
print("全記事をChromaDBへ登録完了！")
