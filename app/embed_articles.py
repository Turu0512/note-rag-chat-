# embed_articles.py

import glob
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb import PersistentClient

client = PersistentClient(
    path="./chroma_db",                # ← ここがディレクトリ
    settings=Settings(),               # ← デフォルト設定で OK
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
# コレクションがなければ作成
collection = client.get_or_create_collection("note_articles")

# 日本語クエリ/日本語コーパス向けの多言語モデル
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

for filepath in glob.glob("articles/*.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    emb = model.encode([text])[0]
    collection.add(
        ids=[filepath],
        embeddings=[emb.tolist()],
        documents=[text],
        metadatas=[{"filename": filepath}],
    )
    print(f"登録: {filepath}")

print("全記事をChromaDBへ登録完了！")
