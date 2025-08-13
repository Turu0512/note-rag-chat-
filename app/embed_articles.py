import os
import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def main():
    # ChromaDB クライアントを初期化（persist_directory は Docker コンテナ内パス）
    client = chromadb.Client(Settings(
        persist_directory="./chroma_db"
    ))

    # コレクションを取得 or 作成
    existing = [c.name for c in client.list_collections()]
    if "note_articles" in existing:
        collection = client.get_collection("note_articles")
    else:
        collection = client.create_collection("note_articles")

    # 埋め込み済み件数を取得（count() は int または dict を返す可能性がある）
    raw_count = collection.count()
    count = raw_count.get("count") if isinstance(raw_count, dict) else raw_count
    if count and count > 0:
        print(f"[embed] note_articles コレクションに既に {count} 件の埋め込みがあります。処理をスキップします。")
        return

    # 埋め込みモデルをロード
    model = SentenceTransformer("all-MiniLM-L6-v2")
    articles_dir = "./articles"

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    # テキストファイル毎に埋め込みを作成
    for path in glob.glob(os.path.join(articles_dir, "*.txt")):
        fname = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        emb = model.encode([text])[0]
        ids.append(fname)
        embeddings.append(emb.tolist())
        documents.append(text)
        metadatas.append({"filename": fname})

    # バルクでコレクションに追加
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

    print(f"[embed] 埋め込み処理 完了: {len(ids)} 件。")


if __name__ == "__main__":
    main()
