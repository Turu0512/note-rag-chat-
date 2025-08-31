import os
import glob
from typing import List

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
COLLECTION  = os.environ.get("CHROMA_COLLECTION", "note_articles")
ARTICLES_DIR = os.environ.get("ARTICLES_DIR", "./articles")
FORCE_REINDEX = os.environ.get("FORCE_REINDEX", "0") == "1"


def main() -> None:
    client = PersistentClient(path=PERSIST_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"embedding_model": EMBED_MODEL})

    existing = col.count()
    if existing > 0 and not FORCE_REINDEX:
        print(f"[embed] 既存インデックスを使用: {existing} 件（model={EMBED_MODEL}）")
        return
    if existing > 0 and FORCE_REINDEX:
        col.delete(where={})
        print(f"[embed] 既存 {existing} 件を削除して再作成します…")

    paths = sorted(glob.glob(os.path.join(ARTICLES_DIR, "*.txt")))
    if not paths:
        print(f"[embed] 入力記事がありません: {ARTICLES_DIR}")
        return

    model = SentenceTransformer(EMBED_MODEL)
    texts: List[str] = []
    ids:   List[str] = []
    metas: List[dict] = []

    for p in paths:
        with open(p, encoding="utf-8") as f:
            t = f.read().strip()
        if not t:
            continue
        fname = os.path.basename(p)
        texts.append(t)
        ids.append(fname)
        metas.append({"filename": fname})

    embs = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()
    col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
    print(f"[embed] 埋め込み処理 完了: {len(ids)} 件。")


if __name__ == "__main__":
    main()
