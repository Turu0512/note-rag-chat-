import os
import glob
import logging
from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
ARTICLES_DIR = os.environ.get("ARTICLES_DIR", "./articles")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "note_articles")

def main():
    # 永続化を明示的に ON
    client = chromadb.Client(Settings(
        is_persistent=True,
        persist_directory=CHROMA_DIR,
        anonymized_telemetry=False,
    ))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 既存件数を確認
    try:
        current = collection.count()
    except Exception:
        current = 0
    if current > 0:
        logger.info(f"[embed] 既に {current} 件あるためスキップします。")
        return

    # 埋め込みモデル
    logger.info(f"[embed] Use SentenceTransformer: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    ids: List[str] = []
    metadatas = []
    documents = []
    embeddings = []

    paths = sorted(glob.glob(os.path.join(ARTICLES_DIR, "*.txt")))
    if not paths:
        logger.warning(f"[embed] 文章ファイルが見つかりません: {ARTICLES_DIR}/*.txt")
        return

    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = (f.read() or "").strip()
        if not text:
            continue
        vec = model.encode([text])[0].tolist()
        fname = os.path.basename(path)

        ids.append(fname)
        documents.append(text)
        embeddings.append(vec)
        metadatas.append({"filename": fname})

    if not ids:
        logger.warning("[embed] 追加対象がありませんでした。")
        return

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 反映確認
    try:
        total = collection.count()
    except Exception:
        total = -1
    logger.info(f"[embed] 埋め込み処理 完了: {total} 件。")

if __name__ == "__main__":
    main()
