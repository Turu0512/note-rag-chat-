import os
import re
import logging
from typing import List
from fastapi import FastAPI, HTTPException, Request
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # ★ 新SDK

# ── ログ ─────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 環境変数 ─────────────────────────────────────────────────────
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR   = os.environ.get("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "note_articles")

# ── Chroma（永続化ONで初期化） ────────────────────────────────────
client_chroma = chromadb.Client(Settings(
    is_persistent=True,
    persist_directory=CHROMA_DIR,
    anonymized_telemetry=False,
))
names = [c.name for c in client_chroma.list_collections()]
if COLLECTION_NAME in names:
    collection = client_chroma.get_collection(COLLECTION_NAME)
else:
    logger.warning(f"{COLLECTION_NAME} コレクションが見つからなかったため空のコレクションを作成します。先に embed_articles.py を実行してください。")
    collection = client_chroma.create_collection(COLLECTION_NAME)

try:
    total = collection.count()
except Exception as e:
    logger.error(f"collection.count() でエラー: {e}")
    total = -1
logger.info(f"ChromaDB ready. collection={COLLECTION_NAME}, count={total}, dir={os.path.abspath(CHROMA_DIR)}")

# ── 埋め込みモデル（検索用） ───────────────────────────────────────
logger.info(f"Use SentenceTransformer: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# ── OpenAI クライアント（新SDK） ─────────────────────────────────
oai = OpenAI()  # OPENAI_API_KEY は環境変数で
logger.info(f"使用モデル: {OPENAI_MODEL}")

# ── FastAPI ─────────────────────────────────────────────────────
app = FastAPI()

def _extract_keywords_jp(q: str, max_k: int = 4) -> List[str]:
    parts = re.split(r"[、。！？\s/・,.;:()\[\]{}「」『』\"'…]+", q)
    parts = [p for p in parts if len(p) >= 2]
    stop = {"について", "教えて", "ブログ", "記事", "ください", "こと", "もの"}
    kws = []
    for p in parts:
        if p in stop:
            continue
        if p not in kws:
            kws.append(p)
    return kws[:max_k] or [q]

@app.post("/query")
async def query(request: Request):
    body = await request.body()
    logger.info(f"Incoming request body: {body.decode('utf-8')}")

    try:
        payload = await request.json()
        question = payload.get("question", "").strip()
        logger.info(f"Parsed question: {question}")
        if not question:
            raise ValueError("質問が空です。")
    except Exception as e:
        logger.error(f"Request parsing error: {e}")
        raise HTTPException(status_code=400, detail="不正なリクエストです。")

    # 1) ベクトル検索
    docs = []; metas = []; ids = []
    try:
        qvec = embedder.encode([question])[0].tolist()
        res = collection.query(
            query_embeddings=[qvec],
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        ids = (res.get("ids") or [[]])[0] if isinstance(res.get("ids"), list) else []
        dists = res.get("distances", [[]])[0] or []
        logger.info(f"[vector] hits={len(docs)} dists={dists[:3]}")
    except Exception as e:
        logger.error(f"Chroma vector query error: {e}")

    # 2) 0件なら contains 部分一致
    if not docs:
        try:
            kws = _extract_keywords_jp(question)
            logger.info(f"[contains] keywords={kws}")
            found_docs, found_metas, found_ids = [], [], []
            for kw in kws:
                r = collection.get(where_document={"$contains": kw}, limit=5)
                if r.get("documents"):
                    found_docs.extend(r["documents"])
                    found_metas.extend(r.get("metadatas", [{}] * len(r["documents"])))
                    found_ids.extend(r.get("ids", [""] * len(r["documents"])))
                if len(found_docs) >= 5:
                    break
            docs = found_docs[:5]; metas = found_metas[:5]; ids = found_ids[:5]
            logger.info(f"[contains] hits={len(docs)}")
        except Exception as e:
            logger.error(f"Chroma contains-get error: {e}")

    # 3) それでも0なら直近5件を返す
    if not docs:
        try:
            r = collection.get(limit=5)
            docs = r.get("documents", [])[:5]
            metas = r.get("metadatas", [])[:5]
            ids   = r.get("ids", [])[:5]
            logger.info(f"[fallback get] hits={len(docs)}")
        except Exception as e:
            logger.error(f"Chroma last-resort get error: {e}")

    if not docs:
        logger.info("関連記事が見つかりませんでした。")
        raise HTTPException(status_code=404, detail="関連記事が見つかりませんでした")

    context = "\n\n".join(docs if isinstance(docs, list) else [docs])

    # ── OpenAI Chat（新SDK） ───────────────────────────────────────
    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system",  "content": "記事の内容をもとに日本語で簡潔に回答してください。必要なら記事のファイル名も括弧で添えてください。"},
                {"role": "user",    "content": f"質問: {question}\n\n関連記事:\n{context}"},
            ],
            temperature=0.3,
        )
        answer = resp.choices[0].message.content
        logger.info("OpenAI API call succeeded.")
    except Exception as e:
        logger.error(f"OpenAI API エラー: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API エラー: {e}")

    return {"answer": answer}
