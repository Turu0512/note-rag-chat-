import os
import logging
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, Request
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ログ
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 環境変数
OPENAI_MODEL  = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PERSIST_DIR   = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION    = os.environ.get("CHROMA_COLLECTION", "note_articles")

# Chroma（新方式へ統一）
client = PersistentClient(path=PERSIST_DIR)
try:
    collection = client.get_collection(COLLECTION)
except Exception:
    log.warning(f"{COLLECTION} コレクションが見つからず作成します（embed_articles.py が先に走る想定）")
    collection = client.get_or_create_collection(COLLECTION)

count = collection.count()
log.info(f"ChromaDB ready. collection={COLLECTION}, count={count}, dir={os.path.abspath(PERSIST_DIR)}")

# 埋め込み（クエリ用・記事作成時と同じモデル）
log.info(f"Use SentenceTransformer: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# OpenAI
oai = OpenAI()
log.info(f"使用モデル: {OPENAI_MODEL}")

app = FastAPI()


def vector_search(q: str, k: int = 5) -> Tuple[List[str], List[str], List[dict], List[float]]:
    q_emb = embedder.encode([q])[0].tolist()
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # ← ids は include に入れない
    )
    docs = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    dists = res.get("distances", [[]])[0] or []
    ids   = (res.get("ids", [[]])[0] or [])  # ids は include しなくても返る
    log.info(f"[vector] hits={len(docs)} dists={dists[:3]}")
    return ids, docs, metas, dists


@app.post("/query")
async def query(request: Request):
    body = await request.body()
    log.info(f"Incoming request body: {body.decode('utf-8')}")
    try:
        payload  = await request.json()
        question = (payload.get("question") or "").strip()
        log.info(f"Parsed question: {question}")
        if not question:
            raise ValueError("質問が空です。")
    except Exception:
        raise HTTPException(status_code=400, detail="不正なリクエストです。")

    ids, docs, metas, dists = vector_search(question, k=5)
    if not docs:
        raise HTTPException(status_code=404, detail="関連記事が見つかりませんでした")

    # LLM へ渡す文脈を整形
    snippets = []
    for i, doc in enumerate(docs):
        fn = (metas[i] or {}).get("filename") if i < len(metas) else None
        dist = f"(dist={dists[i]:.3f})" if i < len(dists) else ""
        header = f"[{i+1}] {fn or (ids[i] if i < len(ids) else 'doc')} {dist}"
        snippets.append(f"{header}\n{doc}")
    context = "\n\n---\n\n".join(snippets)

    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは検索補助を行うアシスタントです。"
                        "与えられた記事抜粋のみを根拠に、日本語で簡潔かつ具体的に答えてください。"
                        "根拠が弱い場合は『記事からは断定できません』と述べてください。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"質問: {question}\n\n"
                        "参考記事の抜粋（最大5件）:\n"
                        f"{context}\n\n"
                        "上の情報だけを使って回答してください。"
                    ),
                },
            ],
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        log.error(f"OpenAI API エラー: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API エラー: {e}")

    return {"answer": answer}
