import os
import json
import logging
from typing import List, Tuple

from fastapi import FastAPI, HTTPException, Request
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 環境変数 ────────────────────────────────────────────────────────────────
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PERSIST_DIR  = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL  = os.environ.get(
    "EMBED_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
COLLECTION   = os.environ.get("CHROMA_COLLECTION", "note_articles")

# ── Chroma（PersistentClient で統一） ────────────────────────────────────
client = PersistentClient(path=PERSIST_DIR)

def _get_collection():
    """常に最新のコレクションハンドルを取得（embed後の再作成にも追従）"""
    return client.get_or_create_collection(COLLECTION)

# 起動ログ
try:
    _cnt = _get_collection().count()
    log.info(f"ChromaDB ready. collection={COLLECTION}, count={_cnt}, dir={os.path.abspath(PERSIST_DIR)}")
except Exception as e:
    log.warning(f"Chroma 初期化時に count 取得で例外: {e}")

# ── SentenceTransformer（検索時も embed と同一モデル） ──────────────────
log.info(f"Use SentenceTransformer: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# ── OpenAI ────────────────────────────────────────────────────────────────
oai = OpenAI()
log.info(f"使用モデル: {OPENAI_MODEL}")

# ── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI()


def vector_search(q: str, k: int = 5) -> Tuple[List[str], List[str], List[dict], List[float]]:
    """クエリ文字列 q に対してベクトル検索を行い、候補を返す。"""
    q_emb = embedder.encode([q])[0].tolist()
    coll = _get_collection()
    try:
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["ids", "documents", "metadatas", "distances"],
        )
    except NotFoundError:
        # embed 側でコレクションが再作成された直後など
        coll = _get_collection()
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["ids", "documents", "metadatas", "distances"],
        )

    docs  = res.get("documents", [[]])[0] or []
    metas = res.get("metadatas", [[]])[0] or []
    dists = res.get("distances", [[]])[0] or []
    ids   = (res.get("ids", [[]])[0] or [])
    log.info(f"[vector] hits={len(docs)} dists={dists[:3]}")
    return ids, docs, metas, dists


@app.get("/health")
async def health():
    coll = _get_collection()
    return {"status": "ok", "chroma_count": coll.count(), "embed_model": EMBED_MODEL}


@app.post("/query")
async def query(request: Request):
    """
    参考記事の抜粋だけを根拠に LLM が回答するエンドポイント。
    返り値は厳密な JSON スキーマ（summary, points）に固定。
    """
    body_bytes = await request.body()
    log.info(f"Incoming request body: {body_bytes.decode('utf-8', errors='ignore')}")
    try:
        payload  = await request.json()
        question = (payload.get("question") or "").strip()
        log.info(f"Parsed question: {question}")
        if not question:
            raise ValueError("質問が空です。")
    except Exception:
        raise HTTPException(status_code=400, detail="不正なリクエストです。JSONに 'question' を含めてください。")

    # ベクトル検索して文脈を整形
    ids, docs, metas, dists = vector_search(question, k=5)
    if not docs:
        raise HTTPException(status_code=404, detail="関連記事が見つかりませんでした")

    snippets = []
    for i, doc in enumerate(docs):
        fn   = (metas[i] or {}).get("filename") if i < len(metas) else None
        dist = f"(dist={dists[i]:.3f})" if i < len(dists) else ""
        header = f"[{i+1}] {fn or (ids[i] if i < len(ids) else 'doc')} {dist}"
        snippets.append(f"{header}\n{doc}")
    context = "\n\n---\n\n".join(snippets)

    # 出力を“純JSON”に固定するためのスキーマ
    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "要約（プレーンテキスト、Markdownや装飾は使わない）"
            },
            "points": {
                "type": "array",
                "description": "重要ポイント（プレーンテキスト）",
                "items": {"type": "string"}
            }
        },
        "required": ["summary", "points"],
        "additionalProperties": False
    }

    try:
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            response_format={  # ここで“厳密JSON”を要求
                "type": "json_schema",
                "json_schema": {
                    "name": "note_answer",
                    "schema": schema,
                    "strict": True
                }
            },
            messages=[
                {
                    "role": "system",
                    "content": (
                        "あなたは検索補助を行うアシスタントです。"
                        "以下の参考記事抜粋『のみ』を根拠に日本語で簡潔かつ具体的に回答してください。"
                        "出力は必ず指定のJSONスキーマ（summary, points）に一致させ、"
                        "Markdownの記号（*, _, ~, `, #, -, +, 数字. など）や絵文字は使わないでください。"
                        "箇条書きが必要な場合は points に文字列の配列として入れてください。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"質問: {question}\n\n"
                        "参考記事の抜粋（最大5件）:\n"
                        f"{context}\n\n"
                        "上の情報だけを使って回答し、プレーンテキストの JSON を返してください。"
                    ),
                },
            ],
        )
        # response_format=json_schema の場合、content は厳密JSON文字列で返ってくる
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        log.error(f"OpenAI API エラー: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API エラー: {e}")

    return data
