# main.py

import os
import json
import logging
from typing import List, Tuple, Dict, Any

import requests  # ← 追加（Ollama用）
from fastapi import FastAPI, HTTPException, Request
from chromadb import PersistentClient
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # .env 読み込み

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── 環境変数 ────────────────────────────────────────────────────────────────
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PERSIST_DIR  = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
COLLECTION   = os.environ.get("CHROMA_COLLECTION", "note_articles")

# コンテキスト制御
MAX_DOCS      = int(os.environ.get("MAX_DOCS", "10"))
MAX_DOC_CHARS = int(os.environ.get("MAX_DOC_CHARS", "1200"))

# 追加: LLMバックエンド切替（openai / ollama）
LLM_BACKEND  = os.environ.get("LLM_BACKEND", "openai")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL    = os.environ.get("LLM_MODEL", "gpt-oss:20b")

# ── Chroma ────────────────────────────────────────────────────────────────
client = PersistentClient(path=PERSIST_DIR)

def _get_collection():
    """embed による drop/recreate 後でも常に最新のコレクションを掴む"""
    return client.get_or_create_collection(COLLECTION)

try:
    _cnt = _get_collection().count()
    log.info(f"ChromaDB ready. collection={COLLECTION}, count={_cnt}, dir={os.path.abspath(PERSIST_DIR)}")
except Exception as e:
    log.warning(f"Chroma 初期化時に count 取得で例外: {e}")

# ── Embedding ─────────────────────────────────────────────────────────────
log.info(f"Use SentenceTransformer: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)

# ── OpenAI ────────────────────────────────────────────────────────────────
oai = OpenAI()
log.info(f"使用モデル(OPENAI): {OPENAI_MODEL}")
log.info(f"LLM_BACKEND={LLM_BACKEND}, LLM_MODEL={LLM_MODEL}, LLM_BASE_URL={LLM_BASE_URL}")

# ── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI()


def _clip(text: str, limit: int) -> str:
    if not text or limit <= 0:
        return text or ""
    return text[:limit]


def _collect_sources(metas: List[Dict[str, Any]], dists: List[float], ids: List[str]) -> List[Dict[str, Any]]:
    sources = []
    for i in range(min(len(metas), len(dists), len(ids))):
        m = metas[i] or {}
        sources.append({
            "id": ids[i],
            "filename": m.get("filename") or ids[i],
            "distance": float(dists[i]) if i < len(dists) else None,
            "url": m.get("source_canonical"),
            "published_at": m.get("published_at"),
            "updated_at": m.get("updated_at"),
        })
    return sources


def vector_search(q: str, k: int = 5) -> Tuple[List[str], List[str], List[dict], List[float]]:
    """クエリ文字列 q に対してベクトル検索を行い、候補を返す。"""
    q_emb = embedder.encode([q])[0].tolist()
    coll = _get_collection()
    try:
        # ※ include に 'ids' は入れない（現行 Chroma は非対応）
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    except NotFoundError:
        # embed 側でコレクションが再作成された直後など
        coll = _get_collection()
        res = coll.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

    docs  = (res.get("documents")  or [[]])[0] or []
    metas = (res.get("metadatas")  or [[]])[0] or []
    dists = (res.get("distances")  or [[]])[0] or []
    ids   = (res.get("ids")        or [[]])[0] or []  # ids はレスポンスに含まれる
    log.info(f"[vector] hits={len(docs)} dists={dists[:3]}")
    return ids, docs, metas, dists


def gen_json_answer(system_prompt: str, user_prompt: str, schema: dict) -> dict:
    """
    必ずJSONを返す共通ラッパ。
    - OpenAI: response_format=json_schema を使用
    - Ollama: /api/chat format=json + 強めの指示で JSON 強制
    """
    if LLM_BACKEND == "openai":
        resp = oai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.3,
            max_tokens=900,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "note_chatty_answer", "schema": schema, "strict": True}
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return json.loads(resp.choices[0].message.content)

    # ---- Ollama (gpt-oss) ----
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content":
                "以下の指示に厳密に従い、**JSONのみ**を出力してください。"
                "説明文や前置き、コードブロック記法は書かないでください。"
                "必ず次のJSONスキーマに一致させてください。\n"
                + json.dumps(schema, ensure_ascii=False)
            },
            {"role": "user", "content": user_prompt}
        ],
        "format": "json",
        "options": {"temperature": 0.3}
    }
    r = requests.post(f"{LLM_BASE_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        # 万一JSONが壊れた場合の簡易救済
        import re as _re
        m = _re.search(r"\{.*\}", content, _re.S)
        if not m:
            raise
        return json.loads(m.group(0))


@app.get("/health")
async def health():
    coll = _get_collection()
    return {"status": "ok", "chroma_count": coll.count(), "embed_model": EMBED_MODEL}


@app.post("/query")
async def query(request: Request):
    """
    参考記事抜粋だけを根拠に、ChatGPTらしい自然な「回答（Markdown）」と
    追加で役立つ「suggestions（任意だが、スキーマ上は空配列でも必ず含める）」を返す。
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

    # ベクトル検索して文脈を整形（長すぎるスニペットはクリップ）
    ids, docs, metas, dists = vector_search(question, k=MAX_DOCS)
    if not docs:
        raise HTTPException(status_code=404, detail="関連記事が見つかりませんでした")

    snippets = []
    for i, doc in enumerate(docs):
        fn   = (metas[i] or {}).get("filename") if i < len(metas) else None
        dist = f"(dist={dists[i]:.3f})" if i < len(dists) else ""
        header = f"[{i+1}] {fn or (ids[i] if i < len(ids) else 'doc')} {dist}"
        snippet = _clip(doc, MAX_DOC_CHARS)
        snippets.append(f"{header}\n{snippet}")
    context = "\n\n---\n\n".join(snippets)

    # LLMに要求するスキーマ：answer（必須）、suggestions（必須だが空配列可）
    schema = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": (
                    "ユーザーの質問に対する最終回答（日本語・Markdown可）。"
                    "ブログの文脈から、どこで何をしたか等の具体性を優先し、"
                    "単語羅列ではなく、読み物として滑らかな文にする。"
                    "必要なら見出しや短い箇条書きを使ってもよい。"
                    "事実が不明な点は推測せず『記事からは断定できません』と明記する。"
                )
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 0,
                "description": "ユーザーが次に掘り下げられる観点（例：場所の詳細、費用、持ち物、季節の注意）。空配列でも可。"
            }
        },
        "required": ["answer", "suggestions"],
        "additionalProperties": False
    }

    system_prompt = (
        "あなたはブログ要約・検索のアシスタントです。"
        "以下の参考記事抜粋『のみ』を根拠に、日本語で自然で読みやすい回答を作成してください。"
        "単語の羅列やパワーワードの寄せ集めは避け、具体的な場所・行動・流れを優先してまとめます。"
        "必要に応じて短い見出しや箇条書きを使って構いません。"
        "最後に、ユーザーが次に深掘りできる観点を 2–4 個ほど提案（suggestions）してください。"
        "不明点は推測せず『記事からは断定できません』と記してください。"
    )

    user_prompt = (
        f"質問: {question}\n\n"
        f"参考記事の抜粋（最大{MAX_DOCS}件）:\n"
        f"{context}\n\n"
        "注意:\n"
        "- 上の抜粋に含まれない情報は出さない。\n"
        "- 解答は日本語。マークダウン（見出し/箇条書き）可。\n"
        "- 具体的な場所や行動、出来事を優先して説明する。\n"
    )

    try:
        data = gen_json_answer(system_prompt, user_prompt, schema)
    except Exception as e:
        log.error(f"LLM エラー: {e}")
        raise HTTPException(status_code=500, detail=f"LLM エラー: {e}")

    sources = _collect_sources(metas, dists, ids)

    return {
        "answer": data.get("answer", ""),
        "suggestions": data.get("suggestions", []),
        "sources": sources,
    }
