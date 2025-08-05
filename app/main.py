import os
import logging
from fastapi import FastAPI, HTTPException, Request
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai

# ——— ログ設定 ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ——— ChromaDB クライアント ———
client = chromadb.Client(Settings(
    persist_directory="./chroma_db"
))
# コレクションがなければ作成
collection = client.get_or_create_collection("note_articles")

# ——— 質問の埋め込みモデル ———
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ——— FastAPI アプリ ———
app = FastAPI()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
logger.info(f"使用モデル: {OPENAI_MODEL}")

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

    # 質問をベクトル化してクエリ
    q_emb = embed_model.encode([question])[0].tolist()
    docs = collection.query(
        query_embeddings=[q_emb],
        n_results=5
    )["documents"][0]
    
    if not docs:
        logger.info("関連記事が見つかりませんでした。")
        raise HTTPException(status_code=404, detail="関連記事が見つかりませんでした")
    context = "\n\n".join(docs)
    logger.info(f"Retrieved {len(docs)} documents from ChromaDB.")

    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system",  "content": "記事の内容をもとに回答してください。"},
                {"role": "user",    "content": question + "\n\n" + context},
            ],
            temperature=0.7,
        )
        answer = resp.choices[0].message.content
        logger.info("OpenAI API call succeeded.")
    except Exception as e:
        logger.error(f"OpenAI API エラー: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API エラー: {e}")

    return {"answer": answer}
