# app.py

import os
import json
import requests
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ── 環境変数 ────────────────────────────────────────────────────────────────
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION  = os.environ.get("CHROMA_COLLECTION", "note_articles")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
API_URL     = os.environ.get("NOTE_RAG_API_URL", "http://localhost:8000/query")  # FastAPI /query

# ── キャッシュ化（起動高速化） ────────────────────────────────────────────
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION)

@st.cache_resource
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)

col = get_collection()
embedder = get_embedder()

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("Note記事検索Bot")

tab_llm, tab_vector = st.tabs(["💬 LLM回答（/query 使用）", "🔎 ベクトル検索（ローカル）"])

with tab_llm:
    st.caption("FastAPI の /query を呼び出して、構造化JSON（summary, points）で回答します。")
    q = st.text_input("知りたいことを入力してください（LLM）", key="llm_q")
    display_mode = st.radio("表示モード", ["Markdown", "プレーンテキスト"], horizontal=True)

    if q:
        try:
            resp = requests.post(API_URL, json={"question": q}, timeout=60)
            if resp.status_code != 200:
                st.error(f"APIエラー: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                summary = data.get("summary", "")
                points  = data.get("points", [])

                if display_mode == "Markdown":
                    st.markdown("### 要約")
                    st.markdown(summary)
                    if points:
                        st.markdown("### 重要ポイント")
                        st.markdown("\n".join([f"- {p}" for p in points]))
                else:
                    st.write("要約")
                    st.write(summary)
                    if points:
                        st.write("重要ポイント")
                        for p in points:
                            st.write(f"・{p}")
        except Exception as e:
            st.exception(e)

with tab_vector:
    st.caption("ChromaDBへ直接ベクトル検索します（LLMには投げません）。")
    q2 = st.text_input("検索したい内容を入力してください（ベクトル検索）", key="vec_q")
    topk = st.slider("件数", 1, 10, 3)
    if q2:
        q_emb = embedder.encode([q2])[0].tolist()
        res = col.query(query_embeddings=[q_emb], n_results=topk)
        docs   = (res.get("documents") or [[]])[0]
        metas  = (res.get("metadatas") or [[]])[0]

        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            st.write(f"### [{i}] {meta.get('filename','(no name)')}")
            st.text(doc[:1000])
