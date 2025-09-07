# streamlit_app.py
import os
import json
import requests
import streamlit as st
from uuid import uuid4

API_URL = os.getenv("NOTE_RAG_API_URL", "http://localhost:8000/query")

st.set_page_config(page_title="Note記事検索Bot", page_icon="📝", layout="centered")
st.title("Note記事検索Bot")
st.caption(f"API: {API_URL}")

# ====== セッション状態 ======
if "history" not in st.session_state:
    # 各エントリに一意な "id" を持たせる
    st.session_state.history = []  # [{"id": str, "q": str, "answer": str, "suggestions": [str], "sources": [...]}]
if "pending" not in st.session_state:
    st.session_state.pending = None
if "use_chat_ui" not in st.session_state:
    st.session_state.use_chat_ui = True  # chat_message があれば使う

# ====== サイドバー ======
with st.sidebar:
    st.header("オプション")
    show_sources = st.checkbox("参考ソースを表示", value=True)
    if st.button("履歴をクリア"):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.write("・Enterで送信 / Shift+Enterで改行")
    st.write("・提案ボタンからワンクリックで再質問できます")

# ====== API 呼び出し ======
def call_api(question: str) -> dict:
    try:
        r = requests.post(API_URL, json={"question": question}, timeout=60)
    except requests.RequestException as e:
        st.error(f"APIに接続できませんでした: {e}")
        return {}
    if r.status_code != 200:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        st.error(f"APIエラー {r.status_code}: {detail}")
        return {}
    try:
        return r.json()
    except json.JSONDecodeError:
        st.error("APIから不正なJSONが返ってきました。")
        st.text(r.text)
        return {}

# ====== 表示ユーティリティ ======
def ensure_new_format(data: dict) -> dict:
    """旧 summary/points を新 answer/suggestions に寄せる"""
    if "answer" in data:
        return data
    answer = data.get("summary", "")
    pts = data.get("points") or []
    if pts:
        answer += ("\n\n" if answer else "") + "\n".join(f"- {p}" for p in pts)
    return {"answer": answer, "suggestions": [], "sources": []}

def render_sources(sources: list):
    if not sources:
        return
    with st.expander("参考ソース（近い順）"):
        for s in sources:
            fn = s.get("filename") or s.get("id") or "doc"
            url = s.get("url")
            dist = s.get("distance")
            meta = f"（距離 {dist:.3f}）" if isinstance(dist, (int, float)) else ""
            if url:
                st.markdown(f"- [{fn}]({url}) {meta}")
            else:
                st.markdown(f"- {fn} {meta}")

def render_suggestions(suggestions: list, msg_key: str):
    """メッセージ固有のキー接頭辞でボタンのkey重複を防ぐ"""
    if not suggestions:
        return
    st.caption("次に深掘りできます👇")
    num_cols = min(3, len(suggestions))
    cols = st.columns(num_cols)
    for i, s in enumerate(suggestions):
        col = cols[i % num_cols]
        if col.button(s, key=f"sugg_{msg_key}_{i}"):
            st.session_state.pending = s
            st.rerun()

# ====== チャット表示 ======
def render_chat():
    use_chat = hasattr(st, "chat_message") and st.session_state.use_chat_ui
    for idx, h in enumerate(st.session_state.history):
        # 既存履歴に id が無い場合は付与して永続化
        if "id" not in h:
            h["id"] = uuid4().hex
        msg_key = h["id"]

        q = h["q"]
        a = h.get("answer", "")
        sugg = h.get("suggestions") or []
        srcs = h.get("sources") or []

        if use_chat:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                st.markdown(a or "_（回答なし）_")
                if sugg:
                    render_suggestions(sugg, msg_key=msg_key)
                if show_sources:
                    render_sources(srcs)
                with st.expander("この回答を踏まえて再質問する"):
                    follow = st.text_input(f"follow_{msg_key}", label_visibility="collapsed",
                                           placeholder="例）どこの海岸で入った？")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button("送信", key=f"follow_btn_{msg_key}") and follow.strip():
                            st.session_state.pending = follow.strip()
                            st.rerun()
                    with col2:
                        if st.button("再生成", key=f"regen_{msg_key}"):
                            st.session_state.pending = q
                            st.rerun()
        else:
            # フォールバック（非チャットUI）
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:**\n\n{a}" if a else "**A:** _（回答なし）_")
            if sugg:
                render_suggestions(sugg, msg_key=msg_key)
            if show_sources:
                render_sources(srcs)
            st.markdown("---")

# ====== 送信処理 ======
def ask(question: str):
    data = call_api(question)
    if not data:
        return
    data = ensure_new_format(data)
    st.session_state.history.append({
        "id": uuid4().hex,
        "q": question,
        **data
    })

# ====== メイン描画 ======
render_chat()

# 入力欄（chat_input があれば使う）
use_chat = hasattr(st, "chat_input")
pending = st.session_state.pending
st.session_state.pending = None

if use_chat:
    # 提案などで pending があれば先に送る
    if pending:
        ask(pending)
        st.rerun()
    user_q = st.chat_input("知りたいことを入力してください", key="chat_q", max_chars=2000)
    if user_q and user_q.strip():
        ask(user_q.strip())
        st.rerun()
else:
    with st.form("ask"):
        user_q = st.text_input("知りたいことを入力してください", value=pending or "")
        submitted = st.form_submit_button("送信")
    if submitted and user_q.strip():
        ask(user_q.strip())
        st.rerun()
