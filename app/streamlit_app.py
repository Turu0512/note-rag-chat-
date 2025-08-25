# streamlit_app.py
import os
import json
import requests
import streamlit as st

st.set_page_config(page_title="Note RAG Chat", page_icon="💬", layout="centered")

DEFAULT_API_URL = os.environ.get("NOTE_RAG_API_URL", "http://localhost:8000/query")

with st.sidebar:
    st.header("設定")
    api_url = st.text_input("API URL（FastAPI /query）", value=DEFAULT_API_URL)
    display_mode = st.radio("表示モード", ["Markdown", "プレーンテキスト"], index=0, horizontal=True)
    st.caption("APIは summary / points のJSONを返します。")

st.title("Note 記事チャット")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 既存メッセージを描画
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant" and isinstance(m.get("content"), dict):
            data = m["content"]
            summary = data.get("summary", "")
            points = data.get("points", [])
            if display_mode == "Markdown":
                st.markdown(summary)
                if points:
                    st.markdown("\n".join([f"- {p}" for p in points]))
            else:
                st.write(summary)
                for p in points:
                    st.write(f"・{p}")
            with st.expander("生JSONを表示"):
                st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
        else:
            st.markdown(m.get("content", ""))

# 入力欄
prompt = st.chat_input("質問を入力...")
if prompt:
    # ユーザメッセージを表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # アシスタント応答
    with st.chat_message("assistant"):
        try:
            r = requests.post(api_url, json={"question": prompt}, timeout=120)
            if r.status_code != 200:
                st.error(f"APIエラー: {r.status_code}\n{r.text}")
                st.session_state.messages.append({"role": "assistant", "content": f"APIエラー: {r.status_code}"})
            else:
                data = r.json()
                if display_mode == "Markdown":
                    st.markdown(data.get("summary", ""))
                    if data.get("points"):
                        st.markdown("\n".join([f"- {p}" for p in data["points"]]))
                else:
                    st.write(data.get("summary", ""))
                    for p in data.get("points", []):
                        st.write(f"・{p}")
                with st.expander("生JSONを表示"):
                    st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")
                st.session_state.messages.append({"role": "assistant", "content": data})
        except Exception as e:
            st.error(f"接続失敗: {e}")

with st.sidebar:
    if st.button("履歴をクリア", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
