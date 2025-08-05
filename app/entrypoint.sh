#!/bin/sh
set -e
cd /app

# ./chroma_db が空または存在しなければ embed_articles.py を実行
if [ ! -d "chroma_db" ] || [ -z "$(ls -A chroma_db)" ]; then
  echo "[entrypoint] ChromaDB にデータがないため、埋め込み処理を実行します…"
  python embed_articles.py
  echo "[entrypoint] 埋め込み処理 完了."
else
  echo "[entrypoint] 既存の ChromaDB データがあるのでスキップします."
fi

# CMD 以下を実行（uvicorn main:app など）
exec "$@"
