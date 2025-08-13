#!/bin/sh
set -e

cd /app

echo "[entrypoint] embed_articles.py を実行してコレクションを用意します（既存ならスキップされます）..."
if [ -f ./embed_articles.py ]; then
  # 埋め込みスクリプトは idempotent（既存ならスキップ）にしてある想定
  python embed_articles.py || echo "[entrypoint] embed_articles.py 実行でエラー（続行します）"
else
  echo "[entrypoint] embed_articles.py が見つかりません。スキップします。"
fi

echo "[entrypoint] アプリを起動します..."
exec "$@"
