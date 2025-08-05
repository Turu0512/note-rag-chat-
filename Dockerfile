FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# — 必要なパッケージインストール —
RUN apt-get update && apt-get install -y \
    curl unzip wget gnupg \
    chromium-driver chromium \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# — 作業ディレクトリを /app に —
WORKDIR /app

# — Python依存ライブラリ —
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# — エントリポイントスクリプトを /
COPY app/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# — アプリケーションコード一式を /app に —
COPY . .

# — ENTRYPOINT とデフォルトCMD を設定 —
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
