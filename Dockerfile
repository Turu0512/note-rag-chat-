FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y \
    curl unzip wget gnupg \
    chromium-driver chromium \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python依存ライブラリのインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

# Streamlit起動がデフォルト
CMD ["streamlit", "run", "main.py"]
