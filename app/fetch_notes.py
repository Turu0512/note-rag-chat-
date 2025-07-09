import os
import requests
from bs4 import BeautifulSoup
import html2text

USER_ID = os.environ.get("NOTE_USER_ID", "hinataptyan")  # ← あなたのNoteユーザーIDを指定
BASE_URL = f"https://note.com/{USER_ID}"

SAVE_DIR = "articles"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_article_urls():
    print(f"[INFO] Fetching article list for: {BASE_URL}")
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    res = requests.get(BASE_URL, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(f"/{USER_ID}/n/"):
            full_url = f"https://note.com{href}"
            if full_url not in links:
                links.append(full_url)

    print(f"[INFO] Found {len(links)} articles.")
    return links

def fetch_article_text(url):
    print(f"[INFO] Fetching article: {url}")
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    article_div = soup.find("div", {"id": "js-content"})
    if not article_div:
        return None
    html = str(article_div)
    return html2text.html2text(html)

def main():
    print("[RUNNING] fetch_notes.py started")
    urls = fetch_article_urls()
    for i, url in enumerate(urls):
        text = fetch_article_text(url)
        if text:
            filename = f"{SAVE_DIR}/article_{i+1}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[OK] Saved to {filename}")
        else:
            print(f"[WARN] Skipped (no content): {url}")

if __name__ == "__main__":
    main()
