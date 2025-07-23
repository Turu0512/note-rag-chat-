import requests
import os
import re
from bs4 import BeautifulSoup

USER_ID    = os.environ.get("NOTE_USER_ID", "hinataptyan")
BASE_LIST  = f"https://note.com/api/v2/creators/{USER_ID}/contents?kind=note&page={{page}}"
DETAIL_URL = "https://note.com/api/v3/notes/{key}"
SAVE_DIR   = "articles"
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

def clean_text_and_extract_links(html: str) -> str:
    """HTMLをざっくりテキスト化し、<figure>や<a>のリンクのみ末尾にまとめる"""
    soup = BeautifulSoup(html, "html.parser")
    
    for tag in soup.find_all(True):
      tag.attrs.pop("name", None)
      tag.attrs.pop("id",   None)

    # 1) <figure> 内の data-src または <a> リンクを抽出して削除
    links = []
    for fig in soup.find_all("figure"):
        # data-src があれば優先
        url = fig.get("data-src")
        if not url:
            a = fig.find("a", href=True)
            url = a["href"] if a else None
        if url:
            links.append(url)
        fig.decompose()

    # 2) <script> と <style> は除去
    for tag in soup(["script", "style"]):
        tag.decompose()

    # 3) テキスト化
    text = soup.get_text(separator="\n")

    # 4) 空行・空白を正規化
    text = re.sub(r"\n{2,}", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    clean = "\n".join(line for line in lines if line)

    # 5) リンクを末尾に追記
    if links:
        clean += "\n\n[外部リンク]\n" + "\n".join(f"- {u}" for u in links)

    return clean


def main():
    page = 1
    total = 0

    while True:
        url = BASE_LIST.format(page=page)
        print(f"[INFO] Fetching list: {url}")
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        contents = data.get("contents", [])
        if not contents:
            print("[INFO] No more articles found.")
            break

        for c in contents:
            title    = c.get("name", "no_title")
            note_key = c.get("key")
            slug     = c.get("slug") or str(c.get("id"))

            # 詳細 API から HTML を取得
            det = requests.get(DETAIL_URL.format(key=note_key), headers=headers)
            det.raise_for_status()
            html = det.json().get("data", {}).get("body", "")

            # テキスト化＋リンク抽出
            clean_body = clean_text_and_extract_links(html)

            # ファイル名生成
            safe_slug = "".join(ch if ch.isalnum() else "_" for ch in slug)
            fn = f"{SAVE_DIR}/{page:02d}_{safe_slug}.txt"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(f"タイトル: {title}\n\n")
                f.write(clean_body)

            print(f"[OK] Saved: {fn}")
            total += 1

        if data.get("isLastPage"):
            break
        page += 1

    print(f"[DONE] {total} articles saved.")


if __name__ == "__main__":
    main()
