import requests
import os

USER_ID = os.environ.get("NOTE_USER_ID", "hinataptyan")
BASE_LIST_URL = f"https://note.com/api/v2/creators/{USER_ID}/contents?kind=note&page={{page}}"
DETAIL_URL = "https://note.com/api/v3/notes/{key}"
SAVE_DIR = "articles"
os.makedirs(SAVE_DIR, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}

page = 1
total = 0

while True:
    url = BASE_LIST_URL.format(page=page)
    print(f"[INFO] Fetching list: {url}")
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()["data"]
    contents = data["contents"]
    if not contents:
        print("[INFO] No more articles found.")
        break
    for c in contents:
        title = c.get("name", "no_title")
        note_key = c.get("key")
        slug = c.get("slug") or str(c.get("id"))
        # 詳細APIで全文取得
        detail_resp = requests.get(DETAIL_URL.format(key=note_key), headers=headers)
        detail_resp.raise_for_status()
        detail = detail_resp.json().get("data", {})
        full_body = detail.get("body", "")
        safe_slug = "".join(x if x.isalnum() else "_" for x in slug)
        filename = f"{SAVE_DIR}/{page:02d}_{safe_slug}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"タイトル: {title}\n\n")
            f.write(full_body)
        print(f"[OK] Saved: {filename}")
        total += 1
    if data.get("isLastPage"):
        break
    page += 1

print(f"[DONE] {total} articles saved.")
