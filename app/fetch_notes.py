# fetch_notes.py
import os
import re
import json
import time
import datetime as dt
from typing import Dict, Any, List, Set, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── 環境変数（必要に応じて .env で設定） ──────────────────────────────────
USER_ID: str = os.environ.get("NOTE_USER_ID", "hinataptyan")

# 取得ページの上限（None なら最後まで）
MAX_PAGES_ENV = os.environ.get("MAX_PAGES")
MAX_PAGES = int(MAX_PAGES_ENV) if MAX_PAGES_ENV and MAX_PAGES_ENV.isdigit() else None

# 既存ファイルがあればスキップ（デフォルト: True）
SKIP_EXISTING: bool = os.environ.get("SKIP_EXISTING", "1") == "1"

# 1リクエスト毎の待機（APIに優しく）
SLEEP_SECS: float = float(os.environ.get("SLEEP_SECS", "0.3"))

# ファイル名最大長（拡張子抜き）
FNAME_MAXLEN: int = int(os.environ.get("FNAME_MAXLEN", "96"))

SAVE_DIR: str = os.environ.get("ARTICLES_DIR", "articles")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── API エンドポイント ───────────────────────────────────────────────────
BASE_LIST: str  = f"https://note.com/api/v2/creators/{USER_ID}/contents?kind=note&page={{page}}"
DETAIL_URL: str = "https://note.com/api/v3/notes/{key}"

# 実ブラウザっぽい UA を付与（ブロック回避の一助）
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

# ── HTTP セッション（リトライ/タイムアウト） ────────────────────────────
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(DEFAULT_HEADERS)
    return s

# ── テキスト化＆リンク抽出 ────────────────────────────────────────────────
ABS_HTTP = re.compile(r"^https?://", re.IGNORECASE)

def clean_text_and_extract_links(html: str) -> Tuple[str, List[str]]:
    """
    HTMLをざっくりテキスト化し、<figure> と 通常の <a> のリンクを抽出。
    - <figure> は本文から除去（data-src を優先、無ければ内部の <a href>）
    - 通常の <a> は href を収集し、aタグ自体は unwrap（テキストは残す）
    - <script>/<style> は除去
    - 連続改行や空白を正規化
    戻り値: (テキスト本文, 抽出リンク配列[絶対HTTPのみ])
    """
    soup = BeautifulSoup(html or "", "html.parser")

    # 余計な属性を削除（必要なければ削ってOK）
    for tag in soup.find_all(True):
        tag.attrs.pop("name", None)
        tag.attrs.pop("id",   None)

    links: List[str] = []
    seen: Set[str] = set()

    # 1) figure の抽出（data-src 優先、なければ中の a[href]）
    for fig in soup.find_all("figure"):
        url = fig.get("data-src")
        if not url:
            a = fig.find("a", href=True)
            url = a["href"] if a else None
        if url and ABS_HTTP.match(url) and url not in seen:
            links.append(url)
            seen.add(url)
        fig.decompose()  # 本文からは除去

    # 2) script/style の除去
    for tag in soup(["script", "style"]):
        tag.decompose()

    # 3) 通常の a[href] を抽出しつつ a タグだけ除去（テキストは残す）
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ABS_HTTP.match(href) and href not in seen:
            links.append(href)
            seen.add(href)
        a.unwrap()

    # 4) テキスト化
    text = soup.get_text(separator="\n")

    # 5) 空行・空白を正規化
    text = re.sub(r"\r\n?", "\n", text)          # CRLF→LF
    text = re.sub(r"\n{2,}", "\n\n", text)       # 2連以上の改行は2つに
    lines = [line.strip() for line in text.splitlines()]
    clean = "\n".join(line for line in lines if line)

    return clean, links

# ── ユーティリティ ────────────────────────────────────────────────────────
def sanitize_filename(s: str, max_len: int = 96) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    if len(s) > max_len:
        s = s[:max_len]
    s = s.strip("_") or "note"
    return s

def pick_first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return default

def save_files(basepath: str, title: str, body: str, meta: Dict[str, Any]) -> None:
    # .txt
    with open(basepath + ".txt", "w", encoding="utf-8") as f:
        f.write(f"タイトル: {title}\n\n")
        f.write(body)
    # .json（メタ）
    with open(basepath + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

# ── メイン処理 ────────────────────────────────────────────────────────────
def main() -> None:
    sess = make_session()
    page = 1
    total = 0
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    print(f"[INFO] Fetch start: user={USER_ID}, dir={SAVE_DIR}, started_at={started_at}")

    while True:
        if MAX_PAGES and page > MAX_PAGES:
            print(f"[INFO] Reached MAX_PAGES={MAX_PAGES}, stop.")
            break

        url = BASE_LIST.format(page=page)
        print(f"[INFO] Fetching list: {url}")
        try:
            resp = sess.get(url, timeout=(5, 20))
            resp.raise_for_status()
        except Exception as e:
            print(f"[WARN] List request failed (page={page}): {e}")
            break

        data = (resp.json() or {}).get("data", {})
        contents = data.get("contents", [])
        if not contents:
            print("[INFO] No more articles found.")
            break

        for c in contents:
            title    = c.get("name") or "no_title"
            note_key = c.get("key")
            slug     = c.get("slug") or str(c.get("id") or "")

            if not note_key:
                print("[WARN] Skip: note_key missing")
                continue

            # 詳細API
            try:
                det = sess.get(DETAIL_URL.format(key=note_key), timeout=(5, 20))
                det.raise_for_status()
                det_json = det.json() or {}
            except Exception as e:
                print(f"[WARN] Detail request failed (key={note_key}): {e}")
                continue

            html = (det_json.get("data") or {}).get("body", "") or ""
            clean_body, links = clean_text_and_extract_links(html)

            # ファイル名（pageとslugで安定化、slugはサニタイズ＋長さ制限）
            safe_slug = sanitize_filename(slug, max_len=FNAME_MAXLEN)
            base = f"{SAVE_DIR}/{page:02d}_{safe_slug}"
            txt_path = base + ".txt"
            json_path = base + ".json"

            if SKIP_EXISTING and os.path.exists(txt_path):
                print(f"[SKIP] Exists: {txt_path}")
                continue

            # メタデータを拡充
            # Note API 側の日時キーは揺れる可能性があるため候補を横断的に拾う
            published_at = pick_first(c, ["publishedAt", "publishAt", "published_at", "publish_at", "createdAt", "created_at"])
            updated_at   = pick_first(c, ["updatedAt", "updated_at"])
            canonical_url = f"https://note.com/{USER_ID}/n/{note_key}"

            meta = {
                "user_id": USER_ID,
                "title": title,
                "slug": slug,
                "key": note_key,
                "page": page,
                "source": {
                    "list_api": url,
                    "detail_api": DETAIL_URL.format(key=note_key),
                    "canonical": canonical_url,
                },
                "timestamps": {
                    "published_at": published_at,
                    "updated_at": updated_at,
                    "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
                },
                "link_count": len(links),
                "links": links,
                "length": len(clean_body),
            }

            # 保存
            save_files(base, title, clean_body, meta)
            print(f"[OK] Saved: {txt_path}")
            total += 1

            time.sleep(SLEEP_SECS)  # レート制御（detail）

        if data.get("isLastPage"):
            print("[INFO] Last page reached.")
            break

        page += 1
        time.sleep(SLEEP_SECS)  # レート制御（list）

    print(f"[DONE] {total} articles saved. (user={USER_ID})")


if __name__ == "__main__":
    main()
