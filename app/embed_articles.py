# embed_articles.py
import os
import glob
import json
from typing import List, Dict, Set

from dotenv import load_dotenv
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── ENV ─────────────────────────────────────────────────────────────
PERSIST_DIR   = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION    = os.environ.get("CHROMA_COLLECTION", "note_articles")
ARTICLES_DIR  = os.environ.get("ARTICLES_DIR", "./articles")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
FORCE_REINDEX = os.environ.get("FORCE_REINDEX", "0") == "1"

CHUNK_MAX_CHARS      = int(os.environ.get("CHUNK_MAX_CHARS", "1200"))
CHUNK_OVERLAP_CHARS  = int(os.environ.get("CHUNK_OVERLAP_CHARS", "100"))
BATCH_SIZE           = int(os.environ.get("EMBED_BATCH_ADD_SIZE", "200"))
ENCODE_BATCH_SIZE    = int(os.environ.get("ENCODE_BATCH_SIZE", "32"))

# ── utils ───────────────────────────────────────────────────────────
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def read_sidecar_json(txt_path: str) -> Dict:
    base, _ = os.path.splitext(txt_path)
    json_path = base + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def chunk_text(s: str, max_chars: int, overlap: int) -> List[str]:
    if not s:
        return []
    if max_chars <= 0:
        return [s]
    chunks: List[str] = []
    i = 0
    n = len(s)
    step = max(1, max_chars - max(0, overlap))
    while i < n:
        j = min(n, i + max_chars)
        piece = s[i:j].strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks

def paged_get_all_ids(col, page_size: int = 10000) -> Set[str]:
    existing: Set[str] = set()
    offset = 0
    while True:
        try:
            res = col.get(include=["ids"], limit=page_size, offset=offset)
        except TypeError:
            res = col.get(include=["ids"])
        ids = res.get("ids") or []
        if not ids:
            break
        if ids and isinstance(ids[0], list):
            for g in ids:
                existing.update(g or [])
        else:
            existing.update(ids)
        if len(ids) < page_size:
            break
        offset += page_size
    return existing

def drop_collection_safely(client: PersistentClient, col, name: str) -> None:
    try:
        client.delete_collection(name)
        print(f"[embed] dropped collection: {name}")
    except Exception as e:
        print(f"[embed] delete_collection failed ({e}); fallback to delete-by-ids")
        ids = paged_get_all_ids(col)
        if ids:
            col.delete(ids=list(ids))
            print(f"[embed] deleted {len(ids)} items by ids")

def build_flat_metadata(base_meta: Dict, json_meta: Dict) -> Dict:
    """
    Chroma はメタ値にプリミティブのみ許す。
    サイドカーJSONから必要フィールドだけフラットに抽出して返す。
    """
    out = dict(base_meta)  # filename, chunk など
    if not json_meta:
        return {k: v for k, v in out.items() if v is not None}

    # そのまま転写して良いプリミティブ
    for k in ("user_id", "title", "slug", "key", "page", "link_count", "length"):
        v = json_meta.get(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v

    # ネスト: source / timestamps をフラット化
    src = (json_meta.get("source") or {})
    if isinstance(src, dict):
        for subk in ("canonical", "list_api", "detail_api"):
            v = src.get(subk)
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[f"source_{subk}"] = v

    ts = (json_meta.get("timestamps") or {})
    if isinstance(ts, dict):
        for subk in ("published_at", "updated_at", "downloaded_at"):
            v = ts.get(subk)
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[subk] = v

    # 注意: links は配列なのでメタからは省略（link_count を使う）
    # 他に残った入れ子/配列は入れない

    # None を落として返す
    return {k: v for k, v in out.items() if v is not None}

# ── main ────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[embed] collection={COLLECTION} dir={os.path.abspath(PERSIST_DIR)} model={EMBED_MODEL}")
    client = PersistentClient(path=PERSIST_DIR)
    col = client.get_or_create_collection(COLLECTION, metadata={"embedding_model": EMBED_MODEL})

    total_existing = col.count()
    if FORCE_REINDEX and total_existing > 0:
        drop_collection_safely(client, col, COLLECTION)
        col = client.get_or_create_collection(COLLECTION, metadata={"embedding_model": EMBED_MODEL})
        print(f"[embed] re-created empty collection: {COLLECTION}")
        total_existing = 0

    paths = sorted(glob.glob(os.path.join(ARTICLES_DIR, "*.txt")))
    if not paths:
        print(f"[embed] 入力記事がありません: {ARTICLES_DIR}")
        return

    existing_ids: Set[str] = set()
    if total_existing > 0:
        existing_ids = paged_get_all_ids(col)
        print(f"[embed] 既存ID読み込み: {len(existing_ids)} 件")

    model = SentenceTransformer(EMBED_MODEL)
    print(f"[embed] embedding dim={model.get_sentence_embedding_dimension()}")

    add_ids: List[str] = []
    add_docs: List[str] = []
    add_metas: List[Dict] = []

    added = 0
    skipped = 0
    files_processed = 0

    def flush_batch():
        nonlocal add_ids, add_docs, add_metas, added
        if not add_ids:
            return
        embs = model.encode(add_docs, batch_size=ENCODE_BATCH_SIZE, show_progress_bar=False).tolist()
        col.add(ids=add_ids, documents=add_docs, metadatas=add_metas, embeddings=embs)
        added += len(add_ids)
        print(f"[embed] add: {len(add_ids)} docs (累計 {added})")
        add_ids, add_docs, add_metas = [], [], []

    for p in paths:
        files_processed += 1
        txt = read_text(p)
        if not txt:
            continue
        fname = os.path.basename(p)
        chunks = chunk_text(txt, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        if not chunks:
            continue

        meta_json = read_sidecar_json(p)

        for i, ch in enumerate(chunks):
            doc_id = f"{fname}#{i:03d}"
            if existing_ids and doc_id in existing_ids:
                skipped += 1
                continue

            base_meta = {"filename": fname, "chunk": i}
            meta = build_flat_metadata(base_meta, meta_json)

            add_ids.append(doc_id)
            add_docs.append(ch)
            add_metas.append(meta)

            if len(add_ids) >= BATCH_SIZE:
                flush_batch()

    flush_batch()
    print(f"[embed] 完了 files={files_processed}, added={added}, skipped={skipped}, total_in_collection={col.count()}")

if __name__ == "__main__":
    main()
