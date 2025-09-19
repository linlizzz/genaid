# pip install rank_bm25 nltk
import os, json, re, pickle
from glob import glob
from rank_bm25 import BM25Okapi
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
from utils import load_paths
paths = load_paths()
ALL_GUIDELINE_DIR = paths["ALL_GUIDELINE_DIR"]

DATA_DIR = ALL_GUIDELINE_DIR     # 你的指南 jsonl 目录
OUT_PATH = "bm25_index_chunk.pkl"       # 索引持久化文件

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_finnish(s: str):
    # 简易分词（芬兰语可更换为更好的 tokenizer；先用空格+少量清洗）
    s = normalize_text(s)
    # 也可以在此处加入芬兰语停用词/词干（可选）
    return s.split()

def iter_chunks():
    # 你的 jsonl 结构：每行包含 guideline_id, chunks: [{chunk_id, content/page_content}, ...]
    for path in sorted(glob(os.path.join(DATA_DIR, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                jo = json.loads(line)
                gid = jo.get("guideline_id")
                for ch in jo.get("chunks", []):
                    cid = ch.get("chunk_id")
                    text = ch.get("content") or ch.get("page_content") or ""
                    yield {"guideline_id": gid, "chunk_id": cid, "text": text}

def main_build_bm25_index():
    docs = []
    metas = []  # 与 docs 对应，用于回查 gid/cid
    for rec in iter_chunks():
        toks = tokenize_finnish(rec["text"])
        if not toks: continue
        docs.append(toks)
        metas.append({"guideline_id": rec["guideline_id"], "chunk_id": rec["chunk_id"]})

    bm25 = BM25Okapi(docs)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "metas": metas}, f)
    print(f"[ok] BM25 index saved to {OUT_PATH}, docs={len(metas)}")

if __name__ == "__main__":
    main_build_bm25_index()
