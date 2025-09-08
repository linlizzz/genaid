#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, ast, argparse, re, ast, glob, argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from utils import load_paths

"""
(E5 前缀、keywords 解析、content 抽取都内置)

批量构建多种 embedding 模型与多种索引变体,支持 4 种模式:chunk_only / concat / triplet_raw / fused_from_triplet:
- 对每个模型在 out_root/<sanitized_model>/ 下生成:
  - chunk_only/
  - concat_titlekw_chunk/
  - triplet_raw/  (vectors/{chunk.npy,title.npy,keywords.npy} + faiss/*.index.faiss)
  - fused_alphaX_betaY_gammaZ/  (可选，从 triplet_raw 离线融合)

示例:
python build_faiss_index.py \
  --data_glob "datasets/json_guidelines/**/*.jsonl" \
  --out_root "faiss_store" \
  --models "intfloat/multilingual-e5-base,sentence-transformers/all-MiniLM-L6-v2,BAAI/bge-m3" \
  --modes "chunk_only,concat,triplet,fused" \
  --fusion_weights "1.0,0.6,0.4" \
  --skip_existing
"""

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]
EMBEDDING_MODELS_LIST_PATH = paths["EMBEDDING_MODELS_LIST_PATH"]

# select model
candidates = [
    "sentence-transformers/all-MiniLM-L6-v2",   # 384d, small and fast
    "intfloat/multilingual-e5-base",            # 768d, multilingual, need E5 prefix
    "sentence-transformers/msmarco-MiniLM-L-12-v3",  # 384d, more retrieval-oriented
    "embed-multilingual-v3.0",# 1024,
    "text-embedding-ada-002",# 1536,
    "text-embedding-3-large",# 3072,
    "BAAI/bge-m3",# 1024,
    "intfloat/multilingual-e5-large",# 1024,
    "intfloat/multilingual-e5-large-instruct",# 1024,
    "intfloat/multilingual-e5-base",# 768,
    "intfloat/multilingual-e5-small",# 384,
    "TurkuNLP/sbert-cased-finnish-paraphrase",# 768,
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",#medical specific
    "biobert-base-cased-v1.1"#medical specific
]

data_dir = GUIDELINE_JSON_DIR   # guideline data directory
save_root = GUIDELINE_FAISS_DIR        # mainindex directory
k = 10                           # @k for evaluation



# -------------------- 数据字段工具 --------------------

def extract_chunk_text(ch: dict) -> str:
    # 兼容 content/page_content 以及一些常见变体
    for k in ["content","page_content","text","body","raw_text","paragraphs","sections","data"]:
        if k in ch:
            v = ch[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, list):
                parts = []
                for it in v:
                    if isinstance(it, str) and it.strip():
                        parts.append(it.strip())
                    elif isinstance(it, dict):
                        for kk in ("text","content","value","raw_text","paragraph"):
                            if kk in it and isinstance(it[kk], str) and it[kk].strip():
                                parts.append(it[kk].strip())
                                break
                if parts:
                    return "\n".join(parts).strip()
            if isinstance(v, dict):
                for kk in ("text","content","value","raw_text","paragraph"):
                    if kk in v and isinstance(v[kk], str) and v[kk].strip():
                        return v[kk].strip()
    return ""

def parse_keywords(raw):
    # 兼容 ["A","B"] 与 "['A','B']"
    if isinstance(raw, str):
        s = raw.strip()
        if not s or s.lower() in ("none","null"):
            return []
        try:
            x = ast.literal_eval(s)
            if isinstance(x, (list, tuple)):
                return [str(t).strip() for t in x if str(t).strip()]
            return [str(x).strip()] if str(x).strip() else []
        except Exception:
            return [s] if s else []
    if isinstance(raw, (list, tuple)):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [str(raw).strip()] if str(raw).strip() else []

def load_guidelines(jsonl_glob: str) -> pd.DataFrame:
    rows = []
     # 如果传的是目录，就自动加上/*.jsonl
    if os.path.isdir(jsonl_glob):
        files = sorted(glob.glob(os.path.join(jsonl_glob, "*.jsonl")))
    else:
        files = sorted(glob.glob(jsonl_glob, recursive=True))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in: {jsonl_glob}")
    

    for p in files:
        with open(p, "r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                gid   = obj.get("guideline_id")
                title = obj.get("title", "")
                kws   = ", ".join(parse_keywords(obj.get("keywords", [])))
                chunks = obj.get("chunks", [])
                for ch in chunks:
                    cid = ch.get("chunk_id") or ch.get("id")
                    content = extract_chunk_text(ch)
                    if not content:
                        continue
                    rows.append({
                        "guideline_id": gid,
                        "chunk_id": cid,
                        "title": title,
                        "keywords": kws,
                        "content": content
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid chunks with non-empty content were found.")
    return df

# -------------------- 编码/索引工具 --------------------

def sanitize_model_name(model: str) -> str:
    # 将 "intfloat/multilingual-e5-base" -> "intfloat_multilingual-e5-base"
    # 再把 "." -> "p" 以避免文件系统/路径转义问题
    s = model.strip().replace("/", "_")
    s = s.replace(" ", "_")
    s = s.replace(".", "p")
    s = re.sub(r"[^A-Za-z0-9_\-\+]", "_", s)
    return s

def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in (model_name or "").lower()

def encode_passages(model, texts: List[str], use_e5: bool, metric: str, batch_size=256) -> np.ndarray:
    if use_e5:
        texts = [f"passage: {t}" for t in texts]
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    if metric == "cosine":
        faiss.normalize_L2(vecs)
    return vecs.astype("float32")

def build_faiss_index(vecs: np.ndarray, metric: str = "cosine"):
    if metric == "cosine":
        index = faiss.IndexFlatIP(vecs.shape[1])
    else:
        index = faiss.IndexFlatL2(vecs.shape[1])
    index.add(vecs)
    return index

def save_plain_index(out_dir, index, meta_df, config: dict, fname="index.faiss"):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(out_dir, fname))
    meta_df.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

# -------------------- 模式构建 --------------------

def build_chunk_only(df: pd.DataFrame, model, use_e5: bool, metric: str, out_dir: str, model_name: str):
    vecs = encode_passages(model, df["content"].tolist(), use_e5, metric)
    index = build_faiss_index(vecs, metric)
    save_plain_index(out_dir, index, df, {
        "model": model_name, "mode": "chunk_only", "dim": int(vecs.shape[1]), "metric": metric
    })

def build_concat(df: pd.DataFrame, model, use_e5: bool, metric: str, out_dir: str, model_name: str):
    texts = [f"[TITLE] {t}\n[KEYWORDS] {k}\n[CONTENT]\n{c}"
             for t, k, c in zip(df["title"], df["keywords"], df["content"])]
    vecs = encode_passages(model, texts, use_e5, metric)
    index = build_faiss_index(vecs, metric)
    save_plain_index(out_dir, index, df, {
        "model": model_name, "mode": "concat_titlekw_chunk", "dim": int(vecs.shape[1]), "metric": metric
    })

def build_triplet_raw(df: pd.DataFrame, model, use_e5: bool, metric: str, trip_dir: str, model_name: str):
    os.makedirs(trip_dir, exist_ok=True)
    v_chunk = encode_passages(model, df["content"].tolist(), use_e5, metric)
    v_title = encode_passages(model, df["title"].astype(str).tolist(), use_e5, metric)
    v_kw    = encode_passages(model, df["keywords"].astype(str).tolist(), use_e5, metric)

    # 保存向量
    os.makedirs(os.path.join(trip_dir, "vectors"), exist_ok=True)
    np.save(os.path.join(trip_dir, "vectors", "chunk.npy"), v_chunk)
    np.save(os.path.join(trip_dir, "vectors", "title.npy"), v_title)
    np.save(os.path.join(trip_dir, "vectors", "keywords.npy"), v_kw)

    # 保存三套索引（用于后融合/通道消融）
    os.makedirs(os.path.join(trip_dir, "faiss"), exist_ok=True)
    faiss.write_index(build_faiss_index(v_chunk, metric), os.path.join(trip_dir, "faiss", "chunk.index.faiss"))
    faiss.write_index(build_faiss_index(v_title, metric), os.path.join(trip_dir, "faiss", "title.index.faiss"))
    faiss.write_index(build_faiss_index(v_kw,    metric), os.path.join(trip_dir, "faiss", "keywords.index.faiss"))

    df.to_parquet(os.path.join(trip_dir, "meta.parquet"), index=False)
    with open(os.path.join(trip_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "mode": "triplet_raw", "dim": int(v_chunk.shape[1]), "metric": metric}, f, ensure_ascii=False, indent=2)

def build_fused_from_triplet(trip_dir: str, out_parent: str, weights: Tuple[float,float,float], metric: str):
    alpha, beta, gamma = weights
    df = pd.read_parquet(os.path.join(trip_dir, "meta.parquet"))
    v_chunk = np.load(os.path.join(trip_dir, "vectors", "chunk.npy"))
    v_title = np.load(os.path.join(trip_dir, "vectors", "title.npy"))
    v_kw    = np.load(os.path.join(trip_dir, "vectors", "keywords.npy"))

    fused = alpha * v_chunk + beta * v_title + gamma * v_kw
    if metric == "cosine":
        faiss.normalize_L2(fused)

    tag = f"fused_alpha{alpha}_beta{beta}_gamma{gamma}".replace(".", "p")
    out_dir = os.path.join(out_parent, tag)
    index = build_faiss_index(fused, metric)
    save_plain_index(out_dir, index, df, {
        "model": "from_triplet_raw", "mode": "fused", "weights": [alpha,beta,gamma],
        "dim": int(fused.shape[1]), "metric": metric
    })
    return out_dir

# -------------------- 主程序 --------------------

def main_build_faiss_index(embed_model: List[str]):
    print(GUIDELINE_JSON_DIR)
    print(GUIDELINE_FAISS_DIR)
    print(EMBEDDING_MODELS_LIST_PATH)
    print("Done print paths")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", default=GUIDELINE_JSON_DIR, help="指南 JSONL 的 glob,如 datasets/**/*.jsonl")
    ap.add_argument("--out_root", default=GUIDELINE_FAISS_DIR, help="输出根目录，如 faiss_store")
    # ap.add_argument("--models", type=str, default="intfloat/multilingual-e5-large-instruct", help="逗号分隔的模型列表,和--models_file二选一,默认使用embedding_models.txt")
    ap.add_argument("--models_file", type=str, default="") # EMBEDDING_MODELS_LIST_PATH, help="包含每行一个模型名的文件, .txt文件")
    ap.add_argument("--modes", type=str, default="chunk_only,concat,triplet,fused",
                    help="要构建的模式:chunk_only,concat,triplet,fused(逗号分隔)")
    ap.add_argument("--fusion_weights", type=str, default="1.0,0.6,0.4", help="用于 fused 的 alpha,beta,gamma")
    ap.add_argument("--metric", choices=["cosine","l2"], default="cosine")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--skip_existing", action="store_true", help="如果子目录已存在则跳过")
    args = ap.parse_args()

    # 解析模型列表
    models: List[str] = []
    if embed_model:
        models += [m.strip() for m in embed_model.split(",") if m.strip()]
    if args.models_file:
        with open(args.models_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    models.append(s)
    if not models:
        raise ValueError("Please provide at least one model via --models or --models_file")

    # 解析模式与权重
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    build_chunk = "chunk_only" in modes
    build_concat_mode = "concat" in modes
    build_triplet = "triplet" in modes
    build_fused = "fused" in modes
    alpha, beta, gamma = [float(x) for x in args.fusion_weights.split(",")]

    # 加载数据一次
    df = load_guidelines(args.data_glob)
    print(f"[data] total chunks: {len(df)}  (guidelines={df['guideline_id'].nunique()})")

    os.makedirs(args.out_root, exist_ok=True)

    for model_name in models:
        print(f"\n=== Building for model: {model_name} ===")
        model_dir = os.path.join(args.out_root, sanitize_model_name(model_name))
        os.makedirs(model_dir, exist_ok=True)

        # 只有需要编码的模式才加载模型
        model = None
        use_e5 = needs_e5_prefix(model_name)
        if build_chunk or build_concat_mode or build_triplet:
            print(f"[load] SentenceTransformer({model_name})")
            model = SentenceTransformer(model_name)

        # 1) chunk_only
        if build_chunk:
            out_dir = os.path.join(model_dir, "chunk_only")
            if args.skip_existing and os.path.exists(os.path.join(out_dir, "index.faiss")):
                print(f"[skip] {out_dir} exists")
            else:
                print("[build] chunk_only ...")
                build_chunk_only(df, model, use_e5, args.metric, out_dir, model_name)
                print(f"[done] {out_dir}")

        # 2) concat
        if build_concat_mode:
            out_dir = os.path.join(model_dir, "concat_titlekw_chunk")
            if args.skip_existing and os.path.exists(os.path.join(out_dir, "index.faiss")):
                print(f"[skip] {out_dir} exists")
            else:
                print("[build] concat_titlekw_chunk ...")
                build_concat(df, model, use_e5, args.metric, out_dir, model_name)
                print(f"[done] {out_dir}")

        # 3) triplet_raw
        trip_dir = os.path.join(model_dir, "triplet_raw")
        if build_triplet:
            if args.skip_existing and os.path.exists(os.path.join(trip_dir, "vectors", "chunk.npy")):
                print(f"[skip] {trip_dir} exists")
            else:
                print("[build] triplet_raw ...")
                build_triplet_raw(df, model, use_e5, args.metric, trip_dir, model_name)
                print(f"[done] {trip_dir}")

        # 4) fused_from_triplet
        if build_fused:
            # 依赖 triplet_raw 已存在（或刚构建）
            if not os.path.exists(os.path.join(trip_dir, "vectors", "chunk.npy")):
                raise FileNotFoundError(f"triplet_raw not found for fused at: {trip_dir}")
            print(f"[build] fused (alpha={alpha}, beta={beta}, gamma={gamma}) ...")
            fused_dir = build_fused_from_triplet(trip_dir, model_dir, (alpha,beta,gamma), args.metric)
            print(f"[done] {fused_dir}")

    print("\nAll done ✓")

if __name__ == "__main__":
    main_build_faiss_index(embed_model)
