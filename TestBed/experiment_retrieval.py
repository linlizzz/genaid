#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/retrieval") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 

# config
from utils import load_paths

paths = load_paths()
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]
PROMPTS_PATH = paths["PROMPTS_PATH"]
NOTES_REFINEMENT = paths["NOTES_REFINEMENT"]
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]



# =========================
# 工具函数
# =========================

def load_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def load_notes_refinement(path: str) -> Dict[str, dict]:
    """
    notes_rewritten_preview.jsonl:
      { "note_id": str, "summary": str, "keywords_json": {...}, "combo_query": str }
    """
    mp = {}
    if not path:
        return mp
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            nid = obj.get("note_id")
            if nid:
                mp[nid] = obj
    return mp

def keywords_to_query(kjson: dict) -> str:
    """
    把 keywords_json 里各类实体拼成一个检索串（带软权重，重复以近似加权）
    """
    if not kjson:
        return ""
    ents = (kjson.get("entities") or {})
    buckets = [
        ("conditions", 2.0),
        ("diagnoses", 2.0),
        ("symptoms", 1.5),
        ("tests_ordered", 1.5),
        ("test_results", 1.8),
        ("medications", 1.6),
        ("procedures", 1.5),
        ("treatment_plan", 1.7),
        ("allergies", 1.3),
        ("other_keywords", 1.0),
    ]
    parts = []
    for key, w in buckets:
        arr = ents.get(key) or []
        for v in arr:
            v = str(v).strip()
            if v:
                parts.append((" " + v) * max(1, int(round(w))))
    return " ".join(parts).strip()

def build_query_for_mode(
    mode: str,
    note_text: str,
    note_id: str,
    preview_map: Dict[str, dict]
) -> Tuple[str, dict]:
    """
    返回 (query_text, aux);aux 内含 {summary, keywords_json}
    优先使用预计算的 notes_refinement文档: notes_rewritten_preview.jsonl
    """
    aux = {"summary": "", "keywords_json": {}}
    pre = preview_map.get(note_id)

    if mode == "raw":
        return note_text, aux

    if mode == "summary":
        if pre and pre.get("summary"):
            aux["summary"] = pre["summary"]
            return pre["summary"], aux
        else:
            # 没有预计算就退回原文（不调用在线 LLM）
            return note_text, aux

    if mode == "keywords":
        if pre and pre.get("keywords_json"):
            aux["keywords_json"] = pre["keywords_json"]
            q = keywords_to_query(pre["keywords_json"])
            return q if q else note_text, aux
        else:
            return note_text, aux

    if mode == "combo":
        if pre:
            summary = pre.get("summary", "")
            aux["summary"] = summary
            if pre.get("combo_query"):
                return pre["combo_query"], aux
            else:
                kj = pre.get("keywords_json", {})
                aux["keywords_json"] = kj
                qk = keywords_to_query(kj)
                combo = (summary or "") + (("\n\nKEYWORDS: " + qk) if qk else "")
                return combo if combo.strip() else note_text, aux
        else:
            return note_text, aux

    # 默认返回 raw
    return note_text, aux

# -------------------------
# FAISS 索引封装
# -------------------------
class FaissDenseIndex:
    def __init__(self, index, meta_df: pd.DataFrame, metric: str = "cosine"):
        self.index = index
        self.meta = meta_df.reset_index(drop=True)
        self.metric = metric

    @classmethod
    def load(cls, dirpath: str):
        idx_path = os.path.join(dirpath, "index.faiss")
        meta_path = os.path.join(dirpath, "meta.parquet")
        cfg_path  = os.path.join(dirpath, "config.json")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"index.faiss not found in {dirpath}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.parquet not found in {dirpath}")
        index = faiss.read_index(idx_path)
        meta = pd.read_parquet(meta_path)
        metric = "cosine"
        if os.path.exists(cfg_path):
            try:
                cfg = json.load(open(cfg_path,"r",encoding="utf-8"))
                metric = cfg.get("metric","cosine")
            except Exception:
                pass
        return cls(index, meta, metric)

    def search(self, qvec: np.ndarray, top_k: int = 10):
        """
        qvec: shape (d,) or (1,d) float32 L2-normalized if cosine
        returns: list of dict rows with score + meta
        """
        if qvec.ndim == 1:
            qvec = qvec[None, :]
        D, I = self.index.search(qvec.astype("float32"), top_k)
        out = []
        for idx, score in zip(I[0], D[0]):
            row = self.meta.iloc[int(idx)]
            out.append({
                "score": float(score),
                "guideline_id": row.get("guideline_id"),
                "chunk_id": row.get("chunk_id"),
                "title": row.get("title"),
                "keywords": row.get("keywords"),
                "content": row.get("content"),
                "index_row": int(idx),
            })
        return out

# -------------------------
# metrics
# -------------------------
def dcg_at_k(rels: List[int], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(rels[:k], start=1):
        if r > 0:
            dcg += (2**r - 1) / np.log2(i + 1)
    return dcg

def ndcg_at_k(rels: List[int], k: int) -> float:
    dcg = dcg_at_k(rels, k)
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(hits_binary: List[int], k: int) -> float:
    for i, h in enumerate(hits_binary[:k], start=1):
        if h:
            return 1.0 / i
    return 0.0

def precision_at_k(hits_binary: List[int], k: int) -> float:
    return sum(hits_binary[:k]) / float(k) if k > 0 else 0.0

def recall_at_k(hits_binary: List[int], gt_count: int, k: int) -> float:
    return sum(hits_binary[:k]) / float(gt_count) if gt_count > 0 else 0.0

def ap_at_k(hits_binary: List[int], gt_count: int, k: int) -> float:
    """
    Average Precision@K:
    AP@K = (sum_{i=1..K} Precision@i * rel_i) / min(gt_count, K)
    其中 rel_i ∈ {0,1} 表示第 i 个命中是否相关。
    当 gt_count=0 时返回 0.
    """
    if gt_count <= 0:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, rel in enumerate(hits_binary[:k], start=1):
        if rel:
            hits += 1
            sum_prec += hits / float(i)  # Precision@i
    denom = float(min(gt_count, k))
    return sum_prec / denom if denom > 0 else 0.0


# -------------------------
# 主流程
# -------------------------
def main_experiment_retrieval(model_faiss_dir_name: str, query_model: str):
    ap = argparse.ArgumentParser()
    # 路径：默认采用你提供的常量，可由命令行覆盖
    ap.add_argument("--notes", type=str, default=CLINICAL_NOTES_PATH,
                    help="clinical_notes.jsonl(含 note_id, text, linked_guideline_ids)")
    ap.add_argument("--notes_refinement", type=str, default=NOTES_REFINEMENT,
                    help="notes_rewritten_preview.jsonl(含 note_id, summary, keywords_json, combo_query)")
    ap.add_argument("--faiss_root", type=str, default=GUIDELINE_FAISS_DIR,
                    help="FAISS 根目录（其下有各模型/子索引目录）")
    # 选择要对比的 embedding 子目录（默认三种）
    ap.add_argument("--use_chunk", action="store_true", help="包含 chunk_only 子目录")
    ap.add_argument("--use_concat", action="store_true", help="包含 concat_titlekw_chunk 子目录")
    ap.add_argument("--use_fused", action="store_true", help="包含 fused_alpha1p0_beta0p6_gamma0p4 子目录")
    # 查询输入模式
    ap.add_argument("--modes", type=str, default="raw,summary,keywords,combo",
                    help="逗号分隔:raw,summary,keywords,combo")
    # 模型与检索参数
    # ap.add_argument("--query_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="用于编码查询的 SentenceTransformers 模型")
    ap.add_argument("--top_k", type=int, default=10, help="评测@K(同时也用于检索返回 K)")
    # 输出
    ap.add_argument("--out_detail", type=str, default="exp_details.jsonl")
    ap.add_argument("--out_summary", type=str, default="exp_summary.csv")
    args = ap.parse_args()

    # 默认：若未显式选择，则启用三种（为了显式可控，这里采用“至少选一个”的逻辑）
    if not (args.use_chunk or args.use_concat or args.use_fused):
        args.use_chunk = args.use_concat = args.use_fused = True

    # 读取 notes 与 notes_refinement
    if not args.notes:
        raise ValueError("请提供 --notes 或在脚本顶部 paths['CLINICAL_NOTES_PATH'] 设置路径")
    notes = load_jsonl(args.notes)
    preview_map = load_notes_refinement(args.notes_refinement) if args.notes_refinement else {}

    if len(notes) == 0:
        raise ValueError("notes 文件为空或解析失败")

    # 组装 faiss 子目录
    faiss_variants = []
    # 按你当前模型 all-MiniLM-L6-v2 的目录名推断：build 脚本默认是把 / -> _, . -> p
    # 但更通用的方式：直接在 root 下递归查找包含 index.faiss 的“叶子目录”。
    # 这里根据你的需求，直接拼接三大固定子目录：
    model_root_candidates = []
    '''
    # 自动查找包含 all-MiniLM-L6-v2 的目录
    for name in os.listdir(args.faiss_root):
        if "MiniLM" in name or "minilm" in name.lower():
            model_faiss_dir_name = name
            model_root_candidates.append(os.path.join(args.faiss_root, name))
    if not model_root_candidates:
        # 若未找到，就退而直接用 root（假设你把三种子目录直接放 root 下）
        model_root_candidates = [args.faiss_root]
    '''
    
    model_root_candidates = [os.path.join(args.faiss_root, model_faiss_dir_name)]
    print(f"[debug] model_root_candidates: {model_root_candidates}")

    # 在候选模型目录下，按开关加入三种子目录
    for model_root in model_root_candidates:
        if args.use_chunk:
            d = os.path.join(model_root, "chunk_only")
            if os.path.exists(os.path.join(d, "index.faiss")):
                faiss_variants.append(("chunk", d))
        if args.use_concat:
            d = os.path.join(model_root, "concat_titlekw_chunk")
            if os.path.exists(os.path.join(d, "index.faiss")):
                faiss_variants.append(("concat", d))
        if args.use_fused:
            # 注意：你的 fused 目录名可能因权重不同而变化；这里匹配默认名称
            d = os.path.join(model_root, "fused_alpha1p0_beta0p6_gamma0p4")
            if os.path.exists(os.path.join(d, "index.faiss")):
                faiss_variants.append(("fused10604", d))

    if not faiss_variants:
        raise FileNotFoundError("在 faiss_root 下没有找到包含 index.faiss 的子目录，请检查路径。")

    print("[FAISS variants]")
    for name, d in faiss_variants:
        print(f"  - {name} -> {d}")

    # 加载 FAISS 索引们
    indices = []
    for name, d in faiss_variants:
        print(f"[load] {name}")
        indices.append((name, FaissDenseIndex.load(d)))

    # 查询编码模型
    print(f"[query encoder] loading {query_model}")
    qmodel = SentenceTransformer(query_model)

    # 评测主循环
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    detail_f = open(f"results/{model_faiss_dir_name}/{args.out_detail}", "w", encoding="utf-8")

    rows_summary = []  # 聚合汇总
    # 准备 notes 的 ground truth
    note_items = []
    for obj in notes:
        nid = obj.get("note_id")
        text = obj.get("text", "") or obj.get("note_text", "")
        gt = obj.get("linked_guideline_ids") or obj.get("gold_guideline_ids") or []
        gt = [str(x) for x in gt if str(x).strip()]
        if nid and text.strip():
            note_items.append((nid, text.strip(), gt))

    print(f"[notes] total valid notes: {len(note_items)}")
    if len(note_items) == 0:
        raise ValueError("没有有效的 note (缺少 note_id 或 text)")

    for (emb_name, db) in indices:
        for mode in modes:
            print(f"\n[run] embedding={emb_name}  mode={mode}")
            all_metrics = {"MRR": [], "nDCG": [], "Recall": [], "Precision": [], "mAP": []}
            for nid, text, gt_ids in tqdm(note_items, desc=f"{emb_name}/{mode}"):
                # 构造查询
                qtext, aux = build_query_for_mode(mode, text, nid, preview_map)

                print(f"[debug] mode: {mode}, note_id: {nid}, qtext: {qtext}")

                if not qtext.strip():
                    # 查询为空则跳过或计为 0
                    row_detail = {
                        "note_id": nid, "mode": mode, "embedding": emb_name,
                        "query_used": "", "top_k": [], "metrics": {"MRR":0,"nDCG":0,"Recall":0,"Precision":0},
                        "gt": gt_ids
                    }
                    detail_f.write(json.dumps(row_detail, ensure_ascii=False) + "\n")
                    for k in all_metrics: all_metrics[k].append(0.0)
                    continue

                # 编码查询、L2 归一化（配合余弦）
                q = qmodel.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

                # 检索
                hits = db.search(q[0], top_k=args.top_k)

                print(f"[debug] mode={mode} hits_len={len(hits)} top_guidelines={[h['guideline_id'] for h in hits[:5]]}")

                # 将命中里的 guideline_id 映射为文档级（去重保持顺序）
                ranked_guidelines = []
                seen = set()
                for h in hits:
                    gid = h["guideline_id"]
                    if gid not in seen:
                        ranked_guidelines.append(gid)
                        seen.add(gid)

                # binary relevance 向量（与 ground truth 比对）
                gt_set = set(gt_ids)
                rel = [1 if (g in gt_set) else 0 for g in ranked_guidelines]
                mrr = mrr_at_k(rel, args.top_k)
                ndcg = ndcg_at_k(rel, args.top_k)
                prec = precision_at_k(rel, args.top_k)
                rec = recall_at_k(rel, len(gt_set), args.top_k) if len(gt_set) > 0 else 0.0
                ap  = ap_at_k(rel, len(gt_set), args.top_k)

                all_metrics["MRR"].append(mrr)
                all_metrics["nDCG"].append(ndcg)
                all_metrics["Precision"].append(prec)
                all_metrics["Recall"].append(rec)
                all_metrics["mAP"].append(ap)

                row_detail = {
                    "note_id": nid,
                    "embedding": emb_name,
                    "mode": mode,
                    "query_used": qtext[:2000],
                    "gt": gt_ids,
                    "top_k": [{"guideline_id": h["guideline_id"], "score": h["score"], "title": h["title"]} for h in hits]
                }
                detail_f.write(json.dumps(row_detail, ensure_ascii=False) + "\n")

            # 聚合
            summ = {
                "embedding": emb_name,
                "input_mode": mode,
                "N": len(note_items),
                "K": args.top_k,
                "MRR": float(np.mean(all_metrics["MRR"])) if all_metrics["MRR"] else 0.0,
                "nDCG": float(np.mean(all_metrics["nDCG"])) if all_metrics["nDCG"] else 0.0,
                "Precision": float(np.mean(all_metrics["Precision"])) if all_metrics["Precision"] else 0.0,
                "Recall": float(np.mean(all_metrics["Recall"])) if all_metrics["Recall"] else 0.0,
                "mAP": float(np.mean(all_metrics["mAP"])) if all_metrics["mAP"] else 0.0,
            }
            rows_summary.append(summ)
            print(f"[summary] {summ}")

    detail_f.close()
    pd.DataFrame(rows_summary).to_csv(f"results/{model_faiss_dir_name}/{args.out_summary}", index=False)
    print(f"\n[done] details -> results/{model_faiss_dir_name}/{args.out_detail}")
    print(f"[done] summary -> results/{model_faiss_dir_name}/{args.out_summary}")
    print("Tips: 使用 export_latex_tables.py / plot_experiment_results.py 进一步出表与画图。")

if __name__ == "__main__":
    main_experiment_retrieval(model_faiss_dir_name, query_model)
