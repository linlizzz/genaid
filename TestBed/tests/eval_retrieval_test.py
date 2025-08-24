import json
import argparse
import math
from typing import List, Dict
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from retrieval.guideline_faiss import GuidelineFAISS
from utils import load_paths

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]

def dcg(rels: List[int]) -> float:
    return sum(((2**r - 1) / math.log2(i+2)) for i, r in enumerate(rels))

def ndcg_at_k(ground_truth: List[str], ranked_ids: List[str], k: int) -> float:
    rels = [1 if rid in ground_truth else 0 for rid in ranked_ids[:k]]
    ideal_rels = sorted(rels, reverse=True)
    idcg = dcg(ideal_rels)
    return (dcg(rels) / idcg) if idcg > 0 else 0.0

def mrr_at_k(ground_truth: List[str], ranked_ids: List[str], k: int) -> float:
    for i, rid in enumerate(ranked_ids[:k]):
        if rid in ground_truth:
            return 1.0 / (i + 1)
    return 0.0

def recall_at_k(ground_truth: List[str], ranked_ids: List[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in ground_truth)
    return hits / len(ground_truth)

def precision_at_k(ground_truth: List[str], ranked_ids: List[str], k: int) -> float:
    k = min(k, len(ranked_ids))
    if k == 0:
        return 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in ground_truth)
    return hits / k

def average_precision_at_k(ground_truth: List[str], ranked_ids: List[str], k: int) -> float:
    """Average Precision (AP) at k for a single query"""
    if not ground_truth:
        return 0.0
    hits, sum_precisions = 0, 0.0
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid in ground_truth:
            hits += 1
            sum_precisions += hits / i   # precision at this rank
    return sum_precisions / len(ground_truth)


def mean_average_precision(ground_truths: List[List[str]], rankings: List[List[str]], k: int) -> float:
    """mAP@k across multiple queries"""
    ap_scores = [
        average_precision_at_k(gt, ranked, k)
        for gt, ranked in zip(ground_truths, rankings)
    ]
    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0


def evaluate_guideline_level(db: GuidelineFAISS, notes_jsonl: str, k: int):
    """
    notes_jsonl: clinical_notes.jsonl
      each line: {"note_id": "...", "text": "...", "linked_guideline_ids": ["hoi_04010", "dnd00035", ...]}
    """
    mrr_total = ndcg_total = recall_total = prec_total = map_total = 0.0
    n = 0
    with open(notes_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            query = ex["text"]
            gt_guidelines = ex.get("linked_guideline_ids", [])

            # "grouped by guideline" retrieval
            hits = db.search_grouped_by_guideline(query, top_k_chunks=max(50, k*5), top_m_guidelines=k)
            ranked_ids = [h["guideline_id"] for h in hits]

            mrr_total += mrr_at_k(gt_guidelines, ranked_ids, k)
            ndcg_total += ndcg_at_k(gt_guidelines, ranked_ids, k)
            recall_total += recall_at_k(gt_guidelines, ranked_ids, k)
            prec_total += precision_at_k(gt_guidelines, ranked_ids, k)
            map_total += average_precision_at_k(gt_guidelines, ranked_ids, k)
            n += 1

    if n == 0:
        print("No queries found.")
        return

    print(f"[Guideline-level] Queries: {n}")
    print(f"MRR@{k}:       {mrr_total / n:.4f}")
    print(f"nDCG@{k}:      {ndcg_total / n:.4f}")
    print(f"Recall@{k}:    {recall_total / n:.4f}")
    print(f"Precision@{k}: {prec_total / n:.4f}")
    print(f"MAP@{k}:       {map_total / n:.4f}")

## TO BE IMPROVED: chunk-level retrieval
def evaluate_chunk_level(db: GuidelineFAISS, qrels_jsonl: str, k: int):
    """
    qrels_jsonl:
    each line: {"query": "...", "relevant_chunk_ids": ["hoi_04010_chunk_02", ...]}
    """
    mrr_total = ndcg_total = recall_total = prec_total = map_total = 0.0
    n = 0
    with open(qrels_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            query = ex["query"]
            gt_chunks = ex.get("relevant_chunk_ids", [])

            hits = db.search(query, top_k=k)
            ranked_ids = [h["chunk_id"] for h in hits]

            mrr_total += mrr_at_k(gt_chunks, ranked_ids, k)
            ndcg_total += ndcg_at_k(gt_chunks, ranked_ids, k)
            recall_total += recall_at_k(gt_chunks, ranked_ids, k)
            prec_total += precision_at_k(gt_chunks, ranked_ids, k)
            map_total += average_precision_at_k(gt_chunks, ranked_ids, k)
            n += 1

    if n == 0:
        print("No queries found.")
        return

    print(f"[Chunk-level] Queries: {n}")
    print(f"MRR@{k}:       {mrr_total / n:.4f}")
    print(f"nDCG@{k}:      {ndcg_total / n:.4f}")
    print(f"Recall@{k}:    {recall_total / n:.4f}")
    print(f"Precision@{k}: {prec_total / n:.4f}")
    print(f"MAP@{k}:       {map_total / n:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, default=GUIDELINE_FAISS_DIR)
    ap.add_argument("--file", type=str, default=CLINICAL_NOTES_PATH, help="clinical_notes.jsonl（guideline-level）或 qrels.jsonl（chunk-level）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--level", choices=["guideline", "chunk"], default="guideline",
                    help="guideline/chunk-level labels")
    args = ap.parse_args()

    db = GuidelineFAISS.load(args.index_dir)
    if args.level == "guideline":
        evaluate_guideline_level(db, args.file, args.k)
    else:
        evaluate_chunk_level(db, args.file, args.k)


## Guideline-level retrieval
# python -u TestBed/tests/eval_retrieval_test.py --index_dir faiss_store --file clinical_notes.jsonl --level guideline --k 5
## Chunk-level retrieval
# python -u TestBed/tests/eval_retrieval_test.py --index_dir faiss_store --file qrels.jsonl --level chunk --k 5