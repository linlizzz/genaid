import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from retrieval.guideline_faiss import GuidelineFAISS
from retrieval.hybrid_rrf_search import HybridRetriever
from retrieval.cross_encoder_rerank import Reranker
import json
import argparse
from utils import load_paths

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_index", type=bool, default=False)
    ap.add_argument("--hybrid_retrieval", type=bool, default=True)
    ap.add_argument("--rerank", type=bool, default=True)
    args = ap.parse_args()

    if not args.build_index: # Build the index (batch import JSONL from directory)
        db = GuidelineFAISS(model_name="sentence-transformers/all-MiniLM-L6-v2", metric="cosine")
        print("Done GuidelineFAISS")
        db.insert_from_jsonl_dir(GUIDELINE_JSON_DIR, pattern="*.jsonl", batch_size=512)
        print("Done insert_from_jsonl_dir")
        db.save(GUIDELINE_FAISS_DIR)  # save index and metadata
        print("Done save", "\n")
    # Load and query directly (no need to recompute vectors)
    db = GuidelineFAISS.load(GUIDELINE_FAISS_DIR)

    # query = "Vähäoireisilla nielukipupotilailla"
    with open(CLINICAL_NOTES_PATH, "r") as f:
        first_note = json.loads(f.readline())
        query = first_note["text"]
    print("="*50, "Query", "="*50)
    print(query, "\n")
    
    # a) Return the most similar chunk
    # print("ntotal vectors in FAISS =", db.index.ntotal)   # the number of vectors in FAISS
    # print("meta rows =", len(db.meta), "\n")   # the number of rows in the metadata

    if args.hybrid_retrieval:
        hy = HybridRetriever(db)
        hits = hy.search(query, top_k=10, dense_k=50, bm25_k=50)
        if args.rerank:
            rr = Reranker() # cross-encoder reranking
            hits = rr.rerank(query, hits, top_k=10)
    else:
        hits = db.search(query, top_k=5)
    for h in hits:
        print(f"[{h['score']:.4f}] {h['guideline_id']} | {h['chunk_id']} | {h['title']}")
        print(h["content"], "\n")

### TO DO: concatenate title + keywords with chunk content into BM25 corpus for better sparse retrieval




'''

# b) Aggregate by guideline (by document, select representative chunk)
docs = db2.search_grouped_by_guideline(query, top_k_chunks=30, top_m_guidelines=5)
for d in docs:
    print(f"[best={d['best_score']:.4f}] {d['guideline_id']} | {d['title']} | chunk={d['best_chunk']}")
    print(d["best_content"], "\n")

'''