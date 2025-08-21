# hybrid_retriever.py
import re, unicodedata
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

def _normalize(txt: str) -> str:
    # Clean text: lowercase, remove Markdown table vertical bars, normalize accents, remove extra whitespace
    t = txt.lower()
    t = re.sub(r"\|+", " ", t)
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _tokenize(txt: str) -> List[str]:
    # Simple tokenization (can be replaced with a stronger tokenizer/stopwords)
    return re.findall(r"[a-zåäöA-ZÅÄÖ0-9\-]+", txt.lower())

class HybridRetriever:
    """
    Combine GuidelineFAISS (dense) with BM25 (sparse):
    - First retrieve topN for each
    - RRF fusion (score = Σ 1/(k + rank))
    - Return top_k results (with source and score)
    """
    def __init__(self, db, bm25_k: int = 1_000, rrf_k: int = 60):
        self.db = db
        self.rrf_k = rrf_k

        # Build BM25 corpus (use chunk content; also concatenate title/keywords for better recall)
        self.corpus = [ _normalize(c) for c in db.meta["content"].astype(str).tolist() ]
        self.tokenized = [ _tokenize(c) for c in self.corpus ]
        self.bm25 = BM25Okapi(self.tokenized)
        # Record index_id -> meta row
        self.id2row = db.meta  # directly reference

    def search(self, query: str, top_k: int = 10, dense_k: int = 50, bm25_k: int = 50) -> List[Dict]:
        # Dense retrieval
        dense_hits = self.db.search(query, top_k=dense_k)  # Return with index_id. Add index_id to GuidelineFAISS.search if not already.
        # If search doesn't return index_id, add "index_id": int(idx) to GuidelineFAISS.search

        # Sparse retrieval
        q_norm = _normalize(query)
        q_tok = _tokenize(q_norm)
        scores = self.bm25.get_scores(q_tok)  # length = corpus size
        # Take top bm25_k indices
        bm25_idx = scores.argsort()[-bm25_k:][::-1]
        bm25_hits = [{"index_id": int(i), "bm25_score": float(scores[i])} for i in bm25_idx]

        # RRF fusion
        rrf = {}
        # Dense: give rank sequentially
        for rank, h in enumerate(dense_hits, start=1):
            idx = h["index_id"]
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        # Sparse
        for rank, h in enumerate(bm25_hits, start=1):
            idx = h["index_id"]
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        # Sort and take top_k
        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Assemble results
        out = []
        for idx, s in fused:
            row = self.id2row.iloc[idx]
            out.append({
                "guideline_id": row["guideline_id"],
                "chunk_id": row["chunk_id"],
                "title": row["title"],
                "keywords": row["keywords"],
                "content": row["content"],
                "fused_score": float(s),
                "index_id": int(idx)
            })
        return out
