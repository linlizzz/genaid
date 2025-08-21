# rerank.py
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list, top_k: int = 10):
        # candidates: [{"content": "...", "index_id": ... , ...}, ...]
        pairs = [(query, c["content"]) for c in candidates]
        scores = self.model.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        out = []
        for c, sc in ranked:
            c = dict(c)
            c["rerank_score"] = float(sc)
            out.append(c)
        return out
