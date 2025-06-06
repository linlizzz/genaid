import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import joblib


class GuidelineRetriever:
    def __init__(self, mode="embedding", index_path=None, model_name=None):
        self.mode = mode
        self.index_path = index_path
        self.model_name = model_name

        if self.mode == "bm25":
            self._load_bm25_index()
        elif self.mode == "embedding":
            self._load_embedding_index()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _load_bm25_index(self):
        index = joblib.load(self.index_path)
        self.vectorizer = index["vectorizer"]
        self.matrix = index["matrix"]
        self.chunks = index["chunks"]

    def _load_embedding_index(self):
        with open(self.index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        self.embeddings = np.array(index["embeddings"])
        self.chunks = index["chunks"]
        self.model = SentenceTransformer(index.get("model_name", self.model_name or "sentence-transformers/all-MiniLM-L6-v2"))

    def retrieve(self, query_text, top_k=5):
        if self.mode == "bm25":
            query_vec = self.vectorizer.transform([query_text])
            scores = (self.matrix @ query_vec.T).toarray().squeeze()
        else:  # embedding
            query_emb = self.model.encode([query_text])[0]
            scores = cosine_similarity([query_emb], self.embeddings)[0]

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:top_k]

        # Return top-k results with scores
        results = [{
            "chunk_id": self.chunks[i]["chunk_id"],
            "guideline_id": self.chunks[i]["guideline_id"],
            "content": self.chunks[i]["content"],
            "score": float(scores[i])
        } for i in top_k_idx]

        return results


if __name__ == "__main__":
    # For embedding-based retrieval
    retriever = GuidelineRetriever(
        mode="embedding",
        index_path="retrieval/index/embedding_index.json"
    )

    clinical_note = "Potilas valittaa pitkittynyttä yskää ja hengityksen vinkunaa."
    results = retriever.retrieve(clinical_note, top_k=3)

    for i, res in enumerate(results):
        print(f"Rank {i+1} | Score: {res['score']:.3f}")
        print(f"Chunk ID: {res['chunk_id']}")
        print(f"Content: {res['content'][:300]}...\n")
