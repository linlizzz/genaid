## FAISS index for guideline chunks
import os
import glob
import json
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Optional, Iterable
from sentence_transformers import SentenceTransformer
import glob, os
from utils import load_paths

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms

## E5-based models need special prefixing
def _needs_e5_prefix(model_name: str) -> bool:
    name = model_name.lower()
    return "e5" in name  # e.g. intfloat/multilingual-e5-base

def _prep_passages(texts, use_e5: bool):
    if not use_e5:
        return texts
    return [f"passage: {t}" for t in texts]

def _prep_query(q, use_e5: bool):
    return f"query: {q}" if use_e5 else q


class GuidelineFAISS:
    """
    - Adapt to JSON structure (with chunks)
    - SentenceTransformers generates embeddings
    - FAISS cosine/L2 similarity retrieval
    - Support saving/loading of index and metadata
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        metric: str = "cosine"   # "cosine" or "l2"
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        metric = metric.lower()
        if metric not in {"cosine", "l2"}:
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.metric = metric

        # select index
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

        # metadata corresponds to index
        self.meta = pd.DataFrame(columns=[
            "guideline_id", "chunk_id", "title", "keywords", "content"
        ])

        self._normalized = (self.metric == "cosine")

    # ---------- data import (JSON with chunks) ----------
    def insert_from_jsonl_file(self, jsonl_path: str, batch_size: int = 512):
        """
        Read a single JSONL file (structure described in ../datasets/README.md), and insert each chunk into the index.
        """
        texts = []
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                guideline_id = obj.get("guideline_id")
                title = obj.get("title")
                keywords = obj.get("keywords", [])
                keywords_str = ", ".join(map(str, keywords)) if isinstance(keywords, list) else str(keywords)
                chunks = obj.get("chunks", [])
                if not chunks:
                    continue

                for ch in chunks:
                    chunk_id = ch.get("chunk_id")
                    content = ch.get("content", "")
                    if not content:
                        continue
                    rows.append({
                        "guideline_id": guideline_id,
                        "chunk_id": chunk_id,
                        "title": title,
                        "keywords": keywords_str,
                        "content": content
                    })
                    texts.append(content)

                    # batch encode and add to index, avoid large files taking up memory
                    if len(texts) >= batch_size:
                        self._encode_and_add(texts, rows, batch_size=len(texts))
                        texts, rows = [], []

        # process last batch
        if texts:
            self._encode_and_add(texts, rows, batch_size=len(texts))

        print(f"Inserted from JSONL: {jsonl_path} | total: {self.index.ntotal}")


    def insert_from_jsonl_dir(self, dir_path: str, pattern: str = "*.jsonl", batch_size: int = 512, limit: Optional[int] = None):
        """
        Batch import JSONL files from directory. pattern can be adjusted, e.g. '**/*.jsonl' for recursive.
        """
        files = sorted(glob.glob(os.path.join(dir_path, pattern), recursive=True))
        if limit:
            files = files[:limit]
        for p in files:
            self.insert_from_jsonl_file(p, batch_size=batch_size)

    def _encode_and_add(self, texts: List[str], rows: List[Dict], batch_size: int):
        if not texts:
            return
        all_vecs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            use_e5 = _needs_e5_prefix(self.model_name)
            vecs = self.model.encode(_prep_passages(batch, use_e5), convert_to_numpy=True, show_progress_bar=False)
            # vecs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            if self.metric == "cosine":
                vecs = _l2_normalize(vecs)
            all_vecs.append(vecs.astype("float32"))

        mat = np.vstack(all_vecs)
        self.index.add(mat)
        self.meta = pd.concat([self.meta, pd.DataFrame(rows)], ignore_index=True)

    # ---------- search ----------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Return top_k results:
          - guideline_id, chunk_id, title, keywords, content, score
        larger cosine score or smaller L2 score 
        """
        use_e5 = _needs_e5_prefix(self.model_name)
        q = self.model.encode([_prep_query(query, use_e5)], convert_to_numpy=True)
        # q = self.model.encode([query], convert_to_numpy=True)
        if self.metric == "cosine":
            q = _l2_normalize(q)
        distances, indices = self.index.search(q.astype("float32"), top_k)
        scores = distances[0].tolist()
        idx = indices[0].tolist()
        # print("="*50, "Scores", "="*50)
        # print(scores)
        # print("="*50, "Indices", "="*50)
        # print(idx)

        out = []
        for idx, sc in zip(indices[0], scores):
            if 0 <= idx < len(self.meta):
                row = self.meta.iloc[idx]
                out.append({
                    "guideline_id": row["guideline_id"],
                    "chunk_id": row["chunk_id"],
                    "title": row["title"],
                    "keywords": row["keywords"],
                    "content": row["content"],
                    "score": float(sc),
                    "index_id": int(idx)
                })
        # print("="*50, "Search", "="*50)
        # print(out)
        return out

    def search_grouped_by_guideline(self, query: str, top_k_chunks: int = 20, top_m_guidelines: int = 5) -> List[Dict]:
        """
        First get top_k_chunks most similar chunks, then aggregate by guideline_id, and return the best score and representative chunk for each guideline.
        Scenario: need to do recall "by document".
        """
        hits = self.search(query, top_k=top_k_chunks)
        # aggregate
        groups: Dict[str, Dict] = {}
        for h in hits:
            gid = h["guideline_id"]
            if gid not in groups or self._better(h, groups[gid]):
                groups[gid] = {
                    "guideline_id": gid,
                    "title": h["title"],
                    "keywords": h["keywords"],
                    "best_chunk": h["chunk_id"],
                    "best_score": h["score"],
                    "best_content": h["content"]
                }
        # sort: cosine descending; L2 ascending
        reverse = (self.metric == "cosine")
        sorted_guides = sorted(groups.values(), key=lambda x: x["best_score"], reverse=reverse)
        return sorted_guides[:top_m_guidelines]

    def _better(self, a: Dict, b: Dict) -> bool:
        # compare two hits: larger cosine score or smaller L2 score
        if self.metric == "cosine":
            return a["score"] > b["best_score"]
        else:
            return a["score"] < b["best_score"]

    # ---------- save / load ----------
    def save(self, save_dir: str):
        """
        Save:
          - index.faiss
          - meta.parquet
          - config.json
        """
        _ensure_dir(save_dir)

        path = os.path.join(save_dir, "index.faiss")
        print("Saving index to:", os.path.abspath(path))
        faiss.write_index(self.index, path)
        print("Done faiss.write_index")
        self.meta.to_parquet(os.path.join(save_dir, "meta.parquet"), index=False)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"model_name": self.model_name, "dim": self.dim, "metric": self.metric, "normalized": self._normalized},
                f, ensure_ascii=False, indent=2
            )
        print(f"Saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: str):
        """
        Load the directory saved by save().
        """
        index_path = os.path.join(save_dir, "index.faiss")
        meta_path = os.path.join(save_dir, "meta.parquet")
        conf_path = os.path.join(save_dir, "config.json")
        print(f"Loading index from: {index_path}")
        print(f"Loading meta from: {meta_path}")
        print(f"Loading config from: {conf_path}")

        if not (os.path.exists(index_path) and os.path.exists(meta_path) and os.path.exists(conf_path)):
            raise FileNotFoundError("index.faiss/meta.parquet/config.json is missing")

        with open(conf_path, "r", encoding="utf-8") as f:
            conf = json.load(f)

        obj = cls(model_name=conf["model_name"], metric=conf.get("metric", "cosine"))
        obj.index = faiss.read_index(index_path)
        obj.meta = pd.read_parquet(meta_path)
        obj._normalized = conf.get("normalized", obj.metric == "cosine")

        if obj.index.ntotal != len(obj.meta):
            raise ValueError(f"Index count ({obj.index.ntotal}) != metadata rows ({len(obj.meta)})")

        if conf.get("dim") and conf["dim"] != obj.dim:
            print(f"WARNING: saved dim={conf['dim']} != current model dim={obj.dim}. Ensure model matches!")
        return obj

