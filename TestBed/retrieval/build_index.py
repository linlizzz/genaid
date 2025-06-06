import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Where to store index
INDEX_DIR = "retrieval/index"
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)


def load_guideline_chunks(guideline_file):
    """Load guideline chunks from JSONL."""
    chunks = []
    with open(guideline_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for chunk in item["chunks"]:
                chunks.append({
                    "guideline_id": item["guideline_id"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"]
                })
    return chunks


def build_bm25_index(chunks, output_path):
    """Build sparse BM25-style index using TfidfVectorizer."""
    corpus = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer().fit(corpus)
    X = vectorizer.transform(corpus)

    index = {
        "vectorizer": vectorizer,
        "matrix": X,
        "chunks": chunks
    }
    # Save using joblib
    import joblib
    joblib.dump(index, output_path)
    print(f"✅ Saved BM25 index to {output_path}")


def build_embedding_index(chunks, output_path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Build dense index using sentence embeddings."""
    model = SentenceTransformer(model_name)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    index = {
        "embeddings": embeddings.tolist(),
        "chunks": chunks,
        "model_name": model_name
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"Saved embedding index to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--guideline_path", type=str, default="/scratch/work/zhangl9/genaid/TestBed/datasets/Vältä_viisaasti.jsonl")
    parser.add_argument("--mode", choices=["bm25", "embedding"], default="embedding")
    parser.add_argument("--output_path", type=str,
                        help="Path to save the index file",
                        default=os.path.join(INDEX_DIR, "embedding_index.json"))
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    chunks = load_guideline_chunks(args.guideline_path)

    if args.mode == "bm25":
        build_bm25_index(chunks, args.output_path)
    else:
        build_embedding_index(chunks, args.output_path, args.model_name)
