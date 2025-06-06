import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))

from retriever import GuidelineRetriever
from tqdm import tqdm

def load_clinical_notes(notes_path):
    notes = []
    with open(notes_path, "r", encoding="utf-8") as f:
        for line in f:
            notes.append(json.loads(line))
    return notes

def save_results(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved retrieval results to: {output_path}")

def main():
    # Settings
    notes_path = "/scratch/work/zhangl9/genaid/TestBed/datasets/clinical_notes.jsonl"
    index_path = "/scratch/work/zhangl9/genaid/TestBed/retrieval/index/embedding_index.json"  # or BM25 index
    output_path = "/scratch/work/zhangl9/genaid/TestBed/retrieval/retrieval_results.jsonl"
    mode = "embedding"  # or "bm25"
    top_k = 5

    retriever = GuidelineRetriever(mode=mode, index_path=index_path)

    notes = load_clinical_notes(notes_path)
    all_results = []

    for note in tqdm(notes, desc="Retrieving guidelines"):
        note_id = note["note_id"]
        text = note["text"]

        retrieved_chunks = retriever.retrieve(text, top_k=top_k)

        all_results.append({
            "note_id": note_id,
            "text": text,
            "retrieved_chunks": retrieved_chunks
        })

    save_results(all_results, output_path)



if __name__ == "__main__":
    main()
