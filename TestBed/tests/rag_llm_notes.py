import os
from openai import OpenAI
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from retrieval.guideline_faiss import GuidelineFAISS
from utils import load_paths

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def rag_answer(note_text: str, db: GuidelineFAISS, top_k: int = 5, max_tokens_per_chunk: int = 300):
    """
    输入临床笔记 note_text，检索 guideline，调用 LLM 给出答案
    """
    # Step 1: 检索相关 chunks
    retrieved = db.search_rag(
        query=note_text,
        top_k=top_k,
        max_tokens_per_chunk=max_tokens_per_chunk
    )

    # Step 2: 拼接成 RAG 上下文
    context_blocks = []
    for r in retrieved:
        block = f"[{r['guideline_id']} | {r['chunk_id']} | {r['title']}]\n{r['content']}"
        context_blocks.append(block)
    context_text = "\n\n---\n\n".join(context_blocks)

    # 构造 Prompt
    prompt = f"""
You are a medical assistant. Answer the clinical question based on the following guidelines.

Clinical note:
{note_text}

Relevant guideline chunks:
{context_text}

Answer the note by summarizing key points and aligning them with guidelines.
Cite guideline_id and chunk_id when relevant.
    """.strip()

    # Step 3: 调用 LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 你也可以用 gpt-4o / gpt-4-turbo
        messages=[
            {"role": "system", "content": "You are a medical expert assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.2
    )

    # Step 4: 输出
    answer = response.choices[0].message.content
    return answer, retrieved


if __name__ == "__main__":
    # 加载已有索引
    db = GuidelineFAISS.load("faiss_store")

    # 输入一条临床笔记
    note_text = "Potilaalla on pitkittynyt yskä ja kuume, epäillään pneumoniaa."

    answer, retrieved = rag_answer(note_text, db, top_k=3)

    print("\n--- LLM Answer ---\n")
    print(answer)

    print("\n--- Sources ---\n")
    for r in retrieved:
        print(f"- {r['guideline_id']} | {r['chunk_id']} | {r['title']}")
