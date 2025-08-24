## end-to-end RAG demo，patient story → summarizer → retriever（FAISS/Hybrid/rerank） → guideline alignment
import os
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from dataclasses import dataclass
from typing import Dict, List, Optional

from guideline_faiss import GuidelineFAISS
from guideline_retriever import GuidelineRetriever

# config
from utils import read_secrets, load_paths

secrets = read_secrets()
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
OPENAI_MODEL = secrets["OPENAI_MODEL"]

paths = load_paths()
GUIDELINE_JSON_DIR = paths["GUIDELINE_JSON_DIR"]
GUIDELINE_FAISS_DIR = paths["GUIDELINE_FAISS_DIR"]
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]

@dataclass
class LLMConfig:
    # "openai" or "none" (placeholder/debug, return truncated text)
    backend: str = "openai"
    # OpenAI
    openai_model: str = OPENAI_MODEL
    openai_api_key_env: str = OPENAI_API_KEY
    # avoid too long context
    max_tokens_per_chunk: int = 320
    final_k: int = 6                # RAG final chunks

@dataclass
class RetrievalConfig:
    mode: str = "hybrid"            # "dense" | "hybrid"
    dense_k: int = 300
    bm25_k: int = 600
    do_rerank: bool = True          # need to configure cross-encoder in retriever
    final_k: int = 6
    mmr: bool = True

PROMPT_DIR = "prompts/"
PROMPT_SUMMARY = os.path.join(PROMPT_DIR, "summarize_patient_story.txt")
PROMPT_ALIGN   = os.path.join(PROMPT_DIR, "guideline_alignment.txt")

FAISS_DIR = "/scratch/work/zhangl9/genaid/TestBed/retrieval/faiss_store/"  # index directory

# prompt utils

def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def render_prompt(tpl: str, mapping: Dict[str, str]) -> str:
    out = tpl
    for k, v in mapping.items():
        out = out.replace(f"{{{{{k}}}}}", v)
    return out

# LLM client

class LLM:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.backend = cfg.backend.lower()
        if self.backend == "openai":
            from openai import OpenAI
            api_key = os.environ.get(cfg.openai_api_key_env)
            if not api_key:
                raise RuntimeError(f"Missing {cfg.openai_api_key_env} in environment.")
            self.client = OpenAI(api_key=api_key)
        elif self.backend == "none":
            self.client = None
        else:
            raise ValueError("Unsupported backend. Use 'openai' or 'none'.")

    def complete(self, system_prompt: str, user_prompt: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
        if self.backend == "none":
            # 调试用占位：返回 user_prompt 最后 1200 字符
            return user_prompt[-1200:]
        # OpenAI
        resp = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content

# Pipeline

class RAGPipeline:
    def __init__(self, llm_cfg: LLMConfig, ret_cfg: RetrievalConfig, cross_encoder_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # 1) load FAISS index
        self.db = GuidelineFAISS.load(FAISS_DIR)

        # 2) build unified retriever (hybrid retrieval + optional rerank)
        self.retriever = GuidelineRetriever(
            self.db,
            bm25_on=(ret_cfg.mode == "hybrid"),
            rrf_k=60,
            cross_encoder_name=(cross_encoder_name if ret_cfg.do_rerank else None),
        )

        # 3) LLM
        self.llm = LLM(llm_cfg)
        self.llm_cfg = llm_cfg
        self.ret_cfg = ret_cfg

        # 4) load prompts
        self.tpl_summary = load_prompt(PROMPT_SUMMARY)
        self.tpl_align   = load_prompt(PROMPT_ALIGN)

    # Step 1: summarise patient story
    def summarise(self, note_text: str) -> str:
        prompt = render_prompt(self.tpl_summary, {"NOTE_TEXT": note_text})
        # no extra system instruction, just use user prompt; can add light system if needed
        return self.llm.complete(system_prompt="You summarise clinical notes faithfully and concisely.",
                                 user_prompt=prompt,
                                 max_tokens=700, temperature=0.2)

    # Step 2: use summary as query, retrieve and build RAG payload
    def retrieve(self, summary_text: str) -> Dict:
        payload = self.retriever.build_rag_payload(
            query=summary_text,
            mode=self.ret_cfg.mode,
            final_k=self.llm_cfg.final_k,
            max_tokens_per_chunk=self.llm_cfg.max_tokens_per_chunk,
            top_k=self.ret_cfg.final_k,         # for dense mode fallback
            dense_k=self.ret_cfg.dense_k,
            bm25_k=self.ret_cfg.bm25_k,
            do_rerank=self.ret_cfg.do_rerank,
            mmr=self.ret_cfg.mmr,
        )
        return payload

    # Step 3: use "summary + guideline snippets" to align with guidelines
    def align_with_guidelines(self, summary_text: str, rag_payload: Dict) -> str:
        context_text = rag_payload["context_text"]
        prompt = render_prompt(self.tpl_align, {
            "SUMMARY_TEXT": summary_text,
            "GUIDELINE_CONTEXT": context_text
        })
        return self.llm.complete(
            system_prompt="You are a medical expert who grounds every statement in guideline evidence.",
            user_prompt=prompt,
            max_tokens=900,
            temperature=0.2
        )

    # One-shot: input patient story, return (summary, answer, sources)
    def run_once(self, note_text: str) -> Dict:
        summary = self.summarise(note_text)
        rag_payload = self.retrieve(summary)
        answer = self.align_with_guidelines(summary, rag_payload)
        return {
            "summary": summary,
            "answer": answer,
            "sources": rag_payload["sources"],  # [{guideline_id, chunk_id, title, keywords}, ...]
        }

# single & batch

# ---------------- CLI main ----------------

def main_single():
    # config
    llm_cfg = LLMConfig(
        backend="openai",            # or "none" (placeholder/debug)
        openai_model="gpt-4o-mini",
        max_tokens_per_chunk=320,
        final_k=6
    )
    ret_cfg = RetrievalConfig(
        mode="hybrid",               # "dense" | "hybrid"
        dense_k=300,
        bm25_k=600,
        do_rerank=True,
        final_k=6,
        mmr=True
    )

    pipe = RAGPipeline(llm_cfg, ret_cfg, cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # single
    note_text = "Tulosyy: flunssa. \nEsitiedot: Kyseess\u00e4 perusterve 35-v nainen, ei s\u00e4\u00e4n l\u00e4\u00e4kityksi\u00e4. Nyt 2 vko ajan flunssaa. Alkanut kurkkukivulla ja kuumeella. Nyt l\u00e4hinn\u00e4 tukkoinen, painetta poskionteloiden alueella. Ei korvakipua, ei en\u00e4\u00e4 kuumeilua. \nNykytila: Yt hyv\u00e4, sat 98%, hf rauhallinen. Syd\u00e4mest\u00e4 ei ausk poikkeavaa, keuhkoista limaiset, karkeat rahinat l.a. basaalisesti. Siistiytyy yskimisen j\u00e4lkeen. korvat terveet, hieman alipaineiset. Nielu siisti. Kaulalla palp. reakt suurentuneet imusolmukkeet. RHA siisti, sinusscan -/-. L\u00e4mp\u00f6 36.5. \nSuunnitelma: Vaikutelma edelleen virustaudista. Ei bakteeri-infektioon viittaavaa. Oirehoitona Nasonex, acriseu. Lis\u00e4ksi kipul\u00e4\u00e4ke tarv. Suositeltu seesami\u00f6ljy\u00e4 kostuttamaa. Uusi yhteys jos vointi heikkenee tai ei helpota. SVA 3pv\u00e4. \nDiagnoosi: J06.9 - M\u00e4\u00e4ritt\u00e4m\u00e4t\u00f6n akuutti yl\u00e4hengitystieinfektio."
    out = pipe.run_once(note_text)

    print("\n--- SUMMARY ---\n")
    print(out["summary"])
    print("\n--- ANSWER (Guideline-aligned) ---\n")
    print(out["answer"])
    print("\n--- SOURCES ---\n")
    for s in out["sources"]:
        print(f"- {s['guideline_id']} | {s['chunk_id']} | {s['title']}")


def main_batch():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default="clinical_notes.jsonl", help="输入 JSONL（字段：note_id, text, linked_guideline_ids 可选）")
    ap.add_argument("--output", type=str, default="rag_outputs.jsonl",   help="输出 JSONL（逐条追加）")
    ap.add_argument("--faiss_dir", type=str, default="faiss_store")
    ap.add_argument("--prompts_dir", type=str, default="prompts")
    ap.add_argument("--backend", type=str, default="openai", choices=["openai","none"])
    ap.add_argument("--model",   type=str, default="gpt-4o-mini")
    ap.add_argument("--start",   type=int, default=0, help="从第几行开始（便于断点续跑）")
    ap.add_argument("--maxn",    type=int, default=0, help="最多处理多少条（0 表示全部）")
    ap.add_argument("--final_k", type=int, default=6)
    ap.add_argument("--max_tokens_per_chunk", type=int, default=320)
    ap.add_argument("--mode",    type=str, default="hybrid", choices=["dense","hybrid"])
    ap.add_argument("--dense_k", type=int, default=300)
    ap.add_argument("--bm25_k",  type=int, default=600)
    ap.add_argument("--no_rerank", action="store_true", help="禁用交叉编码器重排")
    ap.add_argument("--no_mmr",    action="store_true", help="禁用 MMR 去冗余")
    ap.add_argument("--cross_encoder", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    args = ap.parse_args()

    llm_cfg = LLMConfig(
        backend=args.backend,
        openai_model=args.model,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        final_k=args.final_k
    )
    ret_cfg = RetrievalConfig(
        mode=args.mode,
        dense_k=args.dense_k,
        bm25_k=args.bm25_k,
        do_rerank=(not args.no_rerank),
        final_k=args.final_k,
        mmr=(not args.no_mmr)
    )

    pipe = BatchRAG(
        faiss_dir=args.faiss_dir,
        prompts_dir=args.prompts_dir,
        llm_cfg=llm_cfg,
        ret_cfg=ret_cfg,
        cross_encoder_name=(None if args.no_rerank else args.cross_encoder),
    )

    # 打开输出文件（追加写，便于断点续跑）
    processed = 0
    skipped = 0
    line_no = -1

    # 若输出已存在，可选地记录已处理的 note_id，避免重复
    existing_ids = set()
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f_out_old:
            for l in f_out_old:
                try:
                    obj = json.loads(l)
                    if "note_id" in obj:
                        existing_ids.add(obj["note_id"])
                except Exception:
                    continue

    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(args.output, "a", encoding="utf-8") as f_out:

        for line in f_in:
            line_no += 1
            if line_no < args.start:
                continue
            if args.maxn and processed >= args.maxn:
                break

            line = line.strip()
            if not line:
                continue

            try:
                ex = json.loads(line)
            except Exception as e:
                print(f"[skip line {line_no}] JSON parse error: {e}")
                skipped += 1
                continue

            note_id = ex.get("note_id", f"note_{line_no}")
            if note_id in existing_ids:
                # 已存在，跳过
                continue

            note_text = ex.get("text", "").strip()
            if not note_text:
                print(f"[skip line {line_no}] empty text for note_id={note_id}")
                skipped += 1
                continue

            try:
                # A) summarise
                summary = pipe.summarise(note_text)

                # B) retrieve
                payload = pipe.retrieve_payload(summary)
                context_text = payload["context_text"]
                sources = payload["sources"]

                # C) align
                answer = pipe.align(summary, context_text)

                out_obj = {
                    "note_id": note_id,
                    "summary": summary,
                    "answer": answer,
                    "sources": sources,
                }
            except Exception as e:
                out_obj = {
                    "note_id": note_id,
                    "error": str(e)
                }

            # 逐行输出，随时可中断
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            f_out.flush()
            processed += 1

            if processed % 10 == 0:
                print(f"[progress] processed={processed}, last_note_id={note_id}")

    print(f"Done. processed={processed}, skipped={skipped}, start={args.start}")
    if existing_ids:
        print(f"Skipped {len(existing_ids)} already-written note_ids in {args.output}.")
        
if __name__ == "__main__":
    main_single()
    # main_batch()



