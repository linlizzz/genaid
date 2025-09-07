import os, json, argparse
from typing import Dict
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/retrieval") 

# config
from utils import read_secrets, load_paths, load_prompt, render_prompt

secrets = read_secrets()
OPENAI_API_KEY = secrets["OPENAI_API_KEY"]
OPENAI_MODEL = secrets["OPENAI_MODEL"]
paths = load_paths()
CLINICAL_NOTES_PATH = paths["CLINICAL_NOTES_PATH"]
PROMPTS_PATH = paths["PROMPTS_PATH"]

# ---------------- Local chat adapter (Transformers) ----------------
def chat_complete_local(
    model_id: str,
    messages,
    max_new_tokens: int = 600,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    device_map: str = "auto",
    dtype: str = "bfloat16",   # "bfloat16" | "float16"
):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    torch_dtype = torch.bfloat16 if dtype.lower() in ("bf16","bfloat16") else torch.float16

    tok = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # 将 messages 渲染为纯文本（依赖各模型自带的 chat template）
    prompt_text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    eos_token_id = tok.eos_token_id
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    inputs = tok([prompt_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    # 仅取新生成的部分
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()

# ---------------- OpenAI adapter (可选) ----------------
class OpenAILLM:
    def __init__(self, model: str, api_key_env="OPENAI_API_KEY"):
        from openai import OpenAI
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"ENV {api_key_env} missing.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens=800, temperature=0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content

# ---------------- Unified LLM wrapper ----------------
class LLM:
    def __init__(self, backend: str, model: str, **kwargs):
        self.backend = backend.lower()
        self.model = model
        self.kwargs = kwargs
        if self.backend == "openai":
            self.client = OpenAILLM(model)
        elif self.backend == "local":
            self.client = None
        elif self.backend == "none":
            self.client = None
        else:
            raise ValueError("backend must be one of: local, openai, none")

    def complete(self, system: str, user: str, max_tokens=800, temperature=0.2) -> str:
        if self.backend == "openai":
            return self.client.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        elif self.backend == "local":
            messages = [
                {"role":"system", "content": system},
                {"role":"user",   "content": user},
            ]
            return chat_complete_local(
                model_id=self.model,
                messages=messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                device_map=self.kwargs.get("device_map","auto"),
                dtype=self.kwargs.get("dtype","bfloat16"),
            )
        else:  # none
            # 调试占位：回显 user 提示尾部
            return user[-1200:]

# ---------------- JSON parsing helper ----------------
def parse_llm_json(raw: str) -> Dict:
    s = raw.strip()
    if s.startswith("```"):
        # 容错处理 ```json ... ```
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()
    try:
        return json.loads(s)
    except Exception:
        # 返回空 schema，避免崩溃
        return {"language":"", "entities":{
            "conditions":[], "symptoms":[], "tests_ordered":[], "test_results":[],
            "diagnoses":[], "medications":[], "allergies":[], "procedures":[],
            "treatment_plan":[], "other_keywords":[]
        }}

# ---------------- CLI main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default=CLINICAL_NOTES_PATH)
    ap.add_argument("--out", default="notes_rewritten_preview.jsonl")
    ap.add_argument("--prompts_dir", default=PROMPTS_PATH)
    ap.add_argument("--backend", default="local", choices=["local","openai","none"])
    ap.add_argument("--model", default="LumiOpen/Poro-34B-chat",
                    help="本地: HF 模型ID;OpenAI: 模型名(如 gpt-4o-mini)") # models = ["LumiOpen/Poro-34B-chat", "BioMistral/BioMistral-7B", "utter-project/EuroLLM-9B-Instruct"]
    ap.add_argument("--device_map", default="auto",
                    help='Transformers device map, e.g., "auto" | "cuda:0"')
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16","float16"])
    ap.add_argument("--maxn", type=int, default=0, help="最多预览多少条(0=全部)")
    ap.add_argument("--max_new_tokens_summary", type=int, default=800)
    ap.add_argument("--max_new_tokens_keywords", type=int, default=600)
    args = ap.parse_args()

    # 加载 prompts
    tpl_sum = load_prompt(os.path.join(args.prompts_dir, "summarize_with_aspects.txt"))
    tpl_kw  = load_prompt(os.path.join(args.prompts_dir, "extract_keywords_json.txt"))

    # LLM
    llm = LLM(args.backend, args.model, device_map=args.device_map, dtype=args.dtype)

    n = 0
    with open(args.notes, "r", encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if args.maxn and n >= args.maxn:
                break
            if not line.strip():
                continue
            ex = json.loads(line)
            note_id = ex.get("note_id")
            text = ex.get("text","").strip()
            if not text:
                continue

            # A) Summary
            sum_prompt = render_prompt(tpl_sum, {"NOTE_TEXT": text})
            summary = llm.complete(
                system="You summarise clinical notes faithfully and concisely.",
                user=sum_prompt,
                max_tokens=args.max_new_tokens_summary,
                temperature=0.0
            )

            # B) Keywords JSON
            kw_prompt = render_prompt(tpl_kw, {"NOTE_TEXT": text})
            kw_raw = llm.complete(
                system="You extract entities and return strict JSON only.",
                user=kw_prompt,
                max_tokens=args.max_new_tokens_keywords,
                temperature=0.0
            )
            kjson = parse_llm_json(kw_raw)

            # C) Combo（摘要 + 关键词串）
            def kw_to_query(kj: Dict) -> str:
                ents = kj.get("entities", {})
                buckets = [("conditions",2.0),("diagnoses",2.0),("symptoms",1.5),
                           ("tests_ordered",1.5),("test_results",1.8),
                           ("medications",1.6),("procedures",1.5),
                           ("treatment_plan",1.7),("allergies",1.3),("other_keywords",1.0)]
                parts=[]
                for k,w in buckets:
                    for v in ents.get(k,[]) or []:
                        v = str(v).strip()
                        if v:
                            parts.append((" "+v)*max(1,int(round(w))))
                return " ".join(parts).strip()

            combo = summary + ("\n\nKEYWORDS: " + kw_to_query(kjson) if kjson else "")

            obj = {
                "note_id": note_id,
                "summary": summary,
                "keywords_json": kjson,
                "combo_query": combo[:2000]
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

            # 控制台预览
            print(f"\n=== {note_id} ===")
            print("SUMMARY:\n", summary[:800])
            print("KEYWORDS(JSON):\n", json.dumps(kjson, ensure_ascii=False))
            print("COMBO (preview):\n", combo[:400], "...")

    print(f"\nSaved {n} items to {args.out}")

if __name__ == "__main__":
    main()
