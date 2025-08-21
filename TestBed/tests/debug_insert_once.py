import os, glob, json, sys

import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
sys.path.append("/scratch/work/zhangl9/genaid/TestBed/") 
from retrieval.guideline_faiss import GuidelineFAISS
from utils import load_paths

paths = load_paths()
DATA_DIR = paths["GUIDELINE_JSON_DIR"]   # ← 改成你的目录
PATTERN  = "*.jsonl"         # 需要递归就用这个；不需要就用 "*.jsonl"

def peek_jsonl(path):
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig 兼容 BOM
        for line in f:
            line = line.strip()
            if not line:
                continue
            return line
    return None

def quick_check_one_file(path):
    print(f"\n[检查文件] {path}")
    line = peek_jsonl(path)
    if not line:
        print("  - 文件没有任何非空行（或全部空白）。")
        return False
    first_char = next((c for c in line if not c.isspace()), "")
    if first_char != "{":
        print("  - ⚠️ 这看起来不像 JSONL 的一条完整对象（第一行不是 { 开头）。")
        print("    可能是一个大 JSON 数组或别的格式。展示前 120 字符：")
        print("    ", line[:120])
        return False
    try:
        obj = json.loads(line)
    except Exception as e:
        print("  - ❌ 解析 JSON 失败：", e)
        print("    该行前 200 字符：", line[:200])
        return False

    # 结构检查
    ok = True
    for k in ["guideline_id", "title", "keywords", "chunks"]:
        if k not in obj:
            print(f"  - ❌ 缺少字段: {k}")
            ok = False
    if not ok:
        return False

    chunks = obj.get("chunks") or []
    print(f"  - guideline_id={obj.get('guideline_id')}, title={obj.get('title')}, chunks={len(chunks)}")
    non_empty_chunks = [c for c in chunks if c.get("page_content")]
    print(f"  - 非空 page_content 的 chunk 数量: {len(non_empty_chunks)}")
    if not non_empty_chunks:
        print("  - ❌ 全部 chunk 的 page_content 为空，无法入索引。")
        return False

    sample = non_empty_chunks[0]
    if "chunk_id" not in sample:
        print("  - ❌ chunk 缺少 chunk_id 字段。")
        return False
    return True

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, PATTERN), recursive=True))
    print(f"[匹配文件数] {len(files)} (目录={DATA_DIR}, pattern={PATTERN})")
    if not files:
        print("  - ❌ 没有匹配到任何 .jsonl 文件。请检查路径/通配符。")
        sys.exit(1)

    # 先体检第一份文件结构
    if not quick_check_one_file(files[0]):
        print("\n请先修正数据格式再继续。")
        sys.exit(1)

    # 实测插入一份文件
    db = GuidelineFAISS(model_name="sentence-transformers/all-MiniLM-L6-v2", metric="cosine")
    print("\n[插入前] ntotal=", db.index.ntotal, " meta_rows=", len(db.meta))
    db.insert_from_jsonl_file(files[0], batch_size=64)
    print("[插入后] ntotal=", db.index.ntotal, " meta_rows=", len(db.meta))

    # 如果成功，再批量导入整个目录
    if db.index.ntotal > 0:
        print("\n第一份文件 OK，开始批量导入整个目录...")
        db.insert_from_jsonl_dir(DATA_DIR, pattern=PATTERN, batch_size=512)
        print("[批量后] ntotal=", db.index.ntotal, " meta_rows=", len(db.meta))
    else:
        print("\n❌ 第一份文件都没插入成功，请先修正该文件。")

main()
