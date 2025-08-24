## unified retriever (dense/hybrid/rerank + MMR + RAG payload)
import re, unicodedata, math
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

# --------- 文本预处理 / 实用函数 ---------
def normalize_text(txt: str) -> str:
    t = txt.lower()
    t = re.sub(r"\|+", " ", t)  # 去掉 Markdown 表格竖线
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def simple_tokenize(txt: str) -> List[str]:
    return re.findall(r"[a-zåäöA-ZÅÄÖ0-9\-]+", txt.lower())

def mmr_select(candidates: List[Dict], top_k: int, lambda_mult: float = 0.7, text_key: str = "content") -> List[Dict]:
    """
    最大边际相关性（MMR）去冗余选择。这里用候选的 index_id 作为“去重”依据，
    如果需要基于向量相似度做 MMR，可在 GuidelineFAISS 暴露 embeddings 再实现。
    先给你一个“近似版”：按分数挑高分，并避免相同 guideline 的多个近邻同时入选（文档级多样性）。
    """
    selected, seen_guides = [], set()
    for c in sorted(candidates, key=lambda x: x.get("fused_score", x.get("rerank_score", x.get("score", 0))), reverse=True):
        gid = c.get("guideline_id")
        if gid in seen_guides:
            # 以文档为单位的多样性，避免同一指南的块占满配额
            continue
        selected.append(c)
        seen_guides.add(gid)
        if len(selected) >= top_k:
            break
    # 如果去重导致数量不足，补齐
    if len(selected) < top_k:
        pool = [c for c in candidates if c not in selected]
        for c in pool:
            selected.append(c)
            if len(selected) >= top_k:
                break
    return selected

def truncate_for_rag(text: str, max_tokens: int, tokenizer=None) -> str:
    if not max_tokens or max_tokens <= 0:
        return text
    if tokenizer is None:
        # 粗略：1 token ≈ 4 字符
        return text[: max_tokens * 4]
    toks = tokenizer.encode(text)
    toks = toks[:max_tokens]
    return tokenizer.decode(toks)

def get_tiktoken():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

# --------- E5 指令前缀工具（若你把 E5 集成进 GuidelineFAISS 已经处理了，这里也兼容“上层补救”） ---------
def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in (model_name or "").lower()

def prep_query(q: str, model_name: str) -> str:
    return f"query: {q}" if needs_e5_prefix(model_name) else q

# --------- 统一检索器 ---------
class GuidelineRetriever:
    """
    统一检索器：
      - dense：直接用 GuidelineFAISS.search()
      - hybrid：BM25 + dense → RRF 融合
      - rerank：对 dense/hybrid 候选用 CrossEncoder 重排
    还提供：
      - MMR 去冗余挑选
      - 构建 RAG-ready 片段（带 token 截断、元数据）
    """
    def __init__(self, db, bm25_on: bool = True, rrf_k: int = 60, cross_encoder_name: Optional[str] = None):
        """
        db: 你已有的 GuidelineFAISS 实例（已加载索引）
        bm25_on: 是否启用 Hybrid 的稀疏通道
        rrf_k: RRF 融合常数
        cross_encoder_name: 若提供则支持 rerank（如 "cross-encoder/ms-marco-MiniLM-L-6-v2"）
        """
        self.db = db
        self.rrf_k = rrf_k
        self.model_name = getattr(db, "model_name", "")  # 用于识别 e5
        self.tokenizer = get_tiktoken()

        # 准备 BM25
        self.use_bm25 = bm25_on
        if bm25_on:
            # 可以把 title/keywords 拼进去增强稀疏召回
            contents = (db.meta["title"].astype(str) + " " +
                        db.meta["keywords"].astype(str) + " " +
                        db.meta["content"].astype(str)).tolist()
            corpus = [normalize_text(c) for c in contents]
            self.tokenized = [simple_tokenize(c) for c in corpus]
            self.bm25 = BM25Okapi(self.tokenized)

        # 准备 Cross-Encoder（可选）
        self.cross_encoder = None
        if cross_encoder_name:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(cross_encoder_name)

    # ---------- 基础检索 ----------
    def dense(self, query: str, k: int = 200) -> List[Dict]:
        q = prep_query(query, self.model_name)
        hits = self.db.search(q, top_k=k)
        # 确保有 index_id（GuidelineFAISS.search 内可返回），否则根据位置推断
        for i, h in enumerate(hits):
            if "index_id" not in h:
                # 找到 meta 中对应行（保守：用 chunk_id 定位）
                try:
                    idx = int(h.get("index_id"))
                except Exception:
                    # 兜底：用chunk_id定位
                    cid = h.get("chunk_id")
                    if cid in set(self.db.meta["chunk_id"]):
                        idx = int(self.db.meta.index[self.db.meta["chunk_id"] == cid][0])
                    else:
                        idx = i
                h["index_id"] = idx
        return hits

    def hybrid(self, query: str, top_k: int = 10, dense_k: int = 300, bm25_k: int = 500) -> List[Dict]:
        # 稠密
        dense_hits = self.dense(query, k=dense_k)

        # 稀疏
        if self.use_bm25:
            qn = normalize_text(query)
            qtok = simple_tokenize(qn)
            scores = self.bm25.get_scores(qtok)  # ndarray
            bm25_idx = scores.argsort()[-bm25_k:][::-1]
            bm25_hits = [{"index_id": int(i), "bm25_score": float(scores[i])} for i in bm25_idx]
        else:
            bm25_hits = []

        # RRF 融合
        rrf = {}
        for rank, h in enumerate(dense_hits, start=1):
            idx = h["index_id"]
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, h in enumerate(bm25_hits, start=1):
            idx = h["index_id"]
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for idx, s in fused:
            row = self.db.meta.iloc[idx]
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

    # ---------- 重排 ----------
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not self.cross_encoder:
            # 未配置重排模型，则原样返回Top-K（按 fused_score/score 排序）
            key = lambda x: x.get("fused_score", x.get("score", 0))
            return sorted(candidates, key=key, reverse=True)[:top_k]

        pairs = [(query, c["content"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs, convert_to_numpy=True)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)[:top_k]
        out = []
        for c, sc in ranked:
            cc = dict(c)
            cc["rerank_score"] = float(sc)
            out.append(cc)
        return out

    # ---------- 高层接口 ----------
    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",   # "dense" | "hybrid"
        top_k: int = 10,
        dense_k: int = 300,
        bm25_k: int = 500,
        do_rerank: bool = True,
        final_k: int = 10,
        mmr: bool = True
    ) -> List[Dict]:
        """
        统一入口：
          - mode="dense"  或 "hybrid"
          - do_rerank=True 时，会在候选上做 Cross-Encoder 重排（如果已配置）
          - mmr=True 时做去冗余（文档级）
        """
        if mode == "dense":
            candidates = self.dense(query, k=max(dense_k, top_k))
        else:
            candidates = self.hybrid(query, top_k=max(top_k, final_k, 50), dense_k=dense_k, bm25_k=bm25_k)

        if do_rerank:
            candidates = self.rerank(query, candidates, top_k=max(final_k, top_k))
        else:
            # 统一字段：为下游排序提供依据
            for c in candidates:
                if "rerank_score" not in c:
                    c["rerank_score"] = c.get("fused_score", c.get("score", 0))

        # 去冗余 + 裁剪
        if mmr:
            candidates = mmr_select(candidates, top_k=final_k)
        else:
            candidates = sorted(candidates, key=lambda x: x.get("rerank_score", x.get("fused_score", x.get("score", 0))), reverse=True)[:final_k]

        return candidates

    def build_rag_payload(
        self,
        query: str,
        mode: str = "hybrid",
        final_k: int = 6,
        max_tokens_per_chunk: int = 320,
        **kwargs
    ) -> Dict:
        """
        直接返回 RAG-ready 的 payload：
          - items: 每段带 guideline_id/chunk_id/title/keywords/content(已截断)
          - context_text: 已拼接文本（可直接放到 LLM 上下文）
          - sources: 元信息（便于答案里引用）
        """
        items = self.retrieve(query, mode=mode, final_k=final_k, **kwargs)
        toks = self.tokenizer
        rag_items = []
        for r in items:
            truncated = truncate_for_rag(r["content"], max_tokens_per_chunk, tokenizer=toks)
            rag_items.append({
                "guideline_id": r["guideline_id"],
                "chunk_id": r["chunk_id"],
                "title": r["title"],
                "keywords": r["keywords"],
                "content": truncated,
                "score": r.get("rerank_score", r.get("fused_score", r.get("score", 0)))
            })
        blocks = [f"[{it['guideline_id']} | {it['chunk_id']} | {it['title']}]\n{it['content']}" for it in rag_items]
        context_text = "\n\n---\n\n".join(blocks)
        sources = [{k: it[k] for k in ("guideline_id","chunk_id","title","keywords")} for it in rag_items]
        return {"items": rag_items, "context_text": context_text, "sources": sources}
