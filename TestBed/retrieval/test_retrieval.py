from guideline_faiss import GuidelineFAISS
from guideline_retriever import GuidelineRetriever

# 1) 加载你之前保存好的索引
db = GuidelineFAISS.load("faiss_store")

# 2) 创建统一检索器
#   - 开启 Hybrid（默认 bm25_on=True）
#   - 配置 cross-encoder 做重排（可选）
retr = GuidelineRetriever(
    db,
    bm25_on=True,
    rrf_k=60,
    cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2"  # 如不想重排可设为 None
)

query = "Potilaalla on pitkittynyt yskä ja kuume, epäillään pneumoniaa."

# A) 直接拿融合+重排后的候选（已 MMR 去冗余）
hits = retr.retrieve(query, mode="hybrid", top_k=20, dense_k=300, bm25_k=600, do_rerank=True, final_k=8, mmr=True)
for h in hits:
    print(f"[{h.get('rerank_score', h.get('fused_score', h.get('score',0))):.4f}] {h['guideline_id']} | {h['chunk_id']} | {h['title']}")

# B) 构建 RAG payload（含截断后的 content + context_text + sources）
payload = retr.build_rag_payload(query, mode="hybrid", final_k=6, max_tokens_per_chunk=320)
print("\n--- Context ---\n")
print(payload["context_text"])
print("\n--- Sources ---\n", payload["sources"])

'''
调参建议(按优先级)

嵌入模型:
优先试 intfloat/multilingual-e5-base(注意 E5 前缀);若主要是芬兰语/多语场景,e5 通常更稳.

召回规模:
dense_k 与 bm25_k 取 300800 之间,视数据量而定;top_k/final_k 取 612 比较折中.

RRF 融合:
rrf_k=60 常见默认,可适度微调.

重排:
cross-encoder/ms-marco-MiniLM-L-6-v2 速度快,-L-12-v2 更准但更慢.

MMR 去冗余:
默认开启,以文档(guideline)为单位保证多样性,避免同一文档多个相似块占满配额.

RAG 截断:
max_tokens_per_chunk 300±50 常见;总上下文预算按你的 LLM 限制来.
'''