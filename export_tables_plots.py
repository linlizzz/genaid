#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

METRICS_DEFAULT = ["MRR", "nDCG", "Recall", "Precision", "mAP"]

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc == "ndcg": ren[c] = "nDCG"
        elif lc == "mrr": ren[c] = "MRR"
        elif lc == "recall": ren[c] = "Recall"
        elif lc == "precision": ren[c] = "Precision"
        elif lc == "input_mode": ren[c] = "input_mode"
        elif lc == "embedding": ren[c] = "embedding"
        elif lc == "k": ren[c] = "K"
        elif lc == "n": ren[c] = "N"
        elif lc == "map": ren[c] = "mAP"
    if ren:
        df = df.rename(columns=ren)
    return df

def filter_df(df: pd.DataFrame, embeddings, modes):
    if embeddings:
        df = df[df["embedding"].isin(embeddings)]
    if modes:
        df = df[df["input_mode"].isin(modes)]
    return df

def fmt_val(x, decimals=3):
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return str(x)


# ========== Table ==========

def bold_rowwise_max(pv: pd.DataFrame, decimals=3, skip=("embedding",)) -> pd.DataFrame:
    out = pv.copy()
    value_cols = [c for c in pv.columns if c not in skip]
    for i, row in pv.iterrows():
        vals = []
        for c in value_cols:
            try:
                vals.append(c, float(row[c]))
            except Exception:
                pass
        if not vals:
            continue
        max_c, _ = max(vals, key=lambda x: x[1])
        for c in value_cols:
            val_str = fmt_val(row[c], decimals=decimals)
            if c == max_c:
                out.at[i, c] = f"\\textbf{{{val_str}}}"
            else:
                out.at[i, c] = val_str
    return out

def to_latex(df: pd.DataFrame, caption: str, label: str, outpath: str, index=False):
    tex = df.to_latex(index=index, escape=False, column_format=None, longtable=False, 
                        bold_rows=False, na_rep="", caption=caption, label=label, 
                        multicolumn=True, multicolumn_format="c", header=True)
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[saved] {outpath}")



# ========== Plot ==========

def plot_metric_by_mode(df: pd.DataFrame, metric: str, outdir: str):
    # index: embedding, columns: input_mode
    pivot = df.pivot_table(index="embedding", columns="input_mode", values=metric, aggfunc="mean")
    pivot = pivot.sort_index()

    # 画柱状图（一个指标一张图；不指定颜色和样式）
    ax = pivot.plot(kind="bar")
    ax.set_title(f"{metric} by input mode")
    ax.set_xlabel("Embedding")
    ax.set_ylabel(metric)
    ax.legend(title="Input mode")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{metric}_by_mode.png"), dpi=200)
    plt.close()

def plot_overview(df: pd.DataFrame, metrics, outdir: str):
    # 各 embedding 在所有 input_mode 上的均值（每个指标一张图）
    g = df.groupby("embedding")[metrics].mean().sort_index()
    for m in metrics:
        ax = g[m].plot(kind="bar")
        ax.set_title(f"{m} (averaged over input modes)")
        ax.set_xlabel("Embedding")
        ax.set_ylabel(m)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{m}_overview.png"), dpi=200)
        plt.close()

# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="exp_summary.csv", help="实验汇总 CSV")
    ap.add_argument("--outdir_tables", default="results/tables/", help="LaTeX 输出目录")
    ap.add_argument("--outdir_plots", default="results/plots/", help="图片输出目录")
    ap.add_argument("--metrics", default=",".join(METRICS_DEFAULT))
    ap.add_argument("--embeddings", default="", help="只保留这些 embedding(逗号分隔)")
    ap.add_argument("--modes", default="", help="只保留这些 input_mode(逗号分隔)")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--bold_best", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.outdir_tables)
    ensure_dir(args.outdir_plots)

    df = pd.read_csv(args.summary_csv)
    df = normalize_columns(df)

    need = {"embedding", "input_mode", "MRR", "nDCG", "Recall", "Precision", "mAP"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    embeddings = [s.strip() for s in args.embeddings.split(",") if s.strip()]
    modes = [s.strip() for s in args.modes.split(",") if s.strip()]
    df = filter_df(df, embeddings, modes)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

'''
    # ========== Table ==========

    # 按 embedding 排序，input_mode 列顺序固定：raw,summary,keywords,combo（若存在）
    mode_order = ["raw", "summary", "keywords", "combo"]
    for metric in metrics:
        pv = df.pivot_table(index=["embedding"], columns="input_mode", values=metric, aggfunc="mean")
        # 统一列顺序
        cols = [c for c in mode_order if c in pv.columns] + [c for c in pv.columns if c not in mode_order]
        pv = pv[cols].reset_index()
        pv = pv.sort_values("embedding").reset_index(drop=True)

        pv_fmt = bold_rowwise_max(pv, decimals=args.decimals, skip=("embedding",)) if args.bold_best else pv.copy()
        if not args.bold_best:
            for c in pv_fmt.columns:
                if c != "embedding":
                    pv_fmt[c] = pv_fmt[c].map(lambda x: fmt_val(x, args.decimals))

        k_val = int(df["K"].iloc[0]) if "K" in df.columns and pd.notna(df["K"].iloc[0]) else 10
        caption = f"{metric} by input mode (K={k_val}). Higher is better."
        label = f"tab:{metric.lower()}-by-mode"
        outpath = os.path.join(args.outdir_tables, f"{metric}_by_mode.tex")
        to_latex(pv_fmt, caption, label, outpath, index=False)
'''
    #  ========== 总览表 ==========
    # 已有的：metrics = ["MRR","nDCG","Recall","Precision", "mAP"]
    def _format_metrics_table(df_in, metrics, decimals):
        df = df_in.copy()
        for m in metrics:
            df[m] = df[m].map(lambda x: fmt_val(x, decimals))
        return df
    # 总览表1：按 embedding (chunk / concat / fused）行，列为各指标在所有 input_mode 上的均值
    ov_emb = df.groupby("embedding")[metrics].mean().reset_index()

    # （可选）控制显示顺序
    embedding_order = ["chunk", "concat", "fused"]
    if set(embedding_order) & set(ov_emb["embedding"].unique()):
        ov_emb["embedding"] = pd.Categorical(ov_emb["embedding"], categories=embedding_order, ordered=True)
        ov_emb = ov_emb.sort_values("embedding")
    
    ov_emb_fmt = _format_metrics_table(ov_emb, metrics, args.decimals)
    to_latex(
        ov_emb_fmt,
        caption="Overall retrieval metrics averaged across input modes (rows = embedding).",
        label="tab:overview-by-embedding",
        outpath=os.path.join(args.outdir_tables, "overview_by_embedding.tex"),
        index=False
    )
    # 总览表2：按 input_mode (raw / summary / keywords / combo) 行，列为各指标在所有 embedding 上的均值
    ov_mode = df.groupby("input_mode")[metrics].mean().reset_index()

    # 行顺序控制
    mode_order = ["raw", "summary", "keywords", "combo"]
    if set(mode_order) & set(ov_mode["input_mode"].unique()):
        ov_mode["input_mode"] = pd.Categorical(ov_mode["input_mode"], categories=mode_order, ordered=True)
        ov_mode = ov_mode.sort_values("input_mode")

    ov_mode_fmt = _format_metrics_table(ov_mode, metrics, args.decimals)
    to_latex(
        ov_mode_fmt,
        caption="Overall retrieval metrics averaged across embeddings (rows = input mode).",
        label="tab:overview-by-input-mode",
        outpath=os.path.join(args.outdir_tables, "overview_by_input_mode.tex"),
        index=False
    )
    
    print(f"[saved] tables -> {args.outdir_tables}")
'''
    # ========== Plot ==========
    # 分指标作图
    for m in metrics:
        plot_metric_by_mode(df, m, args.outdir_plots)

    # 总览（每个指标一张）
    plot_overview(df, metrics, args.outdir_plots)

    print(f"[saved] figures -> {args.outdir_plots}")
'''
if __name__ == "__main__":
    main()
