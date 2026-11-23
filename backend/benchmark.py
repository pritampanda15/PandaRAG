"""Retrieval benchmark on human-labelled BEIR test sets (downloads about 8 MB on first run).

    python3 benchmark.py                      # every configuration, default embedder
    PANDARAG_EMBED_MODEL=lsa python3 benchmark.py

SciFact is for tuning; NFCorpus is the check that a change was not just overfitted to SciFact.
"""
import sys
import time

import numpy as np

import rag

K = 10
OFF = {"hybrid": False, "graph": False, "rerank": False, "decompose": False, "hyde": False, "crag": False}
CONFIGS = [("dense only", {}), ("hybrid", {"hybrid": True}), ("hybrid + graph", {"hybrid": True, "graph": True}),
           ("hybrid + MMR", {"hybrid": True, "rerank": True}),
           ("hybrid + decompose", {"hybrid": True, "decompose": True}),
           ("all stages", {k: True for k in OFF})]


def load(name):
    import pandas as pd
    from huggingface_hub import hf_hub_download as get
    tbl = lambda kind: pd.read_parquet(get(f"BeIR/{name}", f"{kind}/{kind}-00000-of-00001.parquet", repo_type="dataset"))
    corpus, queries = tbl("corpus"), tbl("queries")
    qrels = pd.read_csv(get(f"BeIR/{name}-qrels", "test.tsv", repo_type="dataset"), sep="\t", dtype={0: str, 1: str})
    rel = {}
    for q, d, s in qrels.itertuples(index=False):
        if s > 0:
            rel.setdefault(q, {})[d] = int(s)
    text = dict(zip(queries["_id"].astype(str), queries["text"]))
    return corpus, {q: text[q] for q in rel}, rel


def ndcg(ranked, rel):
    gain = lambda s: 2 ** s - 1
    dcg = sum(gain(rel.get(d, 0)) / np.log2(i + 2) for i, d in enumerate(ranked[:K]))
    return dcg / sum(gain(s) / np.log2(i + 2) for i, s in enumerate(sorted(rel.values(), reverse=True)[:K]))


def score(runs, rel):
    """Mean nDCG@10, MRR@10, recall@10."""
    n = [ndcg(runs[q], rel[q]) for q in rel]
    m = [next((1 / (i + 1) for i, d in enumerate(runs[q][:K]) if d in rel[q]), 0) for q in rel]
    r = [len(set(runs[q][:K]) & set(rel[q])) / len(rel[q]) for q in rel]
    return np.mean(n), np.mean(m), np.mean(r)


def unique(docs):
    return list(dict.fromkeys(docs))


def main(names):
    for name in names:
        corpus, queries, rel = load(name)
        ws, ids = rag.Workspace(name), {}
        for i, title, text in corpus.itertuples(index=False):
            ids[ws.add(title or "", text)["id"]] = str(i)
        t = time.time()
        ws.build()
        print(f"\n{name}: {len(ws.docs)} docs, {len(ws.chunks)} chunks, {len(queries)} queries, "
              f"{ws.embedder.label}, indexed in {time.time() - t:.0f} s")
        print(f"  {'':20} nDCG@10   MRR@10   R@10   ms/query")
        rows = [("BM25 only", None)] + CONFIGS
        for label, opts in rows:
            t, runs = time.time(), {}
            for qid, q in queries.items():
                if opts is None:
                    hits = [ws.chunks[c]["doc"] for c in np.argsort(-ws.bm25.scores(rag.tokens(q)))[:30]]
                else:   # 30 chunks, so that 10 distinct documents are left
                    hits = [h["doc"] for h in ws.retrieve(q, {**OFF, **opts}, k=30)[0]]
                runs[qid] = [ids[d] for d in unique(hits)]
            ms = 1000 * (time.time() - t) / len(queries)
            n, m, r = score(runs, rel)
            print(f"  {label:20} {n:.3f}     {m:.3f}    {r:.3f}  {ms:6.0f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["scifact", "nfcorpus"])
