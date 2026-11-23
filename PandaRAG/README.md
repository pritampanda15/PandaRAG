# PandaRAG

One application over the six research repos in this monorepo: a domain-agnostic RAG
engine with a tab per capability, plus live wrappers around the repos' working models.

```bash
sh PandaRAG/run.sh      # http://localhost:8787
```

Requires Python 3.10+. Install with `pip install -r PandaRAG/requirements.txt`. That covers Ask,
Knowledge and Graph, plus Literature and Causal once their projects are cloned; the commented extras in that file enable the ML labs.

The "live" states below describe the author's machine. The ADMET project is this repo's root. The other labs read sibling projects, so clone
`Drug-Repurposing-Intelligence-System`, `Literature-Intelligence`, `OncoMolML`,
`Oncology-Drug-Response-Prediction-System_DL-RAG` and `TC-DrugRAG` (all under
`github.com/pritampanda15`) into the repo root, beside `PandaRAG/`. A missing project only turns
its own lab off. Datasets and trained checkpoints for Repurposing are not included,
and its R-GCN fixes and `train_ranking.py` are not yet published in that project.

Optional environment:

| Variable | Effect |
| --- | --- |
| `ANTHROPIC_API_KEY` | Generative, cited answers via Claude. Without it the engine answers extractively — everything else is unchanged. |
| `PANDARAG_MODEL` | Model id (default `claude-sonnet-5-5`). |
| `PANDARAG_EMBED_MODEL` | A sentence-transformers model name (e.g. `BAAI/bge-small-en-v1.5`). Default is a local TF-IDF + SVD (LSA) embedder — no download, no network, but bag-of-words: this is the biggest single lever on retrieval quality. |
| `PANDARAG_NCBI_EMAIL`, `NCBI_API_KEY` | PubMed etiquette and higher rate limits. |

## Tabs

- **Ask** — the pipeline, with every stage toggleable and traced.
- **Knowledge** — workspaces and documents; paste or upload any text, any domain.
- **Graph** — entity co-occurrence graph mined from your chunks, draggable, clustered.
- **Literature** — live PubMed / bioRxiv search, one click to import abstracts.
- **Molecules** — Caco-2 permeability from the trained GCN + RDKit descriptors.
- **Screening** — JAX Morgan fingerprints, pairwise Tanimoto, MaxMin diverse subset.
- **Causal** — build a mechanistic chain, score it with TC-DrugRAG's formula.
- **Repurposing** — rank diseases for a compound over Hetionet, with an honest model card.
- **Labs** — which repo modules are live on this machine, and the retrieval stack.

## Pipeline

Contextual chunking → query decomposition + HyDE → hybrid BM25 + dense retrieval fused
with reciprocal rank fusion → GraphRAG entity expansion → MMR rerank → corrective
grading (CRAG) → grounded generation with citations → self-reflection metrics
(faithfulness, answer relevance, context precision).

Nothing is domain-specific: the entity extractor, chunker and graph are text-only, so a
workspace of contracts, incident reports or lecture notes behaves the same as one of papers.

## Capacity

Measured on this machine (synthetic ~230-word abstracts, 2.5 chunks each, extractive mode):

| Documents | Chunks | Reindex | Warm query | Memory |
| --- | --- | --- | --- | --- |
| 500 | 1,272 | 0.5 s | 4 ms | 330 MB |
| 2,000 | 5,072 | 2.1 s | 9 ms | 513 MB |
| 10,000 | 25,252 | 14.7 s | 35 ms | 1.5 GB |

Real run: 100 PubMed abstracts fetched in 3.2 s, imported in 34 ms, first query 479 ms
(it pays for the reindex), every query after that 5 ms.

**Bulk import is the cheap path.** `POST /workspaces/{id}/import` adds every paper and the
index rebuilds once, on the next query. Adding documents one at a time pays that rebuild each
time, which is what hurts above a few thousand documents.

Comfortable today: a few thousand documents per workspace. Usable to ~10,000. The ceiling is
the full refit (TF-IDF + SVD over every chunk) and the in-memory store — past that, split
topics into separate workspaces, or make the index incremental and move the store to SQLite.

## Repo wiring

| Module | Repo | State on this machine |
| --- | --- | --- |
| ADMET GNN | `ADMET-Prediction-System-…-with-RAG` | live (trained checkpoint, epoch 69) |
| Screening | `OncoMolML` | live (JAX fingerprints + Tanimoto + MaxMin) |
| Literature | `Literature-Intelligence` | live (PubMed + bioRxiv clients) |
| Causal KG | `TC-DrugRAG` | live (schemas + path scorer) |
| Oncology KB | `Oncology-Drug-Response-…_DL-RAG` | live (seed corpus) |
| Repurposing | `Drug-Repurposing-Intelligence-System` | live (Hetionet 47,031 nodes · trained checkpoint) |
| Variant effects | `OncoMolML` | needs `tensorflow` |

Screening needs `jax` (`pip install "jax[cpu]"`). `pykeen` is in the repurposing repo's
pinned requirements but is imported nowhere in its source.

**Repurposing runs on the real R-GCN.** Hetionet is downloaded (714 MB JSON → 47,031 nodes,
2.25 M edges) and `DrugRepurposingModel` now encodes the whole graph: `flatten_hetero` maps the
heterogeneous graph onto one node index space with a relation id per edge (24 relations, plus
their inverses = 48), and `RGCNEncoder` runs basis-decomposition message passing over the
~4.5 M directed edges before the MLP decoder scores a compound-disease pair. Message passing
costs the same at any batch size, so training is full-batch: one encode per epoch.

Three upstream stubs had to go first — `RGCNEncoder.forward` dropped its convolution result on
the floor and passed `edge_type=0` for every edge, `DrugRepurposingModel` was marked *"Simplified
model for now - use embeddings"*, and `Trainer.evaluate` returned a hardcoded `auc: 0.0`.

**Results.** `train_ranking.py` draws negatives per drug and uses a listwise loss. All
`Compound-treats-Disease` edges are removed from the message-passing graph, in training and when
serving. Filtered ranking of 137 diseases on 114 held-out pairs (one seed, one random pair split):

| Scorer | Median rank | Hits@1 | Hits@10 | MRR |
| --- | --- | --- | --- | --- |
| **R-GCN, treats edges removed** (served) | **3.5** | **0.360** | **0.737** | **0.479** |
| Embeddings only, same loss | 7 | 0.149 | 0.623 | 0.301 |
| Popularity only (ignores the drug) | 13 | 0.096 | 0.447 | 0.220 |

The R-GCN was chosen over embeddings on validation MRR (0.390 vs 0.258). Drugs can appear in both
train and test, so this does not measure performance on compounds with no known indication.

**Leaked results, do not cite.** `ranking_model.pt` (MRR 0.674) and `rgcn_model.pt` (AUC 0.959 / AP
0.872) kept held-out treats edges in the message-passing graph. With val/test edges removed at
inference, `ranking_model.pt` falls to MRR 0.208. The app no longer loads either file.

To rebuild from scratch, inside `Drug-Repurposing-Intelligence-System`:

```bash
python3 scripts/download_data.py && python3 scripts/build_knowledge_graph.py
```

```bash
python3 scripts/train_ranking.py
```

Add `--no-graph --output models/checkpoints/ranking_nograph.pt` for the embedding baseline. Each run writes `<output>.metrics.json` beside
its checkpoint, which is what the model card reads.

The repos are loaded by file path, so nothing has to be installed as a package and the
heavy optional imports in their `__init__.py` files stay out of the way.

## Look

Warm paper, hairline rules, serif display (Newsreader) over IBM Plex Sans/Mono, ochre accent,
archival plate line-art at the margins. A sepia-dark variant lives behind the theme toggle.

## Layout

```
PandaRAG/
  backend/   app.py (API) · rag.py (engine) · labs.py (repo wrappers) · llm.py · test_rag.py
  frontend/  index.html · app.css · app.js      # no build step
  data/      store.json                          # workspaces, written atomically
```

`python3 backend/test_rag.py` checks retrieval, generation, the graph, the empty-workspace
path and document removal.
