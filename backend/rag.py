"""PandaRAG engine: domain-agnostic retrieval-augmented generation.

Pipeline (each stage toggleable, each traced for the UI):
  contextual chunking -> query planning (decomposition + HyDE) -> hybrid retrieval
  (BM25 + dense, Reciprocal Rank Fusion) -> GraphRAG entity expansion -> MMR rerank
  -> corrective grading (CRAG) -> grounded generation with citations
  -> self-reflection metrics (faithfulness / relevance / context precision).
"""
import functools
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import llm

STORE = Path(os.getenv("PANDARAG_DATA", Path(__file__).parent.parent / "data")) / "store.json"
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that these
those it its into than then there their they them we our you your he she his her not no can could would should may might
will shall do does did have has had which who whom what when where why how all any each more most other some such only own
same so too very also about over under between through during before after above below up down out off again further once
here both few nor s t just don now via using use used based within without per among across whereas however thus""".split())
# Everyday words that share an ending with drugs, enzymes and compounds (provide/peptide, increase/kinase).
# shortcut: a hand-kept list, so rarer words still slip through; upgrade to a real NER model if that matters.
NOT_ENTITY = set("""provide provided divide decide inside outside beside aside alongside worldwide nationwide guide override
coincide reside collide subside preside increase decrease release disease purchase database phrase please lease crease
showcase phase erase chase bookcase uppercase lowercase staircase suitcase briefcase
define refine redefine combine examine determine undermine medicine machine baseline online offline engine imagine decline
pipeline guideline routine outline discipline murine marine timeline deadline headline magazine vaccine doctrine feminine
masculine genuine confine intertwine incline underline streamline airline frontline cuisine canine
equine bovine porcine intestine uterine
remain domain certain obtain contain explain maintain retain sustain attain pertain constrain restrain terrain mountain
fountain captain bargain curtain britain uncertain entertain complain refrain abstain ascertain
begin margin origin within herein therein wherein thin skin basin cabin robin virgin ruin admin plugin login builtin
someone anyone everyone alone standalone milestone undergone postpone telephone cornerstone backbone headphone smartphone microphone
gone done none zone
control protocol school symbol patrol petrol preschool carpool enrol
describe prescribe subscribe inscribe transcribe ascribe""".split())
ROMAN = re.compile(r"[IVX]+$")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")


def tokens(text):
    return [t for t in (w.lower() for w in TOKEN.findall(text)) if t not in STOP and len(t) > 1]


def sentences(text):
    return [s.strip() for s in SENT.split(re.sub(r"\s+", " ", text)) if s.strip()]


def entities(text):
    """Cheap entity spotter: capitalised phrases, acronyms, gene-like and alphanumeric identifiers."""
    found = re.findall(r"\b(?:[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)+|[A-Z]{2,}[0-9A-Z]*(?:-[0-9A-Z]+)*|[A-Za-z]+[0-9]+[A-Za-z0-9]*)\b", text)
    found += [w for w in re.findall(r"\b[a-z]{6,}\b", text) if w.endswith(("ib", "mab", "in", "ol", "ase", "one", "ide", "ine"))]
    return {e for e in (f.strip() for f in found)
            if len(e) > 2 and e.lower() not in STOP and e not in NOT_ENTITY and not ROMAN.match(e)}


def chunk_text(text, size=110, overlap=25):
    """Sentence-aligned sliding window (words)."""
    sents, chunks, cur = sentences(text), [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= size:
            chunks.append(" ".join(cur))
            keep, n = [], 0
            for x in reversed(cur):
                n += len(x.split())
                keep.insert(0, x)
                if n >= overlap:
                    break
            cur = keep if len(keep) < len(cur) else []
    if cur and (not chunks or " ".join(cur) not in chunks[-1]):
        chunks.append(" ".join(cur))
    return chunks or ([text.strip()] if text.strip() else [])


class BM25:
    """Okapi BM25 held as a term x chunk sparse matrix, so a query is one sparse row sum
    instead of a Python loop over every chunk."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        lens = np.array([len(d) for d in docs], float)
        self.avg = float(lens.mean()) if self.n else 1.0
        self.vocab = {}
        rows, cols, tfs = [], [], []
        for i, d in enumerate(docs):
            for term, f in Counter(d).items():
                rows.append(self.vocab.setdefault(term, len(self.vocab)))
                cols.append(i)
                tfs.append(f)
        tf = np.array(tfs, float)
        if self.n:
            norm = self.k1 * (1 - self.b + self.b * lens[np.array(cols, int)] / self.avg)
            weight = tf * (self.k1 + 1) / (tf + norm)
            df = np.bincount(np.array(rows, int), minlength=len(self.vocab))
            idf = np.log(1 + (self.n - df + 0.5) / (df + 0.5))
            data = weight * idf[np.array(rows, int)]
        else:
            data = tf
        self.matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(self.vocab), max(self.n, 1)))

    def scores(self, q):
        if not self.n:
            return np.zeros(0)
        qtf = Counter(t for t in q if t in self.vocab)
        if not qtf:
            return np.zeros(self.n)
        idx = [self.vocab[t] for t in qtf]
        # repeated query terms count once per occurrence, as a plain term-by-term sum does
        reps = np.array([qtf[t] for t in qtf], float)[:, None]
        return np.asarray(self.matrix[idx].multiply(reps).sum(axis=0)).ravel()


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "


@functools.lru_cache(maxsize=1)
def neural_model():
    """(name, model), or (None, None) for LSA. PANDARAG_EMBED_MODEL picks a model (downloading it if
    needed) and "lsa" turns neural off. Unset, the default is used only if already on disk, so the
    app never downloads anything by itself."""
    name = os.getenv("PANDARAG_EMBED_MODEL")
    if name == "lsa":
        return None, None
    try:
        from sentence_transformers import SentenceTransformer
        return name or DEFAULT_MODEL, SentenceTransformer(
            name or DEFAULT_MODEL, local_files_only=not name,
            device="cpu")  # Apple's GPU backend ran out of memory indexing 12k chunks, and is not thread-safe
    except (ImportError, OSError):
        if name:
            raise
        return None, None


class Embedder:
    """Dense vectors: a sentence-transformers model if available, else local LSA (TF-IDF + SVD)."""

    def __init__(self, cache=None):
        self.name, self.st = neural_model()
        self.cache = {} if cache is None else cache   # text -> vector, so a rebuild only encodes new chunks

    def fit(self, texts):
        if self.st:
            return
        self.tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
        X = self.tfidf.fit_transform(texts)
        k = max(2, min(256, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0) if X.shape[0] > 2 else None
        if self.svd:
            self.svd.fit(X)

    def encode(self, texts, query=False):
        if self.st:
            if query:
                prefix = BGE_QUERY if "bge-" in self.name.lower() and "-en" in self.name.lower() else ""
                return np.asarray(self.st.encode([prefix + t for t in texts], normalize_embeddings=True))
            new = [t for t in dict.fromkeys(texts) if t not in self.cache]
            if new:
                self.cache.update(zip(new, np.asarray(self.st.encode(new, normalize_embeddings=True, batch_size=64))))
            keep = set(texts)
            for t in [t for t in self.cache if t not in keep]:
                del self.cache[t]
            return np.stack([self.cache[t] for t in texts])
        X = self.tfidf.transform(texts)
        v = self.svd.transform(X) if self.svd else X.toarray()
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    @property
    def label(self):
        return self.name or "LSA-256 (local)"


# Weights of each list in the fusion (BM25 is 1), tuned on SciFact with benchmark.py.
GRAPH_WEIGHT, NEIGHBOUR_WEIGHT = 0.25, 0.3
MMR_NEURAL, MMR_LSA = 0.72, 0.95   # relevance against diversity
LSA_WEIGHT = 0.1   # LSA ranks worse than BM25, so it should not get an equal vote
# Off-topic questions: how similar the model calls unrelated text, which CRAG treats as zero.
PROBES = ["How do I reset my router password?", "Best pizza dough recipe", "Who won the football match last night?",
          "Cheap flights to Lisbon in May", "How to tune a guitar", "Replace a bicycle tyre",
          "What time does the library open?", "Plot of a romantic comedy film"]


def rrf(rankings, weights=None, k=60):
    fused = defaultdict(float)
    for r, w in zip(rankings, weights or [1] * len(rankings)):
        for rank, idx in enumerate(r):
            fused[idx] += w / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True), fused


class Workspace:
    def __init__(self, name, domain="General", wid=None):
        self.id = wid or uuid.uuid4().hex[:8]
        self.name, self.domain = name, domain
        self.docs = {}      # doc_id -> {title, source, text, added}
        self.chunks = []    # {id, doc, text, ctx}
        self._dirty = True
        # Sync endpoints run in a threadpool, so two readers can race into build().
        self._lock = threading.Lock()
        self._vec_cache = {}

    # ---------- ingestion ----------
    def add(self, title, text, source="manual"):
        did = uuid.uuid4().hex[:10]
        self.docs[did] = {"id": did, "title": title.strip() or "Untitled", "source": source,
                          "text": text, "added": time.strftime("%Y-%m-%d %H:%M")}
        lead = (sentences(text) or [""])[0][:200]
        for i, c in enumerate(chunk_text(text)):
            # Contextual retrieval: index each chunk together with document-level context.
            self.chunks.append({"id": f"{did}:{i}", "doc": did, "text": c, "ctx": f"{title}. {lead}"})
        self._dirty = True
        return self.docs[did]

    def remove(self, did):
        self.docs.pop(did, None)
        self.chunks = [c for c in self.chunks if c["doc"] != did]
        self._dirty = True

    # ---------- indexing ----------
    def build(self):
        if not self._dirty:
            return
        with self._lock:
            if not self._dirty:   # another thread rebuilt while this one waited
                return
            texts = [f"{c['ctx']} {c['text']}" for c in self.chunks]
            bm25 = BM25([tokens(t) for t in texts])
            embedder = Embedder(self._vec_cache)
            vecs = np.zeros((0, 1))
            floor = 0.0
            if texts:
                embedder.fit(texts)
                vecs = embedder.encode(texts)
                if embedder.st:
                    floor = float(np.median(embedder.encode(PROBES, query=True) @ vecs.T))
            chunk_ents = [entities(c["text"]) for c in self.chunks]
            g = nx.Graph()
            for i, ents in enumerate(chunk_ents):
                ents = sorted(ents)
                for e in ents:
                    g.add_node(e, chunks=g.nodes[e]["chunks"] | {i} if e in g else {i})
                for a in range(len(ents)):
                    for b in range(a + 1, len(ents)):
                        w = g.get_edge_data(ents[a], ents[b], {"weight": 0})["weight"]
                        g.add_edge(ents[a], ents[b], weight=w + 1)
            # Publish together, so a concurrent reader never sees a half-swapped index.
            self.bm25, self.embedder, self.vecs, self.chunk_ents, self.graph = bm25, embedder, vecs, chunk_ents, g
            self.floor = floor
            self._dirty = False

    def graph_view(self, limit=70):
        self.build()
        g = self.graph
        if not g.number_of_nodes():
            return {"nodes": [], "edges": [], "communities": []}
        top = sorted(g.nodes, key=lambda n: (len(g.nodes[n]["chunks"]), g.degree(n)), reverse=True)[:limit]
        sub = g.subgraph(top)
        comms = list(nx.algorithms.community.greedy_modularity_communities(sub, weight="weight")) if sub.number_of_edges() else [set(top)]
        cid = {n: i for i, c in enumerate(comms) for n in c}
        return {
            "nodes": [{"id": n, "size": len(g.nodes[n]["chunks"]), "community": cid.get(n, 0)} for n in top],
            "edges": [{"source": a, "target": b, "weight": d["weight"]} for a, b, d in sub.edges(data=True)],
            "communities": [{"id": i, "members": sorted(c, key=lambda n: -g.degree(n))[:8]} for i, c in enumerate(comms)],
        }

    # ---------- retrieval ----------
    def plan(self, q, opts, trace):
        t = time.time()
        queries = [q]
        if opts.get("decompose"):
            parts = [p.strip(" ?") for p in re.split(r"\?|;|\band\b|\bvs\.?\b|\bversus\b|\bcompared to\b", q) if len(p.strip().split()) >= 2]
            if llm.available():
                try:
                    out = llm.complete(f"Split this question into at most 3 standalone search queries, one per line, no numbering:\n{q}", max_tokens=200)
                    parts = [l.strip("-• ").strip() for l in out.splitlines() if l.strip()][:3]
                except Exception:
                    pass
            queries += [p for p in parts if p.lower() != q.lower()][:3]
        hyde = None
        if opts.get("hyde") and llm.available():
            try:
                hyde = llm.complete(f"Write a short factual passage (3 sentences) that would answer: {q}", max_tokens=200)
            except Exception:
                hyde = None
        trace.append({"stage": "Query planning", "ms": int((time.time() - t) * 1000),
                      "detail": {"queries": queries, "hyde": bool(hyde)}})
        return queries, hyde

    def retrieve(self, q, opts=None, k=6):
        opts = {"hybrid": True, "graph": True, "rerank": True, "decompose": True, "hyde": True, "crag": True, **(opts or {})}
        trace = []
        self.build()
        if not self.chunks:
            return [], trace, {}
        queries, hyde = self.plan(q, opts, trace)

        t = time.time()
        rankings, weights, pool_n = [], [], 40
        dense_w = 1 if self.embedder.st else LSA_WEIGHT
        dense_texts = queries + ([hyde] if hyde else [])
        qv = self.embedder.encode(dense_texts, query=True)
        for i, qq in enumerate(queries):
            if opts["hybrid"]:
                rankings.append(list(np.argsort(-self.bm25.scores(tokens(qq)))[:pool_n])); weights.append(1)
            rankings.append(list(np.argsort(-(self.vecs @ qv[i]))[:pool_n])); weights.append(dense_w)
        if hyde:
            rankings.append(list(np.argsort(-(self.vecs @ qv[-1]))[:pool_n])); weights.append(dense_w)
        order, fused = rrf(rankings, weights)
        trace.append({"stage": "Hybrid retrieval" if opts["hybrid"] else "Dense retrieval", "ms": int((time.time() - t) * 1000),
                      "detail": {"lists_fused": len(rankings), "candidates": len(order), "embedder": self.embedder.label}})

        t = time.time()
        graph_hits = set()
        if opts["graph"] and self.graph.number_of_nodes():
            # Whole words only: a plain substring test finds MET in "method" and RAS in "KRAS".
            padded = f" {re.sub(r'[^a-z0-9]+', ' ', q.lower())} "
            names = ((e, re.sub(r"[^a-z0-9]+", " ", e.lower()).strip()) for e in self.graph.nodes)
            q_ents = {e for e, n in names if f" {n} " in padded or f" {n}s " in padded} | (entities(q) & set(self.graph.nodes))
            def idf(e):
                return math.log(1 + len(self.chunks) / len(self.graph.nodes[e]["chunks"]))

            nbrs = set()
            for e in q_ents:
                # rare co-occurring entities first; raw counts just return the corpus's hubs
                nbrs |= set(sorted(self.graph[e], key=lambda n: -self.graph[e][n]["weight"] * idf(n))[:4])
            nbrs -= q_ents
            gscore = defaultdict(float)
            for e in q_ents | nbrs:
                for ci in self.graph.nodes[e]["chunks"]:
                    gscore[ci] += idf(e) * (1 if e in q_ents else NEIGHBOUR_WEIGHT)
            # One more list in the fusion, so the graph can nudge the order but not outvote the searches.
            graph_hits = set(sorted(gscore, key=gscore.get, reverse=True)[:pool_n])
            for rank, ci in enumerate(sorted(graph_hits, key=gscore.get, reverse=True)):
                fused[ci] += GRAPH_WEIGHT / (60 + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)
            trace.append({"stage": "GraphRAG expansion", "ms": int((time.time() - t) * 1000),
                          "detail": {"seed_entities": sorted(q_ents)[:8], "neighbours": sorted(nbrs)[:8], "chunks_boosted": len(graph_hits)}})

        t = time.time()
        cand = order[:30]
        rel = self.vecs[cand] @ qv[0]
        if opts["rerank"]:
            # MMR: balance relevance to the query against redundancy with already-picked chunks.
            picked, lam = [], MMR_NEURAL if self.embedder.st else MMR_LSA
            fz = np.array([fused[c] for c in cand]); fz = fz / (fz.max() + 1e-9)
            mix = 0.5 * dense_w
            base = (1 - mix) * fz + mix * (rel - rel.min()) / (np.ptp(rel) + 1e-9)
            while len(picked) < min(k, len(cand)):
                best, bs = None, -1e9
                for j in range(len(cand)):
                    if j in picked:
                        continue
                    red = max((float(self.vecs[cand[j]] @ self.vecs[cand[p]]) for p in picked), default=0)
                    s = lam * base[j] - (1 - lam) * red
                    if s > bs:
                        best, bs = j, s
                picked.append(best)
            final = [(cand[j], float(base[j])) for j in picked]
        else:
            final = [(c, fused[c]) for c in cand[:k]]
        trace.append({"stage": "MMR rerank" if opts["rerank"] else "Top-k", "ms": int((time.time() - t) * 1000), "detail": {"kept": len(final)}})

        t = time.time()
        qtok = set(tokens(q))
        results = []
        for ci, score in final:
            c = self.chunks[ci]
            overlap = len(qtok & set(tokens(c["text"] + " " + c["ctx"]))) / (len(qtok) or 1)
            cos = (float(self.vecs[ci] @ qv[0]) - self.floor) / (1 - self.floor)
            grade = 0.55 * overlap + 0.45 * max(cos, 0)
            label = "relevant" if grade >= 0.35 else "ambiguous" if grade >= 0.15 else "irrelevant"
            results.append({"chunk": c["id"], "doc": c["doc"], "title": self.docs[c["doc"]]["title"],
                            "source": self.docs[c["doc"]]["source"], "text": c["text"], "score": round(score, 4),
                            "grade": round(grade, 3), "label": label, "via_graph": ci in graph_hits})
        crag = {}
        if opts["crag"]:
            kept = [r for r in results if r["label"] != "irrelevant"]
            crag = {"relevant": sum(r["label"] == "relevant" for r in results),
                    "ambiguous": sum(r["label"] == "ambiguous" for r in results),
                    "dropped": len(results) - len(kept),
                    "action": "answer" if any(r["label"] == "relevant" for r in results)
                    else "augment" if kept else "knowledge_gap"}
            results = kept
            trace.append({"stage": "Corrective grading (CRAG)", "ms": int((time.time() - t) * 1000), "detail": crag})
        return results, trace, crag

    # ---------- generation ----------
    def answer(self, q, opts=None, k=6):
        t0 = time.time()
        hits, trace, crag = self.retrieve(q, opts, k)
        t = time.time()
        if not hits:
            ans = ("I couldn't find grounded evidence for that in this workspace. "
                   "Add documents in Knowledge, or pull papers from Literature, then ask again.")
            mode = "none"
        elif llm.available():
            ctx = "\n\n".join(f"[{i+1}] ({h['title']}) {h['text']}" for i, h in enumerate(hits))
            try:
                ans = llm.complete(
                    f"Context passages:\n{ctx}\n\nQuestion: {q}\n\nAnswer using only the passages. Cite every claim with [n]. "
                    "If the passages don't cover something, say so plainly.",
                    system=f"You are PandaRAG, a precise research assistant for the {self.domain} domain.")
                mode = f"generative · {llm.MODEL}"
            except Exception as e:
                ans, mode = self._extractive(q, hits), f"extractive (LLM error: {type(e).__name__})"
        else:
            ans, mode = self._extractive(q, hits), "extractive"
        trace.append({"stage": "Grounded generation", "ms": int((time.time() - t) * 1000), "detail": {"mode": mode}})
        t = time.time()
        metrics = self._reflect(q, ans, hits)
        trace.append({"stage": "Self-reflection", "ms": int((time.time() - t) * 1000), "detail": metrics})
        return {"answer": ans, "sources": hits, "trace": trace, "metrics": metrics, "crag": crag,
                "mode": mode, "total_ms": int((time.time() - t0) * 1000)}

    def _extractive(self, q, hits, n=4):
        qtok = set(tokens(q))
        scored = []
        for i, h in enumerate(hits):
            for s in sentences(h["text"]):
                st = set(tokens(s))
                if len(st) < 4:
                    continue
                scored.append((len(qtok & st) / (len(qtok) or 1) + 0.15 * h["grade"] - 0.01 * i, s, i + 1))
        seen, out = set(), []
        for _, s, ref in sorted(scored, reverse=True):
            if s not in seen:
                seen.add(s)
                out.append(f"{s} [{ref}]")
            if len(out) == n:
                break
        return " ".join(out)

    def _reflect(self, q, ans, hits):
        if not hits:
            return {"faithfulness": 0, "answer_relevance": 0, "context_precision": 0}
        support = [set(tokens(h["text"])) for h in hits]
        sents_ = [s for s in sentences(ans) if len(tokens(s)) >= 3]
        ok = 0
        for s in sents_:
            refs = [int(r) - 1 for r in re.findall(r"\[(\d+)\]", s) if 0 < int(r) <= len(hits)]
            st = set(tokens(re.sub(r"\[\d+\]", "", s)))
            pool = set().union(*(support[r] for r in refs)) if refs else set().union(*support)
            ok += (len(st & pool) / (len(st) or 1)) >= 0.6
        qtok = set(tokens(q))
        return {
            "faithfulness": round(ok / (len(sents_) or 1), 2),
            "answer_relevance": round(len(qtok & set(tokens(ans))) / (len(qtok) or 1), 2),
            "context_precision": round(sum(h["label"] == "relevant" for h in hits) / len(hits), 2),
        }

    # ---------- persistence ----------
    def to_json(self):
        return {"id": self.id, "name": self.name, "domain": self.domain, "docs": list(self.docs.values())}

    def summary(self):
        return {"id": self.id, "name": self.name, "domain": self.domain, "docs": len(self.docs), "chunks": len(self.chunks)}


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.ws = {}
        if STORE.exists():
            for w in json.loads(STORE.read_text()):
                ws = Workspace(w["name"], w["domain"], w["id"])
                for d in w["docs"]:
                    ws.add(d["title"], d["text"], d["source"])
                self.ws[ws.id] = ws

    def save(self):
        STORE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STORE.with_suffix(".tmp")
        tmp.write_text(json.dumps([w.to_json() for w in self.ws.values()]))
        tmp.replace(STORE)  # atomic write so a crash mid-save never corrupts the store
