"""PandaRAG engine: domain-agnostic retrieval-augmented generation with glass-box tracing.

Pipeline (each stage toggleable, each traced for the UI):
  contextual chunking -> conversation rewrite + query planning (parallel decomposition + HyDE)
  -> hybrid retrieval (filtered BM25 + dense, weighted RRF) -> GraphRAG entity & community expansion
  -> MMR rerank -> corrective grading (CRAG + backfill + optional PubMed fallback)
  -> grounded generation with citations -> semantic & polarity-aware self-reflection metrics.
"""
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import json
import math
import os
import re
import sqlite3
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

DATA_DIR = Path(os.getenv("PANDARAG_DATA", Path(__file__).parent.parent / "data"))
LEGACY_STORE = DATA_DIR / "store.json"
DB_PATH = DATA_DIR / "pandarag.db"

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that these
those it its into than then there their they them we our you your he she his her not no can could would should may might
will shall do does did have has had which who whom what when where why how all any each more most other some such only own
same so too very also about over under between through during before after above below up down out off again further once
here both few nor s t just don now via using use used based within without per among across whereas however thus""".split())

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "without", "none", "cannot", "can't",
    "don't", "doesn't", "didn't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "fail", "fails", "failed", "lack", "lacks", "lacking", "absent", "unlikely",
}

ABSTENTION_RE = re.compile(
    r"\b(?:do not cover|don't cover|does not cover|doesn't cover|not mentioned|not covered|"
    r"not stated|no grounded evidence|insufficient evidence|not provided|cannot determine|"
    r"not specified|passages do not|context does not)\b",
    re.I,
)

# Everyday words that share an ending with drugs, enzymes and compounds (provide/peptide, increase/kinase).
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

BIO_DOMAIN_RE = re.compile(
    r"\b(?:bio|onco|med|pharm|chem|drug|clin|genom|neuro|health|life|immun|cardio|patho|cancer|toxic|mol|therap)",
    re.I,
)
GLOBAL_QUERY_RE = re.compile(
    r"\b(?:summari[sz]e|overview|main themes|key takeaways|open questions|overall|across all|what topics|broadly)\b",
    re.I,
)
FOLLOWUP_RE = re.compile(
    r"\b(?:it|its|they|them|their|that|those|these|this|former|latter|above|previous|same|second|first|compare them|tell me more)\b",
    re.I,
)
ABBREV_RE = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Fig|Figs|Eq|Eqs|Ref|Refs|vs|St|No|Vol|pp|p|Inc|Ltd|approx|al|e\.g|i\.e|U\.S)\.",
    re.I,
)
ROMAN = re.compile(r"^[IVX]+$")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
NEG_TOKEN = re.compile(r"[A-Za-z0-9']+|[A-Za-z0-9][A-Za-z0-9\-]+")


def is_bio_domain(domain: str) -> bool:
    return bool(BIO_DOMAIN_RE.search(domain or ""))


def tokens(text: str) -> list[str]:
    return [t for t in (w.lower() for w in TOKEN.findall(text)) if t not in STOP and len(t) > 1]


def negations_in(text: str) -> set[str]:
    return {w.lower() for w in NEG_TOKEN.findall(text) if w.lower() in NEGATIONS}


def sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    # Protect common abbreviations from false sentence splits (e.g., "Fig. 1", "Dr. Wang", "et al.").
    protected = ABBREV_RE.sub(lambda m: m.group(0).replace(".", "\u0000"), cleaned)
    return [s.replace("\u0000", ".").strip() for s in SENT.split(protected) if s.strip()]


def entities(text: str, bio: bool = True) -> set[str]:
    """Entity spotter: capitalised phrases, acronyms, alphanumeric IDs; optional biomedical suffix matcher."""
    found = re.findall(
        r"\b(?:[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)+|[A-Z]{2,}[0-9A-Z]*(?:-[0-9A-Z]+)*|[A-Za-z]+[0-9]+[A-Za-z0-9]*)\b",
        text,
    )
    if bio:
        found += [
            w for w in re.findall(r"\b[a-z]{6,}\b", text)
            if w.endswith(("ib", "mab", "in", "ol", "ase", "one", "ide", "ine"))
        ]
    return {
        e for e in (f.strip() for f in found)
        if len(e) > 2 and e.lower() not in STOP and e not in NOT_ENTITY and not ROMAN.match(e)
    }


def _split_long_block(words: list[str], size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)]


def chunk_text(text: str, size: int = 110, overlap: int = 25) -> list[str]:
    """Sentence-aligned sliding window (words), with word-window fallback for huge unpunctuated blocks."""
    raw_sents = sentences(text)
    sents = []
    max_sent_words = int(size * 1.5)
    for s in raw_sents:
        w = s.split()
        if len(w) > max_sent_words:
            sents.extend(_split_long_block(w, size, overlap))
        else:
            sents.append(s)

    chunks, cur = [], []
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


def text_hash(text: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", text.strip()).encode("utf-8")).hexdigest()


class BM25:
    """Okapi BM25 held as a term x chunk sparse matrix, so a query is one sparse row sum."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        lens = np.array([len(d) for d in docs], float)
        self.avg = float(lens.mean()) if self.n and lens.sum() else 1.0
        self.vocab = {}
        rows, cols, tfs = [], [], []
        for i, d in enumerate(docs):
            for term, f in Counter(d).items():
                rows.append(self.vocab.setdefault(term, len(self.vocab)))
                cols.append(i)
                tfs.append(f)
        tf = np.array(tfs, float)
        if self.n and len(rows):
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
        reps = np.array([qtf[t] for t in qtf], float)[:, None]
        return np.asarray(self.matrix[idx].multiply(reps).sum(axis=0)).ravel()


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY = "Represent this sentence for searching relevant passages: "


@functools.lru_cache(maxsize=1)
def neural_model():
    """(name, model), or (None, None) for LSA. PANDARAG_EMBED_MODEL picks a model (downloading it if
    needed) and 'lsa' turns neural off. Unset, the default is used only if already on disk."""
    name = os.getenv("PANDARAG_EMBED_MODEL")
    if name == "lsa":
        return None, None
    try:
        from sentence_transformers import SentenceTransformer
        return name or DEFAULT_MODEL, SentenceTransformer(
            name or DEFAULT_MODEL, local_files_only=not name,
            device="cpu",  # Apple MPS OOMs on 12k chunks and is not thread-safe
        )
    except (ImportError, OSError):
        if name:
            raise
        return None, None


class DiskVecCache:
    """SQLite-backed vector cache so neural embeddings survive restarts instantly."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path
        self._lock = threading.Lock()
        if db_path:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(db_path) as con:
                con.execute(
                    "CREATE TABLE IF NOT EXISTS embeddings ("
                    "model TEXT NOT NULL, text_hash TEXT NOT NULL, vec BLOB NOT NULL, "
                    "PRIMARY KEY (model, text_hash))"
                )

    def get_many(self, model: str, hashes: list[str]) -> dict[str, np.ndarray]:
        if not self.db_path or not hashes:
            return {}
        out = {}
        with self._lock, sqlite3.connect(self.db_path) as con:
            for i in range(0, len(hashes), 500):
                batch = hashes[i:i + 500]
                q = ",".join("?" * len(batch))
                for h, blob in con.execute(
                    f"SELECT text_hash, vec FROM embeddings WHERE model=? AND text_hash IN ({q})",
                    [model, *batch],
                ):
                    out[h] = np.frombuffer(blob, dtype=np.float32).copy()
        return out

    def put_many(self, model: str, items: list[tuple[str, np.ndarray]]):
        if not self.db_path or not items:
            return
        with self._lock, sqlite3.connect(self.db_path) as con:
            con.executemany(
                "INSERT OR REPLACE INTO embeddings (model, text_hash, vec) VALUES (?, ?, ?)",
                [(model, h, np.asarray(v, dtype=np.float32).tobytes()) for h, v in items],
            )


class Embedder:
    """Dense vectors: a sentence-transformers model if available, else local LSA (TF-IDF + SVD)."""

    def __init__(self, cache=None, disk_cache: DiskVecCache | None = None):
        self.name, self.st = neural_model()
        self.cache = {} if cache is None else cache   # text -> vector (RAM cache)
        self.disk_cache = disk_cache

    def fit(self, texts):
        if self.st:
            return
        self.tfidf = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, stop_words="english")
        X = self.tfidf.fit_transform(texts)
        if X.shape[0] > 2 and X.shape[1] > 2:
            k = max(2, min(256, X.shape[0] - 1, X.shape[1] - 1))
            self.svd = TruncatedSVD(n_components=k, random_state=0)
            self.svd.fit(X)
        else:
            self.svd = None

    def encode(self, texts, query=False, use_cache=True):
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.st:
            if query:
                prefix = BGE_QUERY if "bge-" in self.name.lower() and "-en" in self.name.lower() else ""
                return np.asarray(self.st.encode([prefix + t for t in texts], normalize_embeddings=True), dtype=np.float32)
            if not use_cache:
                # Passage encoding without query prefix and without mutating/pruning the chunk cache (e.g., HyDE, reflection).
                return np.asarray(self.st.encode(list(texts), normalize_embeddings=True, batch_size=64), dtype=np.float32)

            uniq = list(dict.fromkeys(texts))
            missing = [t for t in uniq if t not in self.cache]
            if missing and self.disk_cache:
                hashes = {t: text_hash(t) for t in missing}
                found = self.disk_cache.get_many(self.name, list(hashes.values()))
                for t, h in hashes.items():
                    if h in found:
                        self.cache[t] = found[h]
                missing = [t for t in missing if t not in self.cache]

            if missing:
                vecs = np.asarray(self.st.encode(missing, normalize_embeddings=True, batch_size=64), dtype=np.float32)
                for t, v in zip(missing, vecs):
                    self.cache[t] = v
                if self.disk_cache:
                    self.disk_cache.put_many(self.name, [(text_hash(t), v) for t, v in zip(missing, vecs)])

            keep = set(texts)
            for t in [t for t in self.cache if t not in keep]:
                del self.cache[t]
            return np.stack([self.cache[t] for t in texts])

        X = self.tfidf.transform(texts)
        v = self.svd.transform(X) if self.svd else X.toarray()
        return (v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

    @property
    def label(self):
        return self.name or "LSA-256 (local)"


# Fusion & reranking constants (tuned on SciFact with benchmark.py).
GRAPH_WEIGHT, NEIGHBOUR_WEIGHT = 0.25, 0.3
COMMUNITY_WEIGHT = 0.2
SUBQUERY_WEIGHT = 0.45
MMR_NEURAL, MMR_LSA = 0.72, 0.95
LSA_WEIGHT = 0.1

# Off-topic probes used to calibrate BGE's compressed cosine floor.
PROBES = [
    "How do I reset my router password?", "Best pizza dough recipe", "Who won the football match last night?",
    "Cheap flights to Lisbon in May", "How to tune a guitar", "Replace a bicycle tyre",
    "What time does the library open?", "Plot of a romantic comedy film",
]


def rrf(rankings, weights=None, k=60):
    fused = defaultdict(float)
    for r, w in zip(rankings, weights or [1] * len(rankings)):
        for rank, idx in enumerate(r):
            fused[idx] += w / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True), fused


class Workspace:
    def __init__(self, name, domain="General", wid=None, disk_cache: DiskVecCache | None = None):
        self.id = wid or uuid.uuid4().hex[:8]
        self.name, self.domain = name, domain
        self.docs = {}      # doc_id -> {id, title, source, text, added, hash}
        self.chunks = []    # {id, doc, text, ctx}
        self._hashes = set()
        self._dirty = True
        self._lock = threading.RLock()
        self._vec_cache = {}
        self._disk_cache = disk_cache

    # ---------- ingestion ----------
    def add(self, title, text, source="manual", did=None, added=None, dedup=True):
        h = text_hash(text)
        with self._lock:
            if dedup and h in self._hashes:
                for existing in self.docs.values():
                    if existing.get("hash") == h:
                        return {**existing, "duplicate": True}
            doc_id = did or uuid.uuid4().hex[:10]
            doc = {
                "id": doc_id,
                "title": (title or "").strip() or "Untitled",
                "source": source,
                "text": text,
                "added": added or time.strftime("%Y-%m-%d %H:%M"),
                "hash": h,
            }
            self.docs[doc_id] = doc
            self._hashes.add(h)
            lead = (sentences(text) or [""])[0][:200]
            for i, c in enumerate(chunk_text(text)):
                self.chunks.append({"id": f"{doc_id}:{i}", "doc": doc_id, "text": c, "ctx": f"{doc['title']}. {lead}"})
            self._dirty = True
            return {**doc, "duplicate": False}

    def remove(self, did):
        with self._lock:
            removed = self.docs.pop(did, None)
            self.chunks = [c for c in self.chunks if c["doc"] != did]
            if removed:
                self._hashes = {d.get("hash") for d in self.docs.values() if d.get("hash")}
            self._dirty = True

    # ---------- indexing ----------
    def build(self):
        if not self._dirty:
            return
        with self._lock:
            if not self._dirty:
                return
            texts = [f"{c['ctx']} {c['text']}" for c in self.chunks]
            bm25 = BM25([tokens(t) for t in texts])
            embedder = Embedder(self._vec_cache, disk_cache=self._disk_cache)
            vecs = np.zeros((0, 1), dtype=np.float32)
            floor = 0.0
            if texts:
                embedder.fit(texts)
                vecs = embedder.encode(texts, query=False, use_cache=True)
                if embedder.st:
                    probe_sims = embedder.encode(PROBES, query=True, use_cache=False) @ vecs.T
                    # 25th percentile is safer than median if a workspace happens to overlap 1-2 probe topics
                    floor = float(np.percentile(probe_sims, 35))
            bio = is_bio_domain(self.domain)
            chunk_ents = [entities(c["text"], bio=bio) for c in self.chunks]
            g = nx.Graph()
            for i, ents in enumerate(chunk_ents):
                ents = sorted(ents)
                for e in ents:
                    g.add_node(e, chunks=g.nodes[e]["chunks"] | {i} if e in g else {i})
                for a in range(len(ents)):
                    for b in range(a + 1, len(ents)):
                        w = g.get_edge_data(ents[a], ents[b], {"weight": 0})["weight"]
                        g.add_edge(ents[a], ents[b], weight=w + 1)
            self.bm25, self.embedder, self.vecs, self.chunk_ents, self.graph = bm25, embedder, vecs, chunk_ents, g
            self.floor = floor
            self._dirty = False

    def graph_view(self, limit=70):
        with self._lock:
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

    # ---------- query planning ----------
    def _contextualize(self, q: str, history: list[dict] | None, trace: list) -> str:
        if not history:
            return q
        recent = [m for m in history[-6:] if m.get("role") in ("user", "assistant") and m.get("content")]
        if not recent:
            return q
        t = time.time()
        rewritten = q
        if llm.available():
            try:
                conv = "\n".join(f"{m['role'].title()}: {m['content'][:350]}" for m in recent)
                out = llm.complete(
                    f"Conversation so far:\n{conv}\n\nLatest question: {q}\n\n"
                    "Rewrite the latest question as one concise standalone search query resolving any pronouns or references. "
                    "Return only the rewritten question.",
                    max_tokens=256,
                )
                if out and len(out) < 400:
                    rewritten = out.splitlines()[0].strip(' "\'')
            except Exception:
                pass
        elif FOLLOWUP_RE.search(q):
            prev_users = [m["content"] for m in recent if m["role"] == "user"]
            if prev_users:
                prev_ents = sorted(entities(prev_users[-1], bio=is_bio_domain(self.domain)))
                anchor = " ".join(prev_ents[:4]) or " ".join(tokens(prev_users[-1])[:5])
                if anchor:
                    rewritten = f"{q} ({anchor})"
        if rewritten != q:
            trace.append({"stage": "Follow-up rewrite", "ms": int((time.time() - t) * 1000),
                          "detail": {"original": q, "rewritten": rewritten}})
        return rewritten

    def _regex_decompose(self, q: str) -> list[str]:
        parts = [p.strip(" ?") for p in re.split(r"\?|;|\bvs\.?\b|\bversus\b|\bcompared to\b", q) if len(p.strip().split()) >= 2]
        if len(parts) <= 1 and re.search(r"\band\b", q, re.I):
            cand = [p.strip(" ?") for p in re.split(r"\band\b", q, flags=re.I) if p.strip()]
            # Only split on "and" when both sides look like full clauses (>= 3 words), not noun pairs.
            if len(cand) >= 2 and all(len(c.split()) >= 3 for c in cand):
                parts = cand
        return parts

    def plan(self, q, opts, trace):
        t = time.time()
        queries = [q]
        hyde = None
        want_decomp = bool(opts.get("decompose"))
        want_hyde = bool(opts.get("hyde") and llm.available())

        if llm.available() and (want_decomp or want_hyde):
            def do_decomp():
                try:
                    out = llm.complete(
                        f"Split this question into at most 3 standalone search queries, one per line, no numbering:\n{q}",
                        max_tokens=256,
                    )
                    return [l.strip("-•* 0123456789.)").strip() for l in out.splitlines() if l.strip()][:3]
                except Exception:
                    return self._regex_decompose(q)

            def do_hyde():
                try:
                    return llm.complete(
                        f"Write a short factual passage (3 sentences) that would directly answer: {q}",
                        max_tokens=320,
                    )
                except Exception:
                    return None

            # Run LLM query decomposition and HyDE in parallel instead of sequentially.
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_dec = pool.submit(do_decomp) if want_decomp else None
                f_hyd = pool.submit(do_hyde) if want_hyde else None
                if f_dec:
                    parts = f_dec.result()
                    queries += [p for p in parts if p.lower() != q.lower()][:3]
                if f_hyd:
                    hyde = f_hyd.result() or None
        elif want_decomp:
            parts = self._regex_decompose(q)
            queries += [p for p in parts if p.lower() != q.lower()][:3]

        trace.append({"stage": "Query planning", "ms": int((time.time() - t) * 1000),
                      "detail": {"queries": queries, "hyde": bool(hyde)}})
        return queries, hyde

    # ---------- retrieval ----------
    def retrieve(self, q, opts=None, k=6, history=None, augment_fn=None):
        opts = {
            "hybrid": True, "graph": True, "rerank": True, "decompose": True, "hyde": True, "crag": True,
            **(opts or {}),
        }
        trace = []
        search_q = self._contextualize(q, history, trace)

        with self._lock:
            self.build()
            if not self.chunks:
                return [], trace, {}
            chunks = list(self.chunks)
            docs = dict(self.docs)
            bm25 = self.bm25
            embedder = self.embedder
            vecs = self.vecs
            graph = self.graph
            floor = self.floor
            bio = is_bio_domain(self.domain)

        queries, hyde = self.plan(search_q, opts, trace)

        t = time.time()
        rankings, weights, pool_n = [], [], max(40, k * 4)
        dense_w = 1.0 if embedder.st else LSA_WEIGHT
        qv = embedder.encode(queries, query=True, use_cache=False)
        # Encode HyDE as a *document/passage* (query=False, no BGE_QUERY prefix, no cache mutation).
        hyde_v = embedder.encode([hyde], query=False, use_cache=False)[0] if hyde else None

        for i, qq in enumerate(queries):
            q_w = 1.0 if i == 0 else SUBQUERY_WEIGHT
            if opts["hybrid"]:
                s = bm25.scores(tokens(qq))
                # Only include positive BM25 hits — never feed zero-score chunks into RRF.
                nz = np.flatnonzero(s > 0)
                if nz.size:
                    top_nz = nz[np.argsort(-s[nz])[:pool_n]]
                    rankings.append(list(top_nz))
                    weights.append(q_w)
            rankings.append(list(np.argsort(-(vecs @ qv[i]))[:pool_n]))
            weights.append(dense_w * q_w)

        if hyde_v is not None:
            rankings.append(list(np.argsort(-(vecs @ hyde_v))[:pool_n]))
            weights.append(dense_w * 0.85)

        order, fused = rrf(rankings, weights)
        trace.append({"stage": "Hybrid retrieval" if opts["hybrid"] else "Dense retrieval", "ms": int((time.time() - t) * 1000),
                      "detail": {"lists_fused": len(rankings), "candidates": len(order), "embedder": embedder.label}})

        t = time.time()
        graph_hits = set()
        if opts["graph"] and graph.number_of_nodes():
            all_q_text = " ".join(queries)
            padded = f" {re.sub(r'[^a-z0-9]+', ' ', all_q_text.lower())} "
            names = ((e, re.sub(r"[^a-z0-9]+", " ", e.lower()).strip()) for e in graph.nodes)
            q_ents = (
                {e for e, n in names if n and (f" {n} " in padded or f" {n}s " in padded)}
                | (entities(all_q_text, bio=bio) & set(graph.nodes))
            )

            def idf(e):
                return math.log(1 + len(chunks) / len(graph.nodes[e]["chunks"]))

            nbrs = set()
            for e in q_ents:
                nbrs |= set(sorted(graph[e], key=lambda n: -graph[e][n]["weight"] * idf(n))[:4])
            nbrs -= q_ents
            gscore = defaultdict(float)
            for e in q_ents | nbrs:
                for ci in graph.nodes[e]["chunks"]:
                    gscore[ci] += idf(e) * (1.0 if e in q_ents else NEIGHBOUR_WEIGHT)

            # For broad/overview queries, add top community representatives so GraphRAG covers the corpus.
            community_boosted = 0
            if GLOBAL_QUERY_RE.search(search_q) and graph.number_of_edges():
                top_nodes = sorted(graph.nodes, key=lambda n: len(graph.nodes[n]["chunks"]), reverse=True)[:50]
                sub = graph.subgraph(top_nodes)
                if sub.number_of_edges():
                    comms = list(nx.algorithms.community.greedy_modularity_communities(sub, weight="weight"))
                    for comm in comms[:6]:
                        rep = max(comm, key=lambda n: graph.degree(n))
                        for ci in list(graph.nodes[rep]["chunks"])[:2]:
                            gscore[ci] += COMMUNITY_WEIGHT * idf(rep)
                            community_boosted += 1

            graph_hits = set(sorted(gscore, key=gscore.get, reverse=True)[:pool_n])
            for rank, ci in enumerate(sorted(graph_hits, key=gscore.get, reverse=True)):
                fused[ci] += GRAPH_WEIGHT / (60 + rank + 1)
            order = sorted(fused, key=fused.get, reverse=True)
            detail = {"seed_entities": sorted(q_ents)[:8], "neighbours": sorted(nbrs)[:8], "chunks_boosted": len(graph_hits)}
            if community_boosted:
                detail["community_reps"] = community_boosted
            trace.append({"stage": "GraphRAG expansion", "ms": int((time.time() - t) * 1000), "detail": detail})

        t = time.time()
        cand_limit = max(30, k * 3)
        cand = order[:cand_limit]
        rel = vecs[cand] @ qv[0]
        if opts["rerank"] and cand:
            picked, lam = [], MMR_NEURAL if embedder.st else MMR_LSA
            fz = np.array([fused[c] for c in cand])
            fz = fz / (fz.max() + 1e-9)
            mix = 0.5 * dense_w
            ptp = float(np.ptp(rel))
            rel_norm = (rel - rel.min()) / (ptp + 1e-9) if ptp > 1e-9 else np.ones_like(rel)
            base = (1 - mix) * fz + mix * rel_norm
            # Pick extra candidates when CRAG is enabled so dropped irrelevant chunks can backfill up to k.
            target_n = min(len(cand), k * 2 if opts["crag"] else k)
            while len(picked) < target_n:
                best, bs = None, -1e9
                for j in range(len(cand)):
                    if j in picked:
                        continue
                    red = max((float(vecs[cand[j]] @ vecs[cand[p]]) for p in picked), default=0.0)
                    s = lam * base[j] - (1 - lam) * red
                    if s > bs:
                        best, bs = j, s
                picked.append(best)
            ranked_cands = [(cand[j], float(base[j])) for j in picked]
        else:
            target_n = min(len(cand), k * 2 if opts["crag"] else k)
            max_f = max((fused[c] for c in cand), default=1.0) or 1.0
            ranked_cands = [(c, float(fused[c] / max_f)) for c in cand[:target_n]]
        trace.append({"stage": "MMR rerank" if opts["rerank"] else "Top-k", "ms": int((time.time() - t) * 1000),
                      "detail": {"kept": min(k, len(ranked_cands))}})

        t = time.time()
        qtok = set(tokens(search_q))
        graded = []
        for ci, score in ranked_cands:
            c = chunks[ci]
            overlap = len(qtok & set(tokens(c["text"] + " " + c["ctx"]))) / (len(qtok) or 1)
            cos = (float(vecs[ci] @ qv[0]) - floor) / max(1.0 - floor, 1e-6)
            # Rebalanced CRAG grading: trust calibrated neural similarity so paraphrases aren't dropped.
            if embedder.st:
                grade = 0.25 * overlap + 0.75 * max(cos, 0.0)
            else:
                grade = 0.60 * overlap + 0.40 * max(cos, 0.0)
            label = "relevant" if grade >= 0.35 else "ambiguous" if grade >= 0.15 else "irrelevant"
            doc_info = docs.get(c["doc"], {"title": "Untitled", "source": "unknown"})
            graded.append({
                "chunk": c["id"], "doc": c["doc"], "title": doc_info["title"],
                "source": doc_info["source"], "text": c["text"], "score": round(score, 4),
                "grade": round(grade, 3), "label": label, "via_graph": ci in graph_hits,
            })

        crag = {}
        if opts["crag"]:
            kept = [r for r in graded if r["label"] != "irrelevant"][:k]
            evaluated = graded[:max(k, len(kept))]
            dropped = sum(1 for r in evaluated if r["label"] == "irrelevant")
            action = (
                "answer" if any(r["label"] == "relevant" for r in kept)
                else "augment" if kept else "knowledge_gap"
            )
            crag = {
                "relevant": sum(r["label"] == "relevant" for r in kept),
                "ambiguous": sum(r["label"] == "ambiguous" for r in kept),
                "dropped": dropped,
                "action": action,
            }
            trace.append({"stage": "Corrective grading (CRAG)", "ms": int((time.time() - t) * 1000), "detail": crag})

            # True corrective loop: when a knowledge gap occurs and an external fallback is wired, fetch & retry once.
            if action == "knowledge_gap" and augment_fn is not None:
                t_aug = time.time()
                try:
                    added = augment_fn(search_q)
                except Exception as e:
                    added = 0
                    trace.append({"stage": "CRAG PubMed fallback", "ms": int((time.time() - t_aug) * 1000),
                                  "detail": {"error": str(e)}})
                if added:
                    trace.append({"stage": "CRAG PubMed fallback", "ms": int((time.time() - t_aug) * 1000),
                                  "detail": {"imported": added, "action": "re-retrieving"}})
                    retry_hits, retry_trace, retry_crag = self.retrieve(search_q, opts, k=k, history=None, augment_fn=None)
                    retry_crag["augmented_papers"] = added
                    return retry_hits, trace + retry_trace, retry_crag
            results = kept
        else:
            results = graded[:k]

        return results, trace, crag

    # ---------- generation ----------
    def answer(self, q, opts=None, k=6, history=None, augment_fn=None):
        return list(self.answer_stream(q, opts, k, history=history, augment_fn=augment_fn))[-1]["result"]

    def answer_stream(self, q, opts=None, k=6, history=None, augment_fn=None):
        """Yield {"sources"}, then {"delta"} per piece of text, then {"result"} (authoritative)."""
        t0 = time.time()
        hits, trace, crag = self.retrieve(q, opts, k, history=history, augment_fn=augment_fn)
        rw = next((t["detail"].get("rewritten") for t in trace if t["stage"] == "Follow-up rewrite"), None)
        eff_q = rw or q
        yield {"sources": hits, "crag": crag}
        t = time.time()
        if not hits:
            ans = (
                "I couldn't find grounded evidence for that in this workspace. "
                "Add documents in Knowledge, or fetch papers from PubMed/bioRxiv, then ask again."
            )
            mode = "none"
        elif llm.available():
            ctx = "\n\n".join(f"[{i+1}] ({h['title']}) {h['text']}" for i, h in enumerate(hits))
            try:
                ans = ""
                for piece in llm.stream(
                    f"Context passages:\n{ctx}\n\nQuestion: {q}\n\n"
                    "Answer using only the passages. Cite every claim with [n]. "
                    "If the passages don't cover something, say so plainly.",
                    system=f"You are PandaRAG, a precise research assistant for the {self.domain} domain.",
                    history=history,
                ):
                    ans += piece
                    yield {"delta": piece}
                ans = ans.strip()
                mode = f"generative · {llm.label()}"
            except Exception as e:
                why = e if type(e) is RuntimeError else type(e).__name__
                ans, mode = self._extractive(eff_q, hits), f"extractive (LLM error: {why})"
        else:
            ans, mode = self._extractive(eff_q, hits), "extractive"
        trace.append({"stage": "Grounded generation", "ms": int((time.time() - t) * 1000), "detail": {"mode": mode}})
        t = time.time()
        metrics = self._reflect(eff_q, ans, hits)
        trace.append({"stage": "Self-reflection", "ms": int((time.time() - t) * 1000), "detail": metrics})
        yield {"result": {
            "answer": ans, "sources": hits, "trace": trace, "metrics": metrics, "crag": crag,
            "mode": mode, "total_ms": int((time.time() - t0) * 1000),
        }}

    def _extractive(self, q, hits, n=4):
        qtok = set(tokens(q))
        scored, fallback = [], []
        for i, h in enumerate(hits):
            for s in sentences(h["text"]):
                st = set(tokens(s))
                score = len(qtok & st) / (len(qtok) or 1) + 0.15 * h["grade"] - 0.01 * i
                if len(st) >= 4:
                    scored.append((score, s, i + 1))
                elif s.strip():
                    fallback.append((score, s.strip(), i + 1))
        pool = scored or fallback
        seen, out = set(), []
        for _, s, ref in sorted(pool, reverse=True):
            if s not in seen:
                seen.add(s)
                out.append(f"{s} [{ref}]")
            if len(out) == n:
                break
        if not out and hits:
            out.append(f"{hits[0]['text'][:300].strip()} [1]")
        return " ".join(out)

    def _reflect(self, q, ans, hits):
        if not hits or not ans.strip():
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0}
        support_texts = [h["text"] for h in hits]
        support_toks = [set(tokens(t)) for t in support_texts]
        support_negs = [negations_in(t) for t in support_texts]

        # Reorder "Claim. [1] Next" -> "Claim [1]. Next" before sentence splitting.
        norm_ans = re.sub(r"([.!?])((?:\s*\[\d+\])+)", r"\2\1", ans)
        sents_ = [s for s in sentences(norm_ans) if len(tokens(s)) >= 3]

        use_neural = bool(getattr(self, "embedder", None) and self.embedder.st)
        floor = getattr(self, "floor", 0.0)
        sent_vecs = hit_vecs = q_vec = ans_vec = None
        if use_neural and sents_:
            clean_sents = [re.sub(r"\[\d+\]", "", s).strip() for s in sents_]
            sent_vecs = self.embedder.encode(clean_sents, query=False, use_cache=False)
            hit_vecs = self.embedder.encode(support_texts, query=False, use_cache=False)
            q_vec = self.embedder.encode([q], query=True, use_cache=False)[0]
            ans_vec = self.embedder.encode([re.sub(r"\[\d+\]", "", ans).strip()], query=False, use_cache=False)[0]

        ok = 0
        for idx_s, s in enumerate(sents_):
            clean_s = re.sub(r"\[\d+\]", "", s).strip()
            if ABSTENTION_RE.search(clean_s):
                ok += 1
                continue
            refs = [int(r) - 1 for r in re.findall(r"\[(\d+)\]", s) if 0 < int(r) <= len(hits)]
            target_ids = refs if refs else list(range(len(hits)))

            st = set(tokens(clean_s))
            pool_toks = set().union(*(support_toks[r] for r in target_ids))
            lex_cov = len(st & pool_toks) / (len(st) or 1)

            # Detect polarity/negation flip (e.g. "does NOT cause X" when source asserts "causes X").
            s_neg = bool(negations_in(clean_s))
            p_neg = any(bool(support_negs[r]) for r in target_ids)
            polarity_conflict = s_neg and not p_neg

            if polarity_conflict:
                supported = False
            elif lex_cov >= 0.58:
                supported = True
            elif use_neural and sent_vecs is not None:
                best_cos = max(float(sent_vecs[idx_s] @ hit_vecs[r]) for r in target_ids)
                cal_sim = (best_cos - floor) / max(1.0 - floor, 1e-6)
                supported = cal_sim >= 0.45
            else:
                supported = False
            ok += int(supported)

        faithfulness = ok / (len(sents_) or 1)

        qtok = set(tokens(q))
        lex_rel = len(qtok & set(tokens(ans))) / (len(qtok) or 1)
        if use_neural and q_vec is not None and ans_vec is not None:
            sem_rel = max(0.0, min(1.0, (float(q_vec @ ans_vec) - floor) / max(1.0 - floor, 1e-6)))
            answer_relevance = min(1.0, 0.72 * sem_rel + 0.28 * lex_rel)
        else:
            answer_relevance = lex_rel

        # Rank-aware Average Precision @ k over relevant hits.
        rel_flags = [1 if h["label"] == "relevant" else 0 for h in hits]
        if sum(rel_flags):
            ap = sum((sum(rel_flags[:i + 1]) / (i + 1)) * rel_flags[i] for i in range(len(rel_flags))) / sum(rel_flags)
            # Blend MAP with fraction of relevant hits so 1 relevant at rank 1 out of 6 doesn't look like 100%.
            context_precision = 0.5 * ap + 0.5 * (sum(rel_flags) / len(rel_flags))
        else:
            context_precision = 0.0

        return {
            "faithfulness": round(float(faithfulness), 2),
            "answer_relevance": round(float(answer_relevance), 2),
            "context_precision": round(float(context_precision), 2),
        }

    # ---------- RAG -> Causal KG bridge ----------
    def extract_causal_edges(self, q: str, hits: list[dict]) -> list[dict]:
        """Extract candidate causal hops from retrieved passages to pre-populate the Causal tab."""
        if not hits:
            return []
        if llm.available():
            ctx = "\n\n".join(f"[{i+1}] ({h['title']}) {h['text']}" for i, h in enumerate(hits[:5]))
            try:
                raw = llm.complete(
                    f"From the passages below, extract 2 to 4 mechanistic causal hops relevant to: {q}\n"
                    "Return ONLY a JSON array of objects with keys: source, relation, target, evidence, year, confidence.\n"
                    "Valid relations: INHIBITS, ACTIVATES, INDUCES, PREVENTS, CAUSES, LEADS_TO, REGULATES, TREATS.\n"
                    "Valid evidence: RCT, clinical_observational, meta_analysis, mechanism_validated, in_vivo, in_vitro, literature_mining.\n"
                    f"Passages:\n{ctx}",
                    max_tokens=600,
                )
                m = re.search(r"\[.*\]", raw, re.S)
                if m:
                    parsed = json.loads(m.group(0))
                    clean = [
                        {
                            "source": str(e.get("source", "")).strip(),
                            "relation": str(e.get("relation", "LEADS_TO")).strip().upper(),
                            "target": str(e.get("target", "")).strip(),
                            "evidence": str(e.get("evidence", "literature_mining")).strip(),
                            "year": int(e["year"]) if str(e.get("year", "")).isdigit() else int(time.strftime("%Y")),
                            "confidence": max(0.1, min(1.0, float(e.get("confidence", 0.75)))),
                        }
                        for e in parsed if isinstance(e, dict) and e.get("source") and e.get("target")
                    ]
                    if clean:
                        return clean[:5]
            except Exception:
                pass

        # Deterministic offline fallback: mine causal verb cues and entity pairs from retrieved sentences.
        cue_map = [
            (re.compile(r"\b(?:inhibit|block|suppress|antagoni[sz])", re.I), "INHIBITS"),
            (re.compile(r"\b(?:activat|enhanc|promot|upregulat|overexpress)", re.I), "ACTIVATES"),
            (re.compile(r"\b(?:induc|trigger|caus|driv|limit|mediat)", re.I), "CAUSES"),
            (re.compile(r"\b(?:prevent|reduc|protect|attenuat)", re.I), "PREVENTS"),
            (re.compile(r"\b(?:treat|therap|stabilis|crosslink)", re.I), "TREATS"),
        ]
        bio = is_bio_domain(self.domain)
        edges, seen = [], set()
        for h in hits[:5]:
            ym = re.search(r"\b(19\d{2}|20\d{2})\b", h["text"])
            year = int(ym.group(1)) if ym else int(time.strftime("%Y"))
            for s in sentences(h["text"]):
                ents = sorted(entities(s, bio=bio), key=lambda e: s.find(e))
                if len(ents) < 2:
                    continue
                rel = next((r for pat, r in cue_map if pat.search(s)), "LEADS_TO")
                key = (ents[0].lower(), ents[1].lower())
                if key in seen:
                    continue
                seen.add(key)
                edges.append({
                    "source": ents[0],
                    "relation": rel,
                    "target": ents[1],
                    "evidence": "literature_mining",
                    "year": year,
                    "confidence": round(min(0.95, max(0.5, 0.55 + 0.4 * h.get("grade", 0.5))), 2),
                })
                if len(edges) >= 4:
                    return edges
        return edges

    # ---------- persistence helpers ----------
    def to_json(self):
        with self._lock:
            return {"id": self.id, "name": self.name, "domain": self.domain, "docs": list(self.docs.values())}

    def summary(self):
        with self._lock:
            return {"id": self.id, "name": self.name, "domain": self.domain, "docs": len(self.docs), "chunks": len(self.chunks)}


class Store:
    """SQLite-backed workspace store with automatic legacy store.json migration and persistent vector cache."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.disk_cache = DiskVecCache(self.db_path)
        self.ws = {}
        self._init_db()
        self._load()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    domain TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    text TEXT NOT NULL,
                    added TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_docs_ws ON documents(workspace_id);
                """
            )

    def _load(self):
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute("SELECT id, name, domain FROM workspaces").fetchall()
            if rows:
                for wid, name, domain in rows:
                    ws = Workspace(name, domain, wid, disk_cache=self.disk_cache)
                    doc_rows = con.execute(
                        "SELECT id, title, source, text, added FROM documents WHERE workspace_id=? ORDER BY rowid",
                        (wid,),
                    ).fetchall()
                    for did, title, source, text, added in doc_rows:
                        ws.add(title, text, source, did=did, added=added, dedup=False)
                    self.ws[wid] = ws
                return

        # Migrate from legacy store.json on first boot if present.
        if self.db_path == DB_PATH and LEGACY_STORE.exists():
            try:
                for w in json.loads(LEGACY_STORE.read_text()):
                    ws = Workspace(w["name"], w["domain"], w["id"], disk_cache=self.disk_cache)
                    for d in w.get("docs", []):
                        ws.add(
                            d["title"], d["text"], d.get("source", "manual"),
                            did=d.get("id"), added=d.get("added"), dedup=True,
                        )
                    self.ws[ws.id] = ws
                if self.ws:
                    self.save()
            except Exception:
                pass

    def save(self):
        with self.lock, sqlite3.connect(self.db_path) as con:
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("DELETE FROM documents")
            con.execute("DELETE FROM workspaces")
            for ws in self.ws.values():
                with ws._lock:
                    con.execute(
                        "INSERT INTO workspaces (id, name, domain) VALUES (?, ?, ?)",
                        (ws.id, ws.name, ws.domain),
                    )
                    con.executemany(
                        "INSERT INTO documents (id, workspace_id, title, source, text, added, content_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            (d["id"], ws.id, d["title"], d["source"], d["text"], d["added"], d.get("hash") or text_hash(d["text"]))
                            for d in ws.docs.values()
                        ],
                    )
