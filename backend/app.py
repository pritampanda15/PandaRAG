"""PandaRAG API."""
from html.parser import HTMLParser
import ipaddress
import json
import logging
import os
import re
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import iterate_in_threadpool, run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests

import labs
import llm
import pdf
import rag

MAX_PDF = 50 * 1024 * 1024
MAX_URL_BYTES = 5 * 1024 * 1024
FRONTEND = Path(__file__).parent.parent / "frontend"
app = FastAPI(title="PandaRAG")
store = rag.Store()

if not store.ws:
    ws = rag.Workspace("Oncology Starter", "Oncology", disk_cache=store.disk_cache)
    for d in labs.oncology_corpus():
        ws.add(d["title"], d["text"], d["source"])
    store.ws[ws.id] = ws
    store.save()


def get_ws(wid):
    if wid not in store.ws:
        raise HTTPException(404, "workspace not found")
    return store.ws[wid]


class _HTMLText(HTMLParser):
    """Lightweight stdlib HTML-to-text extractor that drops scripts, styles, nav, and headers."""
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self._in_title = False
        self.title = []
        self.parts = []

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in self.SKIP:
            self._skip += 1
        elif t == "title":
            self._in_title = True
        elif t in ("p", "div", "br", "li", "h1", "h2", "h3", "h4", "tr", "article", "section"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in self.SKIP and self._skip:
            self._skip -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title.append(data)
        else:
            self.parts.append(data)

    def result(self):
        title = re.sub(r"\s+", " ", "".join(self.title)).strip()
        text = re.sub(r"[ \t]+", " ", "".join(self.parts))
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return title, text


def _safe_fetch_url(raw_url: str) -> dict:
    url = raw_url.strip()
    if re.match(r"^10\.\d{4,9}/\S+$", url):
        url = f"https://doi.org/{url}"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("URL must start with http://, https://, or a valid DOI (10.xxxx/...)")
    # Block loopback/private/link-local targets (basic SSRF guard).
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise ValueError("refusing to fetch private or local network address")
    except socket.gaierror as e:
        raise ValueError(f"could not resolve host: {parsed.hostname}") from e

    with requests.get(
        url, headers={"User-Agent": "PandaRAG/1.0 (+https://github.com/pandarag)"},
        timeout=20, stream=True, allow_redirects=True,
    ) as r:
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=65536):
            buf += chunk
            if len(buf) > MAX_URL_BYTES:
                raise ValueError("remote document exceeds 5 MB limit")

    if "application/pdf" in ctype or url.lower().endswith(".pdf"):
        extracted = pdf.extract(bytes(buf))
        title = Path(urlparse(r.url).path).stem or parsed.hostname
        return {"title": title, "text": extracted["text"], "url": r.url}

    body = buf.decode("utf-8", errors="replace")
    if "html" in ctype or "<html" in body[:500].lower():
        parser = _HTMLText()
        parser.feed(body)
        title, text = parser.result()
    else:
        title, text = parsed.hostname, body.strip()
    if not text:
        raise ValueError("no readable text found at that URL")
    return {"title": title or parsed.hostname, "text": text, "url": r.url}


def _make_pubmed_augment(ws: rag.Workspace):
    def augment(q: str) -> int:
        res = labs.literature_search(q, sources=("pubmed",), max_results=5)
        added = 0
        with store.lock:
            for p in res.get("papers", []):
                text = (p.get("abstract") or "").strip()
                if not text:
                    continue
                d = ws.add(p.get("title", "Untitled"), text, p.get("source", "pubmed"))
                if not d.get("duplicate"):
                    added += 1
            if added:
                store.save()
        return added
    return augment


class WSIn(BaseModel):
    name: str
    domain: str = "General"


class DocIn(BaseModel):
    title: str
    text: str
    source: str = "manual"


class BatchDocsIn(BaseModel):
    docs: list[DocIn]


class UrlIn(BaseModel):
    url: str


class AskIn(BaseModel):
    query: str
    k: int = 6
    hybrid: bool = True
    graph: bool = True
    rerank: bool = True
    decompose: bool = True
    hyde: bool = True
    crag: bool = True
    auto_pubmed: bool = False
    history: list[dict] = Field(default_factory=list)


class CompareIn(BaseModel):
    query: str
    k: int = 6
    config_a: dict = Field(default_factory=lambda: {
        "hybrid": True, "graph": False, "rerank": False, "decompose": False, "hyde": False, "crag": False,
    })
    config_b: dict = Field(default_factory=lambda: {
        "hybrid": True, "graph": True, "rerank": True, "decompose": True, "hyde": True, "crag": True,
    })


class CausalExtractIn(BaseModel):
    query: str
    sources: list[dict]


class LitIn(BaseModel):
    query: str
    sources: list[str] = ["pubmed"]
    max_results: int = 12
    days_back: int | None = None


class ImportIn(BaseModel):
    papers: list[dict]


class PathIn(BaseModel):
    edges: list[dict]


class LLMConfigIn(BaseModel):
    provider: str = "extractive"
    model: str = ""
    api_key: str = ""
    base_url: str = ""


@app.get("/api/health")
def health():
    li = llm.info()
    embed_name, _ = rag.neural_model()
    return {
        "ok": True,
        "llm": li["available"],
        "provider": li["provider"],
        "model": li["model"],
        "base_url": li["base_url"],
        "embedder": embed_name or "LSA-256 (local)",
        "plugins": os.getenv("PANDARAG_PLUGINS", "bio").strip().lower(),
        "labs": labs.status(),
        "workspaces": [w.summary() for w in store.ws.values()],
    }


@app.post("/api/llm/config")
def set_llm_config(body: LLMConfigIn):
    try:
        return llm.configure(body.provider, body.model, body.api_key, body.base_url)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/workspaces")
def list_ws():
    return [w.summary() for w in store.ws.values()]


@app.post("/api/workspaces")
def create_ws(body: WSIn):
    with store.lock:
        ws = rag.Workspace(body.name, body.domain, disk_cache=store.disk_cache)
        store.ws[ws.id] = ws
        store.save()
    return ws.summary()


@app.delete("/api/workspaces/{wid}")
def delete_ws(wid: str):
    with store.lock:
        get_ws(wid)
        del store.ws[wid]
        store.save()
    return {"ok": True}


@app.get("/api/workspaces/{wid}/docs")
def docs(wid: str):
    ws = get_ws(wid)
    with ws._lock:
        counts = {}
        for c in ws.chunks:
            counts[c["doc"]] = counts.get(c["doc"], 0) + 1
        return [
            {**d, "text": d["text"][:400], "chars": len(d["text"]), "chunks": counts.get(d["id"], 0)}
            for d in ws.docs.values()
        ]


@app.post("/api/workspaces/{wid}/docs")
def add_doc(wid: str, body: DocIn):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    with store.lock:
        d = get_ws(wid).add(body.title, body.text, body.source)
        if not d.get("duplicate"):
            store.save()
    return {"id": d["id"], "title": d["title"], "duplicate": bool(d.get("duplicate"))}


@app.post("/api/workspaces/{wid}/docs/batch")
def add_docs_batch(wid: str, body: BatchDocsIn):
    """Ingest many documents in one atomic SQLite save (index rebuilds once on next query)."""
    ws = get_ws(wid)
    imported, skipped, results = 0, 0, []
    with store.lock:
        for item in body.docs:
            text = (item.text or "").strip()
            if not text:
                continue
            d = ws.add(item.title, text, item.source or "upload")
            if d.get("duplicate"):
                skipped += 1
                results.append({"title": d["title"], "status": "duplicate"})
            else:
                imported += 1
                results.append({"id": d["id"], "title": d["title"], "status": "imported"})
        if imported:
            store.save()
    return {"imported": imported, "skipped": skipped, "results": results}


@app.post("/api/pdf")
async def pdf_text(request: Request):
    """Raw PDF body in, text out; the page shows it for review before it is ingested."""
    data = bytearray()
    async for part in request.stream():
        data += part
        if len(data) > MAX_PDF:
            raise HTTPException(413, "PDF is larger than 50 MB")
    try:
        return await run_in_threadpool(pdf.extract, bytes(data))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.post("/api/url")
async def fetch_url(body: UrlIn):
    """Fetch a webpage, PDF URL, or DOI and return cleaned text for review."""
    try:
        return await run_in_threadpool(_safe_fetch_url, body.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.delete("/api/workspaces/{wid}/docs/{did}")
def del_doc(wid: str, did: str):
    with store.lock:
        get_ws(wid).remove(did)
        store.save()
    return {"ok": True}


@app.post("/api/workspaces/{wid}/ask")
def ask(wid: str, body: AskIn):
    if not body.query.strip():
        raise HTTPException(400, "query is empty")
    ws = get_ws(wid)
    opts = body.model_dump(exclude={"query", "k", "history", "auto_pubmed"})
    aug = _make_pubmed_augment(ws) if body.auto_pubmed else None
    return ws.answer(body.query, opts, max(1, min(body.k, 12)), history=body.history, augment_fn=aug)


@app.post("/api/workspaces/{wid}/ask/stream")
def ask_stream(wid: str, body: AskIn, request: Request):
    """Same as /ask, as one JSON event per line."""
    if not body.query.strip():
        raise HTTPException(400, "query is empty")
    ws = get_ws(wid)
    opts = body.model_dump(exclude={"query", "k", "history", "auto_pubmed"})
    aug = _make_pubmed_augment(ws) if body.auto_pubmed else None

    events = ws.answer_stream(body.query, opts, max(1, min(body.k, 12)), history=body.history, augment_fn=aug)

    async def lines():
        try:
            async for event in iterate_in_threadpool(events):
                if await request.is_disconnected():
                    return
                yield json.dumps(event) + "\n"
        except Exception as e:
            logging.exception("ask failed")
            yield json.dumps({"error": f"{type(e).__name__}: {e}"}) + "\n"
        finally:
            events.close()
    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/api/workspaces/{wid}/compare")
def compare(wid: str, body: CompareIn):
    """Side-by-side RAG ablation: run the same query under two toggle presets."""
    if not body.query.strip():
        raise HTTPException(400, "query is empty")
    ws = get_ws(wid)
    k = max(1, min(body.k, 12))
    a = ws.answer(body.query, body.config_a, k)
    b = ws.answer(body.query, body.config_b, k)
    return {"query": body.query, "a": a, "b": b}


@app.post("/api/workspaces/{wid}/causal-extract")
def causal_extract(wid: str, body: CausalExtractIn):
    ws = get_ws(wid)
    edges = ws.extract_causal_edges(body.query, body.sources)
    return {"edges": edges}


@app.get("/api/workspaces/{wid}/graph")
def graph(wid: str):
    return get_ws(wid).graph_view()


@app.post("/api/literature")
def literature(body: LitIn):
    try:
        return labs.literature_search(body.query, tuple(body.sources), body.max_results, body.days_back)
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")


@app.post("/api/workspaces/{wid}/import")
def import_papers(wid: str, body: ImportIn):
    ws = get_ws(wid)
    n, skipped = 0, 0
    with store.lock:
        for p in body.papers:
            text = (p.get("abstract") or "").strip()
            if not text:
                continue
            d = ws.add(p.get("title", "Untitled"), text, p.get("source", "literature"))
            if d.get("duplicate"):
                skipped += 1
            else:
                n += 1
        if n:
            store.save()
    return {"imported": n, "skipped": skipped}


@app.get("/api/causal/schema")
def causal_schema():
    try:
        return labs.causal_schema()
    except Exception as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}")


@app.post("/api/causal/score")
def causal_score(body: PathIn):
    try:
        return labs.score_causal_path(body.edges)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}")


@app.get("/api/repurposing")
def repurposing():
    try:
        st = {**labs.repurposing_status(), "corpus": len(labs.repurposing_corpus())}
        if not st["missing"]:
            st["card"] = labs.repurposing_model_card()
        return st
    except Exception as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}")


@app.get("/api/repurposing/vocab")
def repurposing_vocab(q: str = "", kind: str = "drug", limit: int = 25):
    try:
        return labs.repurposing_vocab(q, kind, max(1, min(limit, 2000)))
    except Exception as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}")


@app.get("/api/repurposing/predict")
def repurposing_predict(drug: str, disease: str | None = None, top_k: int = 10):
    try:
        return labs.repurposing_predict(drug, disease, max(1, min(top_k, 25)))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}")


@app.post("/api/workspaces/{wid}/seed/{which}")
def seed(wid: str, which: str):
    """Ingest a corpus that ships with one of the repos (deduplicated)."""
    corpora = {"oncology": labs.oncology_corpus, "repurposing": labs.repurposing_corpus}
    if which not in corpora:
        raise HTTPException(404, "unknown corpus")
    ws = get_ws(wid)
    n, skipped = 0, 0
    with store.lock:
        for d in corpora[which]():
            res = ws.add(d["title"], d["text"], d["source"])
            if res.get("duplicate"):
                skipped += 1
            else:
                n += 1
        if n:
            store.save()
    return {"imported": n, "skipped": skipped}


@app.middleware("http")
async def revalidate(request, call_next):
    """No build step means no hashed filenames, so browsers must revalidate or they mix old and new files."""
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
