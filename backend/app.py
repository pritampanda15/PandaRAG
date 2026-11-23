"""PandaRAG API."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import labs
import llm
import rag

FRONTEND = Path(__file__).parent.parent / "frontend"
app = FastAPI(title="PandaRAG")
store = rag.Store()

if not store.ws:
    ws = rag.Workspace("Oncology Starter", "Oncology")
    for d in labs.oncology_corpus():
        ws.add(d["title"], d["text"], d["source"])
    store.ws[ws.id] = ws
    store.save()


def get_ws(wid):
    if wid not in store.ws:
        raise HTTPException(404, "workspace not found")
    return store.ws[wid]


class WSIn(BaseModel):
    name: str
    domain: str = "General"


class DocIn(BaseModel):
    title: str
    text: str
    source: str = "manual"


class AskIn(BaseModel):
    query: str
    k: int = 6
    hybrid: bool = True
    graph: bool = True
    rerank: bool = True
    decompose: bool = True
    hyde: bool = True
    crag: bool = True


class LitIn(BaseModel):
    query: str
    sources: list[str] = ["pubmed"]
    max_results: int = 12
    days_back: int | None = None


class ImportIn(BaseModel):
    papers: list[dict]


class PathIn(BaseModel):
    edges: list[dict]


class ScreenIn(BaseModel):
    smiles: list[str]
    n_select: int = 0
    nbits: int = 1024
    radius: int = 2


@app.get("/api/health")
def health():
    return {"ok": True, "llm": llm.available(), "model": llm.MODEL if llm.available() else None,
            "labs": labs.status(), "workspaces": [w.summary() for w in store.ws.values()]}


@app.get("/api/workspaces")
def list_ws():
    return [w.summary() for w in store.ws.values()]


@app.post("/api/workspaces")
def create_ws(body: WSIn):
    with store.lock:
        ws = rag.Workspace(body.name, body.domain)
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
    return [{**d, "text": d["text"][:400], "chars": len(d["text"]),
             "chunks": sum(c["doc"] == d["id"] for c in ws.chunks)} for d in ws.docs.values()]


@app.post("/api/workspaces/{wid}/docs")
def add_doc(wid: str, body: DocIn):
    if not body.text.strip():
        raise HTTPException(400, "text is empty")
    with store.lock:
        d = get_ws(wid).add(body.title, body.text, body.source)
        store.save()
    return {"id": d["id"], "title": d["title"]}


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
    opts = body.model_dump(exclude={"query", "k"})
    return ws.answer(body.query, opts, max(1, min(body.k, 12)))


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
    n = 0
    with store.lock:
        for p in body.papers:
            text = (p.get("abstract") or "").strip()
            if not text:
                continue
            ws.add(p.get("title", "Untitled"), text, p.get("source", "literature"))
            n += 1
        store.save()
    return {"imported": n}


@app.get("/api/admet")
def admet(smiles: str):
    try:
        return labs.admet_predict(smiles)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(503, f"{type(e).__name__}: {e}")


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


@app.post("/api/screen")
def screen(body: ScreenIn):
    smiles = [s.strip() for s in body.smiles if s.strip()][:60]
    if not smiles:
        raise HTTPException(400, "no SMILES supplied")
    try:
        return labs.screen(smiles, body.n_select, max(64, min(body.nbits, 4096)), max(0, min(body.radius, 4)))
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
    """Ingest a corpus that ships with one of the repos."""
    corpora = {"oncology": labs.oncology_corpus, "repurposing": labs.repurposing_corpus}
    if which not in corpora:
        raise HTTPException(404, "unknown corpus")
    ws = get_ws(wid)
    with store.lock:
        n = sum(bool(ws.add(d["title"], d["text"], d["source"])) for d in corpora[which]())
        store.save()
    return {"imported": n}


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
