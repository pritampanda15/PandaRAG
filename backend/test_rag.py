"""Self-check: python3 test_rag.py (no pytest needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rag

ws = rag.Workspace("t", "Test")
ws.add("Cisplatin", "Cisplatin is a platinum agent that crosslinks DNA. Resistance arises via ERCC1 overexpression "
                    "and enhanced nucleotide excision repair. Nephrotoxicity limits the cumulative dose in patients. "
                    "Amifostine is used as a cytoprotective agent during therapy.")
ws.add("Paclitaxel", "Paclitaxel stabilises microtubules and arrests mitosis. Resistance involves TUBB3 isotype "
                     "switching and ABCB1 mediated efflux. Peripheral neuropathy is the dose limiting toxicity. "
                     "Albumin bound formulations reduce hypersensitivity reactions.")

assert len(ws.docs) == 2 and len(ws.chunks) >= 2, "chunking produced nothing"

hits, trace, crag = ws.retrieve("What drives cisplatin resistance?")
assert hits, "no hits retrieved"
assert "Cisplatin" in hits[0]["title"], f"wrong top doc: {hits[0]['title']}"
assert {t["stage"] for t in trace} >= {"Query planning", "MMR rerank"}, [t["stage"] for t in trace]

out = ws.answer("What drives cisplatin resistance?")
assert "ERCC1" in out["answer"], out["answer"]
assert out["metrics"]["faithfulness"] >= 0.5, out["metrics"]
assert out["sources"], "answer had no sources"

# streaming: sources, then text, then a result that replaces the text if the LLM fails part-way
saved = rag.llm.available, rag.llm.stream, rag.llm.complete
rag.llm.available = lambda: False
plain = ws.answer("What drives cisplatin resistance?")["answer"]
events = list(ws.answer_stream("What drives cisplatin resistance?"))
assert [next(iter(e)) for e in events] == ["sources", "result"] and events[-1]["result"]["answer"] == plain


def broken(*a, **k):
    yield "half an ans"
    raise RuntimeError("max_tokens")


rag.llm.available, rag.llm.complete = lambda: True, lambda *a: ""
rag.llm.stream = lambda *a, **k: iter(["ERCC1 ", "drives it [1]. "])
events = list(ws.answer_stream("What drives cisplatin resistance?"))
assert [next(iter(e)) for e in events] == ["sources", "delta", "delta", "result"], events
assert events[-1]["result"]["answer"] == "ERCC1 drives it [1]." and "generative" in events[-1]["result"]["mode"]
rag.llm.stream = broken
res = ws.answer("What drives cisplatin resistance?")
assert res["answer"] == plain and "max_tokens" in res["mode"], res
rag.llm.available, rag.llm.stream, rag.llm.complete = saved

# a citation after the full stop belongs to the sentence before it
two = [{"text": "Alpha kinase phosphorylates tubulin strongly.", "label": "relevant"},
       {"text": "Omega lipid membranes absorb cholesterol slowly.", "label": "relevant"}]
quoted = "Alpha kinase phosphorylates tubulin strongly. [1] Omega lipid membranes absorb cholesterol slowly. [2]"
assert ws._reflect("q", quoted, two)["faithfulness"] == 1.0
assert ws._reflect("q", quoted.replace("[1]", "[x]").replace("[2]", "[1]").replace("[x]", "[2]"), two)["faithfulness"] == 0
assert ws._reflect("q", "Alpha kinase phosphorylates tubulin strongly [1].", two)["faithfulness"] == 1.0

ents = rag.entities("We provide EGFR-mutant data; increase in kinase, stage III disease, osimertinib and PC-9 remain. "
                    "Combine MET-amplified KRAS with a peptide.")
assert ents >= {"EGFR", "kinase", "osimertinib", "PC-9", "MET", "KRAS", "peptide"}, ents
assert not ents & {"provide", "increase", "remain", "disease", "EGFR-", "MET-", "III"}, ents
gw = rag.Workspace("g", "Test")
gw.add("g", "MET and RAS signalling interact with EGFR in tumours.")
_, tr, _ = gw.retrieve("Which method covers EGFRs and KRAS?")
seeds = next(x for x in tr if x["stage"] == "GraphRAG expansion")["detail"]["seed_entities"]
assert seeds == ["EGFR"], seeds

# graph expansion may reorder, but must not let a hub entity outvote the searches
assert rag.GRAPH_WEIGHT < 1
fused = rag.rrf([[0, 1], [1, 0]], [1, 0.1])[0]
assert fused == [0, 1], "a low-weight list overturned the main one"

if ws.embedder.st:   # neural: a rebuild re-encodes only new chunks, and removed ones leave the cache
    before = dict(ws._vec_cache)
    did = ws.add("Tmp", "Docetaxel is a taxane that stabilises microtubules in dividing tumour cells.")["id"]
    ws.build()
    assert all(ws._vec_cache[t] is v for t, v in before.items()) and len(ws._vec_cache) == len(before) + 1
    ws.remove(did); ws.build()
    assert set(ws._vec_cache) == set(before)

g = ws.graph_view()
assert g["nodes"], "entity graph empty"

empty = rag.Workspace("e")
assert empty.answer("anything")["sources"] == [], "empty workspace should return no sources"

ws.remove(list(ws.docs)[0])
assert len(ws.docs) == 1 and all(c["doc"] in ws.docs for c in ws.chunks), "removal left orphan chunks"

import io
import pdf
import pypdf


def make_pdf(lines, password=None):
    w = pypdf.PdfWriter()
    page = w.add_blank_page(300, 300)
    if lines:
        N, D = pypdf.generic.NameObject, pypdf.generic.DictionaryObject
        font = D({N("/Type"): N("/Font"), N("/Subtype"): N("/Type1"), N("/BaseFont"): N("/Helvetica")})
        page[N("/Resources")] = D({N("/Font"): D({N("/F1"): font})})
        s = pypdf.generic.DecodedStreamObject()
        s.set_data(("BT /F1 12 Tf 20 200 Td " + " ".join(f"({x}) Tj 0 -20 Td" for x in lines) + " ET").encode())
        page[N("/Contents")] = w._add_object(s)
    if password is not None:
        w.encrypt(password, "owner", algorithm="AES-256")
    out = io.BytesIO()
    w.write(out)
    return out.getvalue()


body = [f"Unique sentence number {w} in the body text here." for w in "abcdefghijkl"]
made = [f"ACS Omega Article\nPage {i + 1} of 4\nPatient {i + 1}\n{0.5 + i}\n" + "\n".join(f"{b} {i}" for b in body) +
        f"\nACS Omega Article again {i}\n{56651 + i}" for i in range(4)]
made[2] = made[2].replace(body[6] + " 2", "ACS Omega Article")   # same words mid-page are content
kept = pdf.drop_running_lines(made)
for i, page in enumerate(kept):
    assert f"Patient {i + 1}" in page and str(0.5 + i) in page and body[0] in page and f"again {i}" in page, page
    assert f"Page {i + 1}" not in page and str(56651 + i) not in page, page
assert [p.splitlines().count("ACS Omega Article") for p in kept] == [0, 0, 1, 0], kept
assert pdf.drop_running_lines(["Same title\nx", "Same title\ny"]) == ["Same title\nx", "Same title\ny"]
got = pdf.extract(make_pdf(["Acquired resis-", "tance to EGFR-", "TKI therapy."]))
assert got["pages"] == 1 and "resistance to EGFR-" in got["text"], got
assert "resistance" in pdf.extract(make_pdf(["resistance"], password=""))["text"]
for bad, why in ((make_pdf([]), "scanned"), (b"not a pdf", "could not read"), (b"", "could not read"),
                 (make_pdf(["x"], password="secret"), "password")):
    try:
        pdf.extract(bad)
    except ValueError as e:
        assert why in str(e), (why, e)
    else:
        raise AssertionError(f"accepted a PDF that should fail: {why}")

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import labs

# two first requests at once must load a lab a single time
runs, start = [], threading.Barrier(4)


@labs.once
def slow():
    runs.append(1)
    time.sleep(0.1)
    return len(runs)


def hit(fn):
    start.wait()
    return fn()


with ThreadPoolExecutor(4) as pool:
    assert list(pool.map(hit, [slow] * 4)) == [1] * 4 and len(runs) == 1
    assert len({id(x[1].CausalPathScorer) for x in pool.map(hit, [labs._causal] * 4)}) == 1

# a module that fails to load must be retried, not returned half-loaded
with tempfile.TemporaryDirectory() as tmp:
    mod = Path(tmp) / "m.py"
    mod.write_text("raise RuntimeError")
    try:
        labs.load_submodule("pr_test", mod, [])
    except RuntimeError:
        pass
    else:
        raise AssertionError("broken module loaded")
    assert "pr_test.m" not in sys.modules
    mod.write_text("value = 12345")
    assert labs.load_submodule("pr_test", mod, []).value == 12345

rep = labs.repurposing_status()
assert isinstance(rep["missing"], list) and rep["deps_ok"] is True, rep
assert len(labs.repurposing_corpus()) >= 3, "repurposing corpus failed to parse"

# --- New regression tests for RAG fixes, SQLite persistence, multi-provider LLM, and cross-tab workflows ---
import os

# 1. Abbreviation-safe sentences & long unpunctuated block chunking
s_abbr = rag.sentences("Dr. Wang showed Fig. 1 vs. control in the U.S. cohort. Resistance emerged later.")
assert len(s_abbr) == 2 and "Fig. 1 vs. control" in s_abbr[0], s_abbr
long_block = " ".join(f"token{i}" for i in range(320))
long_chunks = rag.chunk_text(long_block, size=100, overlap=20)
assert len(long_chunks) >= 3 and all(len(c.split()) <= 110 for c in long_chunks), len(long_chunks)

# 2. Domain-aware entity extraction (no biomedical suffix leakage in non-bio workspaces)
non_bio_ents = rag.entities("The famine timeline and marine cosine baseline remain outside.", bio=False)
assert not (non_bio_ents & {"famine", "cosine", "marine", "baseline"}), non_bio_ents

# 3. Zero-score BM25 filtering (unrelated chunks must not get phantom BM25 RRF credit)
zws = rag.Workspace("z", "General")
for i in range(4):
    zws.add(f"Filler {i}", f"Ordinary filler paragraph number {i} discussing weather and bridges.")
zws.add("Target", "Xylophonequantum is a unique rare phenomenon observed in acoustics.")
zws.build()
s_bm25 = zws.bm25.scores(rag.tokens("xylophonequantum"))
assert (s_bm25 > 0).sum() == 1, s_bm25
zhits, _, _ = zws.retrieve("xylophonequantum", opts={"hybrid": True, "graph": False, "rerank": False, "decompose": False, "hyde": False, "crag": False}, k=5)
assert zhits[0]["title"] == "Target", zhits

# 4. Polarity-aware & abstention-aware self-reflection
pol_hits = [{"text": "Cisplatin causes severe nephrotoxicity in treated patients.", "label": "relevant"}]
assert ws._reflect("Does cisplatin cause nephrotoxicity?", "Cisplatin does not cause nephrotoxicity [1].", pol_hits)["faithfulness"] == 0.0
assert ws._reflect("What is the dosing?", "The passages do not cover dosing schedules.", pol_hits)["faithfulness"] == 1.0

# 5. Extractive fallback never returns empty on tiny chunks
tiny = rag.Workspace("tiny", "General")
tiny.add("Short", "EGFR+")
tiny_out = tiny.answer("EGFR", opts={"crag": False})
assert tiny_out["answer"].strip(), "extractive answer was empty"

# 6. CRAG corrective fallback (augment_fn)
gap_ws = rag.Workspace("gap", "Oncology")
gap_ws.add("Unrelated", "Baking sourdough bread requires flour, water, salt, and fermentation time.")
called = []

def fake_augment(q):
    called.append(q)
    gap_ws.add("Osimertinib", "Osimertinib inhibits mutant EGFR T790M in non-small cell lung cancer.", "pubmed")
    return 1

aug_hits, aug_trace, aug_crag = gap_ws.retrieve("How does osimertinib target EGFR T790M?", augment_fn=fake_augment)
assert called and aug_hits and "Osimertinib" in aug_hits[0]["title"], (called, aug_hits)
assert aug_crag.get("augmented_papers") == 1, aug_crag

# 7. Multi-turn follow-up rewrite (offline)
mt_ws = rag.Workspace("mt", "Oncology")
mt_ws.add("Cisplatin", "Cisplatin crosslinks DNA and causes nephrotoxicity as a dose-limiting toxicity.")
mt_ws.add("Paclitaxel", "Paclitaxel stabilises microtubules and causes peripheral neuropathy.")
mt_hits, mt_trace, _ = mt_ws.retrieve(
    "What toxicity does it cause?",
    history=[{"role": "user", "content": "Tell me about Cisplatin"}, {"role": "assistant", "content": "Cisplatin crosslinks DNA."}],
)
assert any(t["stage"] == "Follow-up rewrite" for t in mt_trace), mt_trace
assert mt_hits and mt_hits[0]["title"] == "Cisplatin", mt_hits

# 8. RAG -> Causal chain extraction (offline fallback)
c_edges = mt_ws.extract_causal_edges("Cisplatin mechanism", [{"title": "Cisplatin", "text": "Cisplatin crosslinks DNA and ERCC1 overexpression drives Cisplatin resistance in 2019.", "grade": 0.8}])
assert c_edges and c_edges[0]["source"] and c_edges[0]["target"], c_edges

# 9. HyDE passage encoding must not mutate/prune chunk vector cache
if mt_ws.embedder.st:
    before_keys = set(mt_ws._vec_cache)
    saved_llm = rag.llm.available, rag.llm.complete
    rag.llm.available = lambda: True
    rag.llm.complete = lambda *a, **k: "Cisplatin induces DNA damage and renal tubular nephrotoxicity."
    try:
        mt_ws.retrieve("Cisplatin renal effects", opts={"decompose": False, "hyde": True})
    finally:
        rag.llm.available, rag.llm.complete = saved_llm
    assert set(mt_ws._vec_cache) == before_keys, "HyDE mutated chunk cache"

# 10. SQLite persistence, deduplication, preserved doc IDs/dates, and persistent vector cache
with tempfile.TemporaryDirectory() as tmp:
    dbp = Path(tmp) / "test.db"
    st1 = rag.Store(db_path=dbp)
    w1 = rag.Workspace("Persist", "Oncology", disk_cache=st1.disk_cache)
    d1 = w1.add("DocA", "Osimertinib overcomes EGFR T790M resistance in lung adenocarcinoma.")
    d2 = w1.add("DocA copy", "Osimertinib overcomes EGFR T790M resistance in lung adenocarcinoma.")
    assert d1["duplicate"] is False and d2["duplicate"] is True and len(w1.docs) == 1
    w1.build()
    st1.ws[w1.id] = w1
    st1.save()

    st2 = rag.Store(db_path=dbp)
    assert w1.id in st2.ws and d1["id"] in st2.ws[w1.id].docs
    assert st2.ws[w1.id].docs[d1["id"]]["added"] == d1["added"]
    if w1.embedder.st:
        cached = st2.disk_cache.get_many(w1.embedder.name, [rag.text_hash(f"{w1.chunks[0]['ctx']} {w1.chunks[0]['text']}")])
        assert len(cached) == 1, "vector was not persisted to SQLite cache"

# 11. Multi-provider LLM auto-detection
env_keys = ("PANDARAG_PROVIDER", "PANDARAG_MODEL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OLLAMA_MODEL", "GEMINI_API_KEY", "GOOGLE_API_KEY")
saved_env = {k: os.environ.pop(k, None) for k in env_keys}
try:
    assert not rag.llm.available()
    os.environ["OLLAMA_MODEL"] = "llama3.2"
    assert rag.llm.available() and rag.llm.info()["provider"] == "ollama" and rag.llm.info()["model"] == "llama3.2"
    os.environ.pop("OLLAMA_MODEL")
    os.environ["GEMINI_API_KEY"] = "test-key"
    assert rag.llm.available() and rag.llm.info()["provider"] == "gemini"
    os.environ.pop("GEMINI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
    assert rag.llm.available() and rag.llm.info()["provider"] == "openai"
finally:
    for k in env_keys:
        os.environ.pop(k, None)
    for k, v in saved_env.items():
        if v is not None:
            os.environ[k] = v

# 12. FastAPI endpoint smoke tests (compare, causal-extract, SSRF URL block)
import app as api_app
try:
    _ = api_app._safe_fetch_url("http://127.0.0.1:8787/")
    raise AssertionError("SSRF guard failed to block localhost")
except ValueError as e:
    assert "private or local" in str(e), e

wid0 = next(iter(api_app.store.ws))
cmp_res = api_app.compare(wid0, api_app.CompareIn(query="What drives cisplatin resistance?"))
assert "a" in cmp_res and "b" in cmp_res and cmp_res["a"]["sources"], cmp_res

# Batch upload + deduplication smoke test on an isolated workspace
tmp_ws = api_app.create_ws(api_app.WSIn(name="BatchTest", domain="General"))
try:
    b_res = api_app.add_docs_batch(tmp_ws["id"], api_app.BatchDocsIn(docs=[
        api_app.DocIn(title="Paper 1", text="Batch ingestion unique document one for testing.", source="pdf"),
        api_app.DocIn(title="Paper 1 dup", text="Batch ingestion unique document one for testing.", source="pdf"),
        api_app.DocIn(title="Paper 2", text="Batch ingestion unique document two for testing.", source="pdf"),
    ]))
    assert b_res["imported"] == 2 and b_res["skipped"] == 1, b_res
finally:
    api_app.delete_ws(tmp_ws["id"])

print("ok — all core, persistence, multi-provider, CRAG, ablation, batch upload, and lab tests pass")
