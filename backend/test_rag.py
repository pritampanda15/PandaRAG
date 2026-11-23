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

sc = labs.screen(["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "notasmiles"], n_select=2)
assert sc["failed"] == ["notasmiles"], sc["failed"]
assert len(sc["molecules"]) == 3 and len(sc["matrix"]) == 3, sc["molecules"]
assert all(abs(sc["matrix"][i][i] - 1.0) < 1e-3 for i in range(3)), "self-similarity should be 1"
assert sc["matrix"][0][1] == sc["matrix"][1][0], "similarity matrix must be symmetric"
assert len(set(sc["selected"])) == 2, sc["selected"]
# the two NSAIDs are more alike than either is to caffeine
assert sc["matrix"][0][1] > sc["matrix"][0][2], sc["matrix"]

rep = labs.repurposing_status()
assert isinstance(rep["missing"], list) and rep["deps_ok"] is True, rep
assert len(labs.repurposing_corpus()) >= 3, "repurposing corpus failed to parse"

print("ok — retrieval, generation, graph, empty-workspace, removal, screening and repurposing paths all pass")
