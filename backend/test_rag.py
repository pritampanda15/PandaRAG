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

import labs

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
