"""Lab modules: thin wrappers over the six research projects bundled in projects/.

Each wrapper loads project modules by file path (the projects aren't installed packages and
some have heavy optional imports in their __init__), and reports status so the UI can
show what is live vs. unavailable on this machine.
"""
import ast
import functools
from collections import defaultdict
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "projects"
REPOS = {
    "admet": ROOT / "ADMET-Prediction-System-Graph-Neural-Networks-with-RAG",
    "repurposing": ROOT / "Drug-Repurposing-Intelligence-System",
    "literature": ROOT / "Literature-Intelligence",
    "oncology": ROOT / "Oncology-Drug-Response-Prediction-System_DL-RAG",
    "oncomolml": ROOT / "OncoMolML",
    "causal": ROOT / "TC-DrugRAG",
}


def once(fn):
    """lru_cache runs fn twice if two threads miss together; a second import then sees a half-loaded
    module, and _repurposing's chdir can be left unrestored. Failures are not cached, so they retry."""
    cached, lock = functools.lru_cache(maxsize=1)(fn), threading.Lock()

    @functools.wraps(fn)
    def wrapper():
        with lock:
            return cached()
    return wrapper


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def namespace(name, path):
    """Register a directory as a bare package (its __init__ is NOT run, so heavy optional
    imports there stay out of the way) and return it, so relative imports inside it resolve."""
    if name in sys.modules:
        return sys.modules[name]
    import types
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


PKG_ROOTS = {"tcdr": "causal", "dri": "repurposing"}


def load_submodule(pkg, path, subdirs):
    """Import `path` as a submodule of a bare package tree rooted at `pkg`.

    `subdirs` are the package dirs between the repo's `src/` and the file, so a
    module's relative imports (`from ..models.x import y`) resolve as usual.
    """
    root = REPOS[PKG_ROOTS[pkg]] / "src" if pkg in PKG_ROOTS else Path(path).parent
    namespace(pkg, root)
    for sub in subdirs:
        namespace(f"{pkg}.{sub}", root / sub)
    name = f"{pkg}.{subdirs[-1]}.{Path(path).stem}" if subdirs else f"{pkg}.{Path(path).stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[name]   # or the next call would return it half-loaded
        raise
    return mod


# ---------------------------------------------------------------- ADMET (live model)
@once
def _admet():
    import torch
    r = REPOS["admet"]
    F = load_module("pr_featurizer", r / "src/data/molecular_featurizer.py")
    G = load_module("pr_gnn", r / "src/models/gnn_model.py")
    feat = F.MolecularFeaturizer()
    model = G.GNNPredictor(input_dim=feat.atom_feature_dim, hidden_dim=128, output_dim=1,
                           num_layers=3, dropout=0.2, architecture="GCN", pooling="mean",
                           edge_dim=feat.bond_feature_dim)
    ckpt_path = r / "models/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return feat, model, ckpt.get("val_metrics", {}), ckpt.get("epoch")


def admet_predict(smiles: str):
    """Caco-2 permeability from the trained GCN, plus RDKit descriptors and rule-of-five style flags."""
    import torch
    from torch_geometric.data import Batch
    feat, model, val, epoch = _admet()
    graph = feat.smiles_to_graph(smiles)
    if graph is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    with torch.no_grad():
        pred = float(model(Batch.from_data_list([graph])).item())
    d = feat.get_molecular_descriptors(smiles) or {}
    flags = []
    if d.get("molecular_weight", 0) > 500: flags.append("MW > 500")
    if d.get("logp", 0) > 5: flags.append("logP > 5")
    if d.get("num_h_donors", 0) > 5: flags.append("HBD > 5")
    if d.get("num_h_acceptors", 0) > 10: flags.append("HBA > 10")
    if d.get("tpsa", 0) > 140: flags.append("TPSA > 140")
    if d.get("num_rotatable_bonds", 0) > 10: flags.append("RotB > 10")
    return {
        "smiles": smiles,
        "endpoint": "Caco-2 permeability (Caco2_Wang)",
        "unit": "log Papp (cm/s)",
        "prediction": round(pred, 3),
        "permeability": "high" if pred > -5.0 else "moderate" if pred > -6.0 else "low",
        "descriptors": {k: round(v, 3) for k, v in d.items()},
        "lipinski_violations": flags,
        "graph": {"atoms": int(graph.num_nodes), "bonds": int(graph.num_edges // 2)},
        "model": {"architecture": "GCN · 3 layers · 128 hidden", "trained_epoch": epoch,
                  "val_metrics": {k: round(float(v), 4) for k, v in (val or {}).items()}},
    }


# ---------------------------------------------------------------- Literature (live APIs)
@once
def _clients():
    r = REPOS["literature"]
    pm = load_module("pr_pubmed", r / "src/clients/pubmed_client.py")
    bx = load_module("pr_biorxiv", r / "src/clients/biorxiv_client.py")
    email = os.getenv("PANDARAG_NCBI_EMAIL", "pandarag@example.com")
    return pm.PubMedClient(email=email, api_key=os.getenv("NCBI_API_KEY")), bx.BioRxivClient()


def literature_search(query: str, sources=("pubmed",), max_results=12, days_back=None):
    pubmed, biorxiv = _clients()
    papers, errors = [], {}
    for s in sources:
        try:
            client = pubmed if s == "pubmed" else biorxiv
            papers += client.search(query=query, max_results=max_results, days_back=days_back) or []
        except Exception as e:
            errors[s] = f"{type(e).__name__}: {e}"
    seen, unique = set(), []
    for p in papers:
        key = (p.get("title") or "").lower()[:90]
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    unique.sort(key=lambda p: p.get("publication_date", ""), reverse=True)
    return {"papers": unique, "errors": errors}


# ---------------------------------------------------------------- Causal KG (TC-DrugRAG)
@once
def _causal():
    """TC-DrugRAG schemas + path scorer, imported with their relative imports intact."""
    root = REPOS["causal"] / "src"
    schemas = load_submodule("tcdr", root / "knowledge_graph/schemas.py", ["knowledge_graph"])
    scorer = load_submodule("tcdr", root / "retrieval/path_scorer.py", ["retrieval"])
    return schemas, scorer


def causal_schema():
    """Expose TC-DrugRAG's causal/evidence vocabulary and the path-scoring weights."""
    s, scorer = _causal()
    return {
        "relations": [e.value for e in s.CausalRelationType],
        "node_types": [e.value for e in s.NodeType],
        "evidence": [{"type": k.value, "weight": v} for k, v in
                     sorted(scorer.CausalPathScorer.EVIDENCE_TYPE_WEIGHTS.items(), key=lambda kv: -kv[1])],
        "weights": {"evidence": 0.35, "causal_depth": 0.25, "temporal": 0.20, "diversity": 0.20},
    }


def score_causal_path(edges):
    """Score a user-drawn causal chain with TC-DrugRAG's formula.

    edges: [{source, relation, target, evidence, year, confidence}]
    """
    s, scorer = _causal()
    W = {e.value: w for e, w in scorer.CausalPathScorer.EVIDENCE_TYPE_WEIGHTS.items()}
    if not edges:
        raise ValueError("path needs at least one edge")
    ev = sum(W.get(e.get("evidence", "literature_mining"), 0.5) * float(e.get("confidence", 0.8)) for e in edges) / len(edges)
    depth = 1.0 / (1.0 + 0.4 * (len(edges) - 1))
    years = [int(e["year"]) for e in edges if str(e.get("year", "")).isdigit()]
    temporal = 1.0 if years == sorted(years) and years else 0.5
    diversity = len({e.get("evidence") for e in edges}) / len(edges)
    total = 0.35 * ev + 0.25 * depth + 0.20 * temporal + 0.20 * diversity
    parts = []
    for i, e in enumerate(edges):
        rel = e.get("relation", "LEADS_TO")
        if i == 0 or e["source"] != edges[i - 1]["target"]:
            parts.append(f"{e['source']} —[{rel}]→ {e['target']}")
        else:
            parts[-1] += f" —[{rel}]→ {e['target']}"
    return {"score": round(total, 3),
            "components": {"evidence": round(ev, 3), "causal_depth": round(depth, 3),
                           "temporal": round(temporal, 3), "diversity": round(diversity, 3)},
            "hops": len(edges),
            "chain": " · ".join(parts),
            "relations_known": [e.get("relation") in {r.value for r in s.CausalRelationType} for e in edges]}


# ---------------------------------------------------------------- Screening (OncoMolML, JAX)
@once
def _jax_models():
    return load_module("pr_jax", REPOS["oncomolml"] / "oncomolml/models/jax_models.py")


def _fingerprints(smiles_list, nbits=1024, radius=2):
    """RDKit graph (ADMET featurizer) -> OncoMolML's JAX Morgan fingerprint."""
    import jax.numpy as jnp
    import numpy as np
    J = _jax_models()
    feat, *_ = _admet()
    fps, kept, failed = [], [], []
    for s in smiles_list:
        g = feat.smiles_to_graph(s)
        if g is None:
            failed.append(s)
            continue
        x = jnp.asarray(np.asarray(g.x, dtype="float32"))
        n = int(x.shape[0])
        adj = np.zeros((n, n), dtype="float32")
        ei = np.asarray(g.edge_index)
        if ei.size:
            adj[ei[0], ei[1]] = 1.0
        fps.append(J.morgan_fingerprint_jax(x, jnp.asarray(adj), radius=radius, nbits=nbits))
        kept.append({"smiles": s, "atoms": n, "bonds": int(g.num_edges // 2)})
    if not fps:
        raise ValueError("no valid SMILES — RDKit could not parse any input")
    return jnp.stack(fps), kept, failed


def screen(smiles_list, n_select=0, nbits=1024, radius=2):
    """Fingerprint a library, compute the Tanimoto matrix, and pick a diverse subset."""
    import numpy as np
    J = _jax_models()
    t = time.perf_counter()
    F, kept, failed = _fingerprints(smiles_list, nbits, radius)
    sim = np.asarray(J.batch_tanimoto_matrix(F))
    n = len(kept)
    for i, m in enumerate(kept):
        m["bits_set"] = int(np.asarray(F[i]).sum())
        others = [sim[i][j] for j in range(n) if j != i]
        m["max_similarity"] = round(float(max(others)), 3) if others else 0.0
        m["mean_similarity"] = round(float(np.mean(others)), 3) if others else 0.0
    picked = []
    k = max(0, min(int(n_select), n))
    if k:
        # diversity_selection is @jit-decorated upstream but loops on n_select, which jit
        # cannot trace; __wrapped__ is the same MaxMin routine without the decorator.
        picked = [int(i) for i in np.asarray(J.diversity_selection.__wrapped__(F, n_select=k))]
    pairs = sorted(({"a": kept[i]["smiles"], "b": kept[j]["smiles"], "similarity": round(float(sim[i][j]), 3)}
                    for i in range(n) for j in range(i + 1, n)),
                   key=lambda p: -p["similarity"])[:8]
    return {"molecules": kept, "failed": failed, "matrix": [[round(float(v), 3) for v in row] for row in sim],
            "selected": picked, "nearest_pairs": pairs, "nbits": nbits, "radius": radius,
            "backend": f"JAX {__import__('jax').__version__} · {__import__('jax').devices()[0].platform.upper()}",
            "ms": int((time.perf_counter() - t) * 1000)}


# ---------------------------------------------------------------- Repurposing (Hetionet R-GCN)
def repurposing_status():
    """What the R-GCN link predictor still needs before it can score drug-disease pairs."""
    r = REPOS["repurposing"]
    loader = r / "src/data/hetionet_loader.py"
    url = ""
    for line in loader.read_text(errors="replace").splitlines():
        if "HETIONET_URL" in line and "http" in line:
            url = line.split('"')[1] if '"' in line else ""
            break
    graph_files = [p.name for p in (r / "data").rglob("*") if p.is_file()] if (r / "data").exists() else []
    ckpts = [str(p.relative_to(r)) for p in r.rglob("*.pt")] + [str(p.relative_to(r)) for p in r.rglob("*.pth")]
    missing = []
    if not graph_files:
        missing.append({"what": "Hetionet graph", "fix": "python scripts/download_data.py", "note": url or "hetio/hetionet v1.0"})
    if not ckpts:
        missing.append({"what": "trained R-GCN weights", "fix": "python scripts/train_ranking.py", "note": "listwise ranking loss, 150 epochs"})
    return {"repo": r.name, "architecture": "R-GCN encoder (3 layers, 128d, 30 bases) + MLP decoder",
            "deps_ok": importlib.util.find_spec("torch_geometric") is not None,
            "missing": missing, "checkpoints": ckpts, "data_files": len(graph_files)}


@once
def _repurposing():
    """Hetionet loader + the trained link-prediction checkpoint."""
    import torch
    r = REPOS["repurposing"]
    cwd = os.getcwd()
    os.chdir(r)  # the loader resolves its cache relative to the repo root
    try:
        loader_mod = load_submodule("dri", r / "src/data/hetionet_loader.py", ["data"])
        model_mod = load_submodule("dri", r / "src/models/link_predictor.py", ["models"])
        loader = loader_mod.HetionetLoader(data_dir="data/raw/hetionet")
        loader.load()
        meta = loader.get_metadata()
        # ranking_model.pt and rgcn_model.pt are skipped: both were trained and scored with
        # held-out treats edges inside the message-passing graph.
        ckpt_path = next((p for p in (r / "models/checkpoints/ranking_noleak.pt",
                                      r / "models/checkpoints/ranking_nograph.pt",
                                      r / "models/checkpoints/best_model.pt") if p.exists()), None)
        if ckpt_path is None:
            raise FileNotFoundError("no trained checkpoint in models/checkpoints")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        graph_mode = any(k.startswith("encoder.") for k in state)
        if graph_mode:  # match training: no treats edge may carry messages
            et = ("Compound", "treats", "Disease")
            loader.graph[et].edge_index = loader.graph[et].edge_index[:, :0]
        model = model_mod.DrugRepurposingModel(
            num_nodes_dict=dict(meta["node_types"]), num_relations=meta["num_edge_types"],
            hidden_dim=128, num_layers=3, num_bases=30,
            graph=loader.graph if graph_mode else None)
        model.load_state_dict(state)
        model.eval()
        model.checkpoint_name = ckpt_path.name
    finally:
        os.chdir(cwd)
    pairs = loader.get_drug_disease_pairs()
    known = {(p["drug_id"], p["disease_id"]) for _, p in pairs.iterrows()}
    drugs = {n["name"]: n["id"] for n in loader.nodes_by_type["Compound"]}
    diseases = {n["name"]: n["id"] for n in loader.nodes_by_type["Disease"]}
    return loader, model, meta, known, drugs, diseases, pairs


def repurposing_vocab(q="", kind="drug", limit=25):
    _, _, _, _, drugs, diseases, _ = _repurposing()
    names = sorted(drugs if kind == "drug" else diseases)
    q = q.strip().lower()
    hits = [n for n in names if q in n.lower()] if q else names
    return {"total": len(names), "matches": hits[:limit]}


def repurposing_predict(drug_name, disease_name=None, top_k=10):
    """Score one compound against one disease, or rank every disease for that compound."""
    import torch
    loader, model, _, known, drugs, diseases, _ = _repurposing()
    if drug_name not in drugs:
        raise ValueError(f"unknown compound: {drug_name!r}")
    cmap = loader.node_mappings["Compound"]
    dmap = loader.node_mappings["Disease"]
    ci = cmap[drugs[drug_name]]
    targets = [disease_name] if disease_name else list(diseases)
    idx = [dmap[diseases[d]] for d in targets if d in diseases]
    names = [d for d in targets if d in diseases]
    if not idx:
        raise ValueError(f"unknown disease: {disease_name!r}")
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.full((len(idx),), ci, dtype=torch.long),
                                     torch.tensor(idx, dtype=torch.long))).tolist()
    rows = [{"disease": n, "score": round(float(s), 4),
             "known": (drugs[drug_name], diseases[n]) in known}
            for n, s in zip(names, scores)]
    rows.sort(key=lambda r: -r["score"])
    return {"drug": drug_name, "results": rows[:top_k],
            "known_indications": sorted(r["disease"] for r in rows if r["known"])[:12]}


def _ranking_check(loader, model, known, drugs, diseases, pairs):
    """Score the task the tab performs: rank all 137 diseases for a drug.

    Held-out AUC is measured against *random* negatives, which a model can win by
    learning which diseases are common. So we also rank every disease for every
    drug with a known indication, and compare against a popularity-only control
    (rank diseases by how many treats edges they have). If the model cannot beat
    that control, its scores carry no drug-specific signal.
    """
    import numpy as np
    import torch
    cmap, dmap = loader.node_mappings["Compound"], loader.node_mappings["Disease"]
    names = list(diseases)
    pos = {name: i for i, name in enumerate(names)}
    idx = torch.tensor([dmap[diseases[d]] for d in names])
    truth = defaultdict(set)
    for _, p in pairs.iterrows():
        truth[p["drug_name"]].add(p["disease_name"])

    ranks = []
    with torch.no_grad():
        for drug, targets in truth.items():
            if drug not in drugs:
                continue
            s = model(torch.full((len(idx),), cmap[drugs[drug]], dtype=torch.long), idx).numpy()
            order = np.argsort(-s)
            place = {int(o): r + 1 for r, o in enumerate(order)}
            ranks += [place[pos[t]] for t in targets if t in pos]

    counts = pairs["disease_name"].value_counts()
    pop_order = sorted(names, key=lambda d: -counts.get(d, 0))
    pop_ranks = [pop_order.index(t) + 1 for targets in truth.values() for t in targets if t in pos]

    def summarise(rs):
        rs = np.array(rs, float)
        return {"median_rank": int(np.median(rs)), "hits_at_10": round(float((rs <= 10).mean()), 3),
                "mrr": round(float((1 / rs).mean()), 3)}

    model_m, pop_m = summarise(ranks), summarise(pop_ranks)
    return {"diseases": len(names), "pairs": len(ranks), "model": model_m,
            "popularity_control": pop_m,
            "random_control": {"median_rank": len(names) // 2,
                               "hits_at_10": round(10 / len(names), 3),
                               "mrr": round(float(np.mean([1 / r for r in range(1, len(names) + 1)])), 3)},
            "beats_popularity": model_m["mrr"] > pop_m["mrr"]}


def _run_metrics(name):
    """Metrics a training run wrote beside its checkpoint, if present."""
    import json
    p = REPOS["repurposing"] / "models/checkpoints" / name
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def repurposing_model_card():
    """Honest model card: what was trained, on how much, and how well it separates held-out pairs."""
    import numpy as np
    import torch
    loader, model, meta, known, drugs, diseases, pairs = _repurposing()
    cmap, dmap = loader.node_mappings["Compound"], loader.node_mappings["Disease"]
    pos = [(cmap[p["drug_id"]], dmap[p["disease_id"]]) for _, p in pairs.iterrows()]
    rng = np.random.default_rng(0)
    neg, seen = [], {(a, b) for a, b in pos}
    while len(neg) < len(pos) * 5:
        c, d = int(rng.integers(len(cmap))), int(rng.integers(len(dmap)))
        if (c, d) not in seen:
            neg.append((c, d))
    allp = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor([a for a, _ in allp]), torch.tensor([b for _, b in allp]))).numpy()
    order = np.argsort(-s)
    yy = y[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    auc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
    graph_mode = getattr(model, "graph_mode", False)
    ckpt = getattr(model, "checkpoint_name", "best_model.pt")
    run = _run_metrics(ckpt.replace(".pt", ".metrics.json")) or {}
    baseline = _run_metrics("baseline_model.metrics.json")
    # A ranking run reports filtered held-out ranks; prefer those over the live
    # all-pairs check, which includes pairs the model trained on.
    held_out_ranking = run.get("held_out_ranking")
    card = {
        "checkpoint": getattr(model, "checkpoint_name", "best_model.pt"),
        "architecture": ("R-GCN encoder (basis decomposition, 48 relations incl. inverses) over all "
                         "47,031 nodes → MLP link decoder") if graph_mode
                        else "Compound + Disease embeddings (128d) → MLP link predictor",
        "graph_mode": graph_mode,
        "objective": run.get("objective", "binary classification, random negatives"),
        "caveat": ("Drug and disease vectors now come from message passing over Hetionet minus treats edges, "
                   "so a score reflects a compound's neighbourhood — its targets, side effects and "
                   "pharmacologic class.") if graph_mode
                  else ("This checkpoint looks up per-node embeddings and never runs the R-GCN encoder, so "
                        "scores reflect the 755 treats edges it memorised, not graph-based inference."),
        "parameters": sum(p.numel() for p in model.parameters()),
        "graph": {"nodes": meta["total_nodes"], "edges": meta["total_edges"],
                  "compounds": len(cmap), "diseases": len(dmap),
                  "message_passing_edges": int(model.edge_index.shape[1]) if graph_mode else 0,
                  "relations": int(model.edge_type.max()) + 1 if graph_mode else 0},
        "training": {"positive_pairs": run.get("train_pairs", len(pos)),
                     "negatives_per_positive": run.get("negatives_per_positive", 5),
                     "epochs": run.get("epochs"), "best_epoch": run.get("best_epoch"),
                     "test_loss": run.get("test_loss")},
        "held_out": {"auc": run.get("test_auc"), "ap": run.get("test_ap"),
                     "test_size": run.get("test_size")},
        "baseline_held_out": {"auc": (baseline or {}).get("test_auc"), "ap": (baseline or {}).get("test_ap")}
        if baseline else None,
        "auc_on_training_distribution": round(auc, 3),
        "ranking": ({**held_out_ranking, "diseases": run.get("diseases", len(dmap)), "filtered": True}
                    if held_out_ranking else
                    _ranking_check(loader, model, known, drugs, diseases, pairs)),
    }
    if held_out_ranking and not held_out_ranking["beats_popularity"]:
        card["caveat"] = (
            "Evaluated with every treats edge removed from the message-passing graph. On held-out pairs "
            "the model does not beat a popularity-only control that ignores the drug, so its scores "
            "carry no demonstrated drug-specific signal.")
    elif not card["ranking"]["beats_popularity"]:
        card["caveat"] = (
            "Held-out AUC is high, but on the task this tab performs — ranking all "
            f"{card['ranking']['diseases']} diseases for one drug — the model does not beat a "
            "popularity-only control that ignores the drug entirely. Training negatives are drawn "
            "at random, which teaches 'is this disease commonly treated' rather than 'does this "
            "drug treat it'. Read the ranking below, not the AUC.")
    elif held_out_ranking:
        card["caveat"] = (
            "Trained as a ranking problem: negatives are other diseases the same compound does not "
            "treat, and the loss optimises the rank of the true disease. On held-out pairs the true "
            f"indication lands at median rank {held_out_ranking['model']['median_rank']:g} of "
            f"{card['ranking']['diseases']}, ahead of the popularity-only control "
            f"({held_out_ranking['popularity_control']['median_rank']:.0f}). No treats edge is visible to "
            "the encoder, so this is not edge lookup. A high-scoring "
            "unseen pair is a hypothesis worth checking, not a result.")
    return card


def repurposing_corpus():
    """The drug-mechanism documents the repo ships in scripts/index_documents.py."""
    src = (REPOS["repurposing"] / "scripts/index_documents.py").read_text(errors="replace")
    docs, meta = [], []
    for name, target in (("sample_docs", docs), ("sample_metadata", meta)):
        start = src.find(f"{name} = [")
        if start == -1:
            continue
        depth, i = 0, src.index("[", start)
        for j in range(i, len(src)):
            depth += (src[j] == "[") - (src[j] == "]")
            if depth == 0:
                target.extend(ast.literal_eval(src[i:j + 1]))
                break
    out = []
    for i, text in enumerate(docs):
        m = meta[i] if i < len(meta) else {}
        title = m.get("drug_name") or m.get("drug_id") or f"Mechanism note {i + 1}"
        out.append({"title": str(title), "text": str(text).strip(), "source": "repurposing-kb"})
    return out


# ---------------------------------------------------------------- Oncology knowledge base
def oncology_corpus():
    d = REPOS["oncology"] / "knowledge_base/documents"
    return [{"title": p.stem.title(), "text": p.read_text(errors="replace"), "source": "oncology-kb"}
            for p in sorted(d.glob("*.txt"))]


# ---------------------------------------------------------------- Status board
def status():
    out = []
    checks = [
        ("ADMET GNN", "admet", "Caco-2 permeability from a trained 3-layer GCN", lambda: _admet() and "live"),
        ("Literature", "literature", "PubMed + bioRxiv retrieval clients", lambda: _clients() and "live"),
        ("Causal KG", "causal", "TC-DrugRAG causal path scoring", lambda: causal_schema() and "live"),
        ("Oncology KB", "oncology", "Curated drug-response corpus", lambda: "live" if oncology_corpus() else "empty"),
        ("Screening", "oncomolml", "JAX Morgan fingerprints, Tanimoto, MaxMin diversity",
         lambda: "live" if screen(["CCO", "CCN"], 1)["molecules"] else "empty"),
        ("Repurposing", "repurposing", "R-GCN link prediction over Hetionet",
         lambda: "live" if not repurposing_status()["missing"]
         else "needs: " + ", ".join(m["what"] for m in repurposing_status()["missing"])),
    ]
    for name, key, desc, probe in checks:
        try:
            state = probe()
        except Exception as e:
            state = f"unavailable: {type(e).__name__}"
        out.append({"name": name, "repo": REPOS[key].name, "description": desc, "state": state})
    return out
