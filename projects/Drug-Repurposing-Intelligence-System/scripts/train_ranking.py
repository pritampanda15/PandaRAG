#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the drug-repurposing model as a ranking problem.

The BCE setup in train_model.py draws negatives uniformly at random, so a model
can score well by learning which diseases are commonly treated at all. Measured
on Hetionet, that model loses to a control that ignores the drug entirely.

This script fixes the objective:

* negatives are drawn **per drug** — other diseases that this compound does not
  treat — so the only way to lower the loss is to tell diseases apart *for a
  given compound*;
* the loss is listwise (softmax over 1 positive + K negatives), which optimises
  the rank of the true disease instead of a per-pair probability;
* evaluation is the filtered ranking protocol: for a held-out (drug, disease)
  pair, rank every disease the drug is not already known to treat, and report
  where the true one lands — next to the popularity-only control.

Usage:
    python scripts/train_ranking.py --epochs 150 --negatives 32
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.hetionet_loader import HetionetLoader
from src.models.link_predictor import DrugRepurposingModel
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_pairs(loader, seed=42):
    """Known (compound, disease) treats pairs as index tensors, split by pair."""
    pairs = loader.get_drug_disease_pairs()
    torch.manual_seed(seed)
    cmap, dmap = loader.node_mappings['Compound'], loader.node_mappings['Disease']
    c = np.array([cmap[r['drug_id']] for _, r in pairs.iterrows()])
    d = np.array([dmap[r['disease_id']] for _, r in pairs.iterrows()])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(c))
    c, d = c[perm], d[perm]
    n_train, n_val = int(0.7 * len(c)), int(0.85 * len(c))
    splits = {
        'train': (c[:n_train], d[:n_train]),
        'val': (c[n_train:n_val], d[n_train:n_val]),
        'test': (c[n_val:], d[n_val:]),
    }

    # Every known positive, for filtered evaluation and for negative sampling.
    known = defaultdict(set)
    for ci, di in zip(c, d):
        known[int(ci)].add(int(di))
    return splits, known, len(dmap), pairs


def strip_treats(graph):
    """
    Empty the compound-treats-disease relation so it carries no messages.

    Held-out edges would leak the answer into evaluation, and training edges
    would teach the decoder to copy an edge that is absent at test time. The
    relation is emptied rather than deleted so relation ids do not shift.
    """
    et = ('Compound', 'treats', 'Disease')
    graph[et].edge_index = graph[et].edge_index[:, :0]
    return graph


def sample_negatives(c_idx, known_train, num_diseases, k, rng):
    """For each positive, k diseases this compound is not known (in train) to treat."""
    neg = np.empty((len(c_idx), k), dtype=np.int64)
    for i, ci in enumerate(c_idx):
        blocked = known_train.get(int(ci), ())
        drawn = rng.integers(0, num_diseases, size=k * 2)
        keep = [x for x in drawn if x not in blocked][:k]
        while len(keep) < k:                      # rare: tiny candidate pool
            x = int(rng.integers(0, num_diseases))
            if x not in blocked:
                keep.append(x)
        neg[i] = keep
    return torch.from_numpy(neg)


def listwise_loss(model, c, pos_d, neg_d):
    """Softmax cross-entropy over [positive, negatives] for each compound."""
    k = neg_d.shape[1]
    c_rep = c.repeat_interleave(k + 1)
    d_all = torch.cat([pos_d.unsqueeze(1), neg_d], dim=1).reshape(-1)
    logits = model(c_rep, d_all).view(-1, k + 1)
    target = torch.zeros(len(c), dtype=torch.long)   # column 0 is the positive
    return F.cross_entropy(logits, target)


@torch.no_grad()
def rank_eval(model, split, known, num_diseases, popularity=None):
    """Filtered ranking: where does the true disease land among all candidates?"""
    model.eval()
    c_idx, d_idx = split
    all_d = torch.arange(num_diseases)
    ranks, pop_ranks = [], []
    for ci, di in zip(c_idx, d_idx):
        scores = model(torch.full((num_diseases,), int(ci), dtype=torch.long), all_d).numpy()
        # Filtered: other known positives of this drug must not outrank the target.
        for other in known.get(int(ci), ()):
            if other != int(di):
                scores[other] = -np.inf
        ranks.append(int((scores > scores[int(di)]).sum()) + 1)
        if popularity is not None:
            pop = popularity.copy()
            for other in known.get(int(ci), ()):
                if other != int(di):
                    pop[other] = -np.inf
            pop_ranks.append(int((pop > pop[int(di)]).sum()) + 1)

    def summarise(rs):
        rs = np.asarray(rs, dtype=float)
        return {'median_rank': float(np.median(rs)), 'mean_rank': round(float(rs.mean()), 1),
                'hits_at_1': round(float((rs <= 1).mean()), 3),
                'hits_at_10': round(float((rs <= 10).mean()), 3),
                'mrr': round(float((1 / rs).mean()), 3)}

    out = {'pairs': len(ranks), 'model': summarise(ranks)}
    if popularity is not None:
        out['popularity_control'] = summarise(pop_ranks)
        out['beats_popularity'] = out['model']['mrr'] > out['popularity_control']['mrr']
    return out


def main():
    parser = argparse.ArgumentParser(description="Train drug repurposing model as ranking")
    parser.add_argument('--config', type=str, default='config/model_config.yaml')
    parser.add_argument('--data-dir', type=str, default='data/raw/hetionet')
    parser.add_argument('--output', type=str, default='models/checkpoints/ranking_noleak.pt')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--negatives', type=int, default=32, help='Negatives per positive')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-graph', action='store_true',
                        help='Embedding-only baseline (no R-GCN encoder)')
    args = parser.parse_args()

    setup_logging(level="INFO", log_file="logs/training/train_ranking.log")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    loader = HetionetLoader(data_dir=args.data_dir)
    loader.load()
    metadata = loader.get_metadata()
    splits, known, num_diseases, pairs = build_pairs(loader, args.seed)
    strip_treats(loader.graph)

    # Negatives may only avoid training positives; val/test edges stay unseen.
    known_train = defaultdict(set)
    for ci, di in zip(*splits['train']):
        known_train[int(ci)].add(int(di))

    logger.info(f"Train {len(splits['train'][0])} / val {len(splits['val'][0])} / "
                f"test {len(splits['test'][0])} pairs over {num_diseases} diseases")

    model = DrugRepurposingModel(
        num_nodes_dict=dict(metadata['node_types']),
        num_relations=metadata['num_edge_types'],
        hidden_dim=int(config['model']['encoder']['hidden_dim']),
        num_layers=int(config['model']['encoder']['num_layers']),
        num_bases=int(config['model']['encoder'].get('num_bases', 30)),
        dropout=float(config['model']['encoder'].get('dropout', 0.3)),
        graph=None if args.no_graph else loader.graph,
    )
    logger.info("Encoder: %s", "R-GCN over the graph minus treats edges" if model.graph_mode
                else "per-node embeddings (no structure)")

    # Popularity control: how often each disease is treated, in the training split only.
    popularity = np.zeros(num_diseases)
    for di in splits['train'][1]:
        popularity[int(di)] += 1

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=float(config['training']['weight_decay']))
    rng = np.random.default_rng(args.seed)
    c_train = torch.from_numpy(splits['train'][0])
    d_train = torch.from_numpy(splits['train'][1])

    best_mrr, best_epoch = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        neg = sample_negatives(splits['train'][0], known_train, num_diseases, args.negatives, rng)
        opt.zero_grad()
        loss = listwise_loss(model, c_train, d_train, neg)
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            val = rank_eval(model, splits['val'], known, num_diseases)
            logger.info(f"Epoch {epoch}/{args.epochs}  loss {loss.item():.4f}  "
                        f"val MRR {val['model']['mrr']:.3f}  "
                        f"hits@10 {val['model']['hits_at_10']:.3f}  "
                        f"median rank {val['model']['median_rank']:.0f}")
            if val['model']['mrr'] > best_mrr:
                best_mrr, best_epoch = val['model']['mrr'], epoch
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                torch.save({'model_state_dict': model.state_dict()}, args.output)
                logger.info(f"  ✓ new best (val MRR {best_mrr:.3f}) saved")

    # Report the best checkpoint, not the last one.
    model.load_state_dict(torch.load(args.output, map_location='cpu')['model_state_dict'])
    model.invalidate_cache()
    test = rank_eval(model, splits['test'], known, num_diseases, popularity=popularity)
    logger.info(f"TEST  model {test['model']}")
    logger.info(f"TEST  popularity {test['popularity_control']}")
    logger.info(f"TEST  beats popularity: {test['beats_popularity']}")

    metrics = {
        'objective': 'listwise ranking, per-drug negatives',
        'encoder': 'rgcn' if model.graph_mode else 'embeddings',
        'message_graph': 'treats edges removed',
        'negatives_per_positive': args.negatives,
        'epochs': args.epochs, 'best_epoch': best_epoch, 'learning_rate': args.lr,
        'parameters': sum(p.numel() for p in model.parameters()),
        'train_pairs': len(splits['train'][0]), 'test_pairs': len(splits['test'][0]),
        'diseases': num_diseases,
        'held_out_ranking': test,
    }
    Path(args.output).with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2))
    logger.info(f"Metrics written to {Path(args.output).with_suffix('.metrics.json')}")


if __name__ == "__main__":
    main()
