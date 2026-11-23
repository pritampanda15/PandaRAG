"""
Relational Graph Convolutional Network (R-GCN) Encoder

Implementation of R-GCN for heterogeneous graphs.

The heterogeneous graph is flattened into one homogeneous node index space
(each node type occupies a contiguous block) plus a relation id per edge, which
is the form ``RGCNConv`` consumes. ``flatten_hetero`` builds that view once; the
encoder then runs full-graph message passing and returns embeddings for every
node, which callers slice per node type via :meth:`slice_type`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple


def flatten_hetero(
    graph: HeteroData,
    add_inverse: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int], int, int]:
    """
    Flatten a HeteroData graph into (edge_index, edge_type) over one index space.

    Parameters
    ----------
    graph : HeteroData
        Heterogeneous graph.
    add_inverse : bool
        Also emit the reverse of every edge as its own relation, so messages
        flow in both directions (Hetionet stores each edge once).

    Returns
    -------
    edge_index : torch.Tensor
        [2, E] global source/target indices.
    edge_type : torch.Tensor
        [E] relation id per edge.
    offsets : Dict[str, int]
        First global index of each node type.
    num_nodes : int
        Total node count.
    num_relations : int
        Relation count (doubled when ``add_inverse``).
    """
    offsets, cursor = {}, 0
    for node_type in graph.node_types:
        offsets[node_type] = cursor
        cursor += graph[node_type].num_nodes

    src_all, dst_all, type_all = [], [], []
    relations = list(graph.edge_types)
    for rel_id, edge_type in enumerate(relations):
        src_type, _, dst_type = edge_type
        edge_index = graph[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        src = edge_index[0] + offsets[src_type]
        dst = edge_index[1] + offsets[dst_type]
        src_all.append(src)
        dst_all.append(dst)
        type_all.append(torch.full((src.numel(),), rel_id, dtype=torch.long))
        if add_inverse:
            src_all.append(dst)
            dst_all.append(src)
            type_all.append(torch.full((src.numel(),), rel_id + len(relations), dtype=torch.long))

    edge_index = torch.stack([torch.cat(src_all), torch.cat(dst_all)])
    edge_type = torch.cat(type_all)
    num_relations = len(relations) * (2 if add_inverse else 1)
    return edge_index, edge_type, offsets, cursor, num_relations


class RGCNEncoder(nn.Module):
    """
    R-GCN encoder for heterogeneous biomedical knowledge graphs.

    Uses relation-specific transformations (basis decomposition) so every edge
    type gets its own propagation while sharing a small set of basis matrices.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_bases: int = 16,
        dropout: float = 0.3
    ):
        """
        Initialize R-GCN encoder.

        Parameters
        ----------
        num_nodes : int
            Total number of nodes in the flattened graph.
        num_relations : int
            Number of relation types (inverses included).
        hidden_dim : int
            Hidden dimension size.
        num_layers : int
            Number of R-GCN layers.
        num_bases : int
            Number of bases for basis decomposition.
        dropout : float
            Dropout rate.
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Layer-0 features: one learned vector per node.
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.normal_(self.node_embedding.weight, std=0.1)

        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations, num_bases=num_bases)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Run relational message passing over the whole graph.

        Parameters
        ----------
        edge_index : torch.Tensor
            [2, E] global edge indices.
        edge_type : torch.Tensor
            [E] relation id per edge.

        Returns
        -------
        torch.Tensor
            [num_nodes, hidden_dim] node embeddings.
        """
        x = self.node_embedding.weight

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    @staticmethod
    def slice_type(
        z: torch.Tensor,
        offsets: Dict[str, int],
        node_type: str,
        idx: torch.Tensor
    ) -> torch.Tensor:
        """Select rows of ``z`` for type-local indices of ``node_type``."""
        return z[idx + offsets[node_type]]
