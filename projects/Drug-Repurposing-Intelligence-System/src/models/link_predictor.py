"""
Link prediction model combining R-GCN encoder with decoder.
"""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple

from .rgcn_encoder import RGCNEncoder, flatten_hetero


class LinkPredictor(nn.Module):
    """
    Link prediction model for drug-disease relationships.
    """

    def __init__(self, hidden_dim: int = 128):
        """
        Initialize link predictor.

        Parameters
        ----------
        hidden_dim : int
            Hidden dimension size
        """
        super().__init__()

        # Simple MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        z_src: torch.Tensor,
        z_dst: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probability.

        Parameters
        ----------
        z_src : torch.Tensor
            Source node embeddings
        z_dst : torch.Tensor
            Destination node embeddings

        Returns
        -------
        torch.Tensor
            Link prediction scores
        """
        # Concatenate embeddings
        z = torch.cat([z_src, z_dst], dim=-1)

        # Decode
        out = self.decoder(z)

        return out.squeeze(-1)


class DrugRepurposingModel(nn.Module):
    """
    Complete model for drug repurposing prediction.
    """

    def __init__(
        self,
        num_nodes_dict: Dict[str, int],
        num_relations: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_bases: int = 30,
        dropout: float = 0.3,
        graph: Optional[HeteroData] = None
    ):
        """
        Initialize complete model.

        Parameters
        ----------
        graph : HeteroData, optional
            The knowledge graph. When given, the model encodes it with the R-GCN
            and drug/disease representations come from message passing. Without
            it the model falls back to plain per-node embeddings (no structure).
        """
        super().__init__()

        self.graph_mode = graph is not None

        if self.graph_mode:
            edge_index, edge_type, offsets, num_nodes, num_rels = flatten_hetero(graph)
            # Buffers so the graph travels with .to(device). Not persisted: it is
            # rebuilt from the dataset on load, and would otherwise add ~190 MB
            # of edge indices to every checkpoint.
            self.register_buffer('edge_index', edge_index, persistent=False)
            self.register_buffer('edge_type', edge_type, persistent=False)
            self.offsets = offsets
            self.encoder = RGCNEncoder(
                num_nodes=num_nodes, num_relations=num_rels, hidden_dim=hidden_dim,
                num_layers=num_layers, num_bases=num_bases, dropout=dropout
            )
            self._z = None
        else:
            self.compound_embed = nn.Embedding(num_nodes_dict.get('Compound', 1000), hidden_dim)
            self.disease_embed = nn.Embedding(num_nodes_dict.get('Disease', 1000), hidden_dim)

        # Link predictor
        self.link_predictor = LinkPredictor(hidden_dim)

    def encode(self, use_cache: bool = False) -> torch.Tensor:
        """
        Embed every node by running the encoder over the whole graph.

        Message passing costs the same whatever the batch size, so in eval the
        result is cached and reused until :meth:`invalidate_cache` is called.
        """
        if use_cache and self._z is not None:
            return self._z
        z = self.encoder(self.edge_index, self.edge_type)
        if use_cache:
            self._z = z
        return z

    def invalidate_cache(self) -> None:
        """Drop the cached node embeddings (call after the weights change)."""
        self._z = None

    def forward(
        self,
        compound_idx: torch.Tensor,
        disease_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for drug-disease pairs.

        Parameters
        ----------
        compound_idx : torch.Tensor
            Compound node indices
        disease_idx : torch.Tensor
            Disease node indices

        Returns
        -------
        torch.Tensor
            Prediction scores
        """
        if self.graph_mode:
            # Cache only in eval; a training pass moves the weights, so the
            # cached embeddings are dropped as soon as one runs.
            if self.training:
                self.invalidate_cache()
            z = self.encode(use_cache=not self.training)
            z_compound = RGCNEncoder.slice_type(z, self.offsets, 'Compound', compound_idx)
            z_disease = RGCNEncoder.slice_type(z, self.offsets, 'Disease', disease_idx)
        else:
            z_compound = self.compound_embed(compound_idx)
            z_disease = self.disease_embed(disease_idx)

        return self.link_predictor(z_compound, z_disease)
