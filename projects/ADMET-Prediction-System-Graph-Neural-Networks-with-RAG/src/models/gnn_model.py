"""
Graph Neural Network architectures for molecular property prediction.

Implements GCN, GAT, and MPNN variants with configurable depth
and pooling strategies.
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    NNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.data import Data, Batch


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer with batch normalization.
    
    Implements the propagation rule from Kipf & Welling (2017).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2
    ):
        """
        Initialize GCN layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].
            
        Returns:
            Updated node features [num_nodes, out_channels].
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GATLayer(nn.Module):
    """
    Graph Attention Network layer.
    
    Uses multi-head attention mechanism from Velickovic et al. (2018).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2
    ):
        """
        Initialize GAT layer.
        
        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension per head.
            heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.bn = nn.BatchNorm1d(out_channels * heads)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with attention."""
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class GNNPredictor(nn.Module):
    """
    Graph Neural Network for molecular property prediction.
    
    Architecture:
        1. Input projection layer
        2. Stacked GNN layers (GCN or GAT)
        3. Global pooling
        4. MLP prediction head
    
    Attributes:
        architecture: Type of GNN layer (GCN, GAT, or MPNN).
        hidden_dim: Hidden layer dimension.
        num_layers: Number of GNN layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        architecture: Literal["GCN", "GAT", "MPNN"] = "GCN",
        pooling: Literal["mean", "max", "sum"] = "mean",
        edge_dim: Optional[int] = None
    ):
        """
        Initialize the GNN predictor.
        
        Args:
            input_dim: Atom feature dimension.
            hidden_dim: Hidden layer dimension.
            output_dim: Number of output properties.
            num_layers: Number of message passing layers.
            dropout: Dropout probability.
            architecture: GNN variant to use.
            pooling: Graph-level pooling method.
            edge_dim: Bond feature dimension (for MPNN).
        """
        super().__init__()
        
        self.architecture = architecture
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            
            if architecture == "GCN":
                self.gnn_layers.append(GCNLayer(in_dim, out_dim, dropout))
            elif architecture == "GAT":
                if i == 0:
                    self.gnn_layers.append(
                        GATLayer(in_dim, out_dim // 4, heads=4, dropout=dropout)
                    )
                else:
                    self.gnn_layers.append(
                        GATLayer(in_dim, out_dim // 4, heads=4, dropout=dropout)
                    )
            elif architecture == "MPNN":
                edge_nn = nn.Sequential(
                    nn.Linear(edge_dim or 14, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, in_dim * out_dim)
                )
                self.gnn_layers.append(NNConv(in_dim, out_dim, edge_nn))
        
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "max":
            self.pool = global_max_pool
        elif pooling == "sum":
            self.pool = global_add_pool
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(
        self,
        data: Data,
        return_embedding: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data: PyG Data or Batch object.
            return_embedding: If True, return graph embedding instead of prediction.
            
        Returns:
            Predictions [batch_size, output_dim] or embeddings [batch_size, hidden_dim].
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        x = self.input_proj(x)
        
        for i, layer in enumerate(self.gnn_layers):
            if self.architecture == "MPNN" and hasattr(data, "edge_attr"):
                x = layer(x, edge_index, data.edge_attr)
                x = F.relu(x)
            else:
                x = layer(x, edge_index)
        
        graph_embedding = self.pool(x, batch)
        
        if return_embedding:
            return graph_embedding
        
        output = self.fc(graph_embedding)
        return output
    
    def get_attention_weights(self, data: Data) -> Optional[torch.Tensor]:
        """
        Extract attention weights from GAT layers for interpretation.
        
        Args:
            data: Input molecular graph.
            
        Returns:
            Attention weights if using GAT architecture.
        """
        if self.architecture != "GAT":
            return None
        
        x = self.input_proj(data.x)
        attention_weights = []
        
        for layer in self.gnn_layers:
            x, (edge_index, alpha) = layer.conv(
                x, data.edge_index, return_attention_weights=True
            )
            attention_weights.append(alpha)
            x = layer.bn(x)
            x = F.elu(x)
        
        return attention_weights


def create_model(
    config: dict,
    input_dim: int,
    edge_dim: Optional[int] = None
) -> GNNPredictor:
    """
    Factory function to create a GNN model from configuration.
    
    Args:
        config: Model configuration dictionary.
        input_dim: Atom feature dimension.
        edge_dim: Bond feature dimension.
        
    Returns:
        Configured GNNPredictor instance.
    """
    return GNNPredictor(
        input_dim=input_dim,
        hidden_dim=config.get("hidden_dim", 128),
        output_dim=config.get("output_dim", 1),
        num_layers=config.get("num_layers", 3),
        dropout=config.get("dropout", 0.2),
        architecture=config.get("architecture", "GCN"),
        pooling=config.get("pooling", "mean"),
        edge_dim=edge_dim
    )
