"""
Molecular featurization for Graph Neural Networks.

Converts SMILES strings to PyTorch Geometric Data objects with
atom and bond features suitable for GNN training.
"""

from typing import List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from tqdm import tqdm


class MolecularFeaturizer:
    """
    Converts SMILES to graph representations for GNNs.
    
    Generates atom-level and bond-level features from molecular
    structures using RDKit.
    
    Attributes:
        atom_feature_dim: Dimension of atom feature vectors.
        bond_feature_dim: Dimension of bond feature vectors.
    """
    
    ATOM_FEATURES = {
        "atomic_num": list(range(1, 119)),
        "chirality": [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER,
        ],
        "degree": list(range(11)),
        "formal_charge": list(range(-5, 6)),
        "num_hs": list(range(9)),
        "hybridization": [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            Chem.rdchem.HybridizationType.UNSPECIFIED,
        ],
    }
    
    BOND_FEATURES = {
        "bond_type": [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ],
        "stereo": [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ],
    }
    
    def __init__(self):
        """Initialize the featurizer and compute feature dimensions."""
        self.atom_feature_dim = self._compute_atom_dim()
        self.bond_feature_dim = self._compute_bond_dim()
    
    def _compute_atom_dim(self) -> int:
        """Compute total atom feature dimension."""
        dim = 0
        for key, values in self.ATOM_FEATURES.items():
            dim += len(values) + 1  # +1 for unknown
        dim += 2  # is_aromatic, is_in_ring (binary)
        return dim
    
    def _compute_bond_dim(self) -> int:
        """Compute total bond feature dimension."""
        dim = 0
        for key, values in self.BOND_FEATURES.items():
            dim += len(values) + 1
        dim += 2  # is_conjugated, is_in_ring
        return dim
    
    def _one_hot_encode(
        self,
        value,
        allowable_set: List,
        include_unknown: bool = True
    ) -> List[float]:
        """
        One-hot encode a value.
        
        Args:
            value: Value to encode.
            allowable_set: List of allowed values.
            include_unknown: Whether to include unknown category.
            
        Returns:
            One-hot encoded list.
        """
        encoding = [0.0] * len(allowable_set)
        
        if value in allowable_set:
            encoding[allowable_set.index(value)] = 1.0
        elif include_unknown:
            encoding.append(1.0)
            return encoding
        
        if include_unknown:
            encoding.append(0.0)
        
        return encoding
    
    def get_atom_features(self, atom: Chem.Atom) -> np.ndarray:
        """
        Extract features for a single atom.
        
        Args:
            atom: RDKit Atom object.
            
        Returns:
            Feature vector as numpy array.
        """
        features = []
        
        features.extend(self._one_hot_encode(
            atom.GetAtomicNum(),
            self.ATOM_FEATURES["atomic_num"]
        ))
        
        features.extend(self._one_hot_encode(
            atom.GetChiralTag(),
            self.ATOM_FEATURES["chirality"]
        ))
        
        features.extend(self._one_hot_encode(
            atom.GetTotalDegree(),
            self.ATOM_FEATURES["degree"]
        ))
        
        features.extend(self._one_hot_encode(
            atom.GetFormalCharge(),
            self.ATOM_FEATURES["formal_charge"]
        ))
        
        features.extend(self._one_hot_encode(
            atom.GetTotalNumHs(),
            self.ATOM_FEATURES["num_hs"]
        ))
        
        features.extend(self._one_hot_encode(
            atom.GetHybridization(),
            self.ATOM_FEATURES["hybridization"]
        ))
        
        features.append(float(atom.GetIsAromatic()))
        features.append(float(atom.IsInRing()))
        
        return np.array(features, dtype=np.float32)
    
    def get_bond_features(self, bond: Chem.Bond) -> np.ndarray:
        """
        Extract features for a single bond.
        
        Args:
            bond: RDKit Bond object.
            
        Returns:
            Feature vector as numpy array.
        """
        features = []
        
        features.extend(self._one_hot_encode(
            bond.GetBondType(),
            self.BOND_FEATURES["bond_type"]
        ))
        
        features.extend(self._one_hot_encode(
            bond.GetStereo(),
            self.BOND_FEATURES["stereo"]
        ))
        
        features.append(float(bond.GetIsConjugated()))
        features.append(float(bond.IsInRing()))
        
        return np.array(features, dtype=np.float32)
    
    def smiles_to_graph(
        self,
        smiles: str,
        label: Optional[float] = None
    ) -> Optional[Data]:
        """
        Convert a SMILES string to a PyTorch Geometric Data object.
        
        Args:
            smiles: SMILES string representation.
            label: Optional target value.
            
        Returns:
            PyG Data object or None if parsing fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        
        x = torch.tensor(np.array(atom_features), dtype=torch.float)
        
        edge_index = []
        edge_attr = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            bond_feat = self.get_bond_features(bond)
            
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bond_feat)
            edge_attr.append(bond_feat)
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, self.bond_feature_dim), dtype=torch.float)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            smiles=smiles
        )
        
        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float)
        
        return data
    
    def featurize_dataset(
        self,
        smiles_list: np.ndarray,
        labels: np.ndarray,
        show_progress: bool = True
    ) -> List[Data]:
        """
        Convert a dataset of SMILES to graph objects.
        
        Args:
            smiles_list: Array of SMILES strings.
            labels: Array of target values.
            show_progress: Whether to show progress bar.
            
        Returns:
            List of PyG Data objects (excludes failed parses).
        """
        graphs = []
        failed = 0
        
        iterator = zip(smiles_list, labels)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc="Featurizing molecules"
            )
        
        for smiles, label in iterator:
            graph = self.smiles_to_graph(smiles, label)
            if graph is not None:
                graphs.append(graph)
            else:
                failed += 1
        
        if failed > 0:
            print(f"Warning: {failed} molecules failed to parse")
        
        return graphs
    
    def get_molecular_descriptors(self, smiles: str) -> Optional[dict]:
        """
        Compute molecular descriptors for interpretation.
        
        Args:
            smiles: SMILES string.
            
        Returns:
            Dictionary of molecular properties or None if parsing fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "fraction_sp3": Descriptors.FractionCSP3(mol),
        }


class MoleculeDataset(Dataset):
    """
    PyTorch Geometric Dataset for molecular graphs.
    
    Wraps a list of Data objects for use with DataLoader.
    """
    
    def __init__(self, graphs: List[Data]):
        """
        Initialize dataset.
        
        Args:
            graphs: List of PyG Data objects.
        """
        super().__init__()
        self.graphs = graphs
    
    def len(self) -> int:
        """Return dataset size."""
        return len(self.graphs)
    
    def get(self, idx: int) -> Data:
        """Get a single graph by index."""
        return self.graphs[idx]
