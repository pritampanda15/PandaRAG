"""
Data loader for ADMET datasets from Therapeutics Data Commons (TDC).

This module handles downloading, splitting, and preparing ADMET benchmark
datasets for training Graph Neural Networks.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from tdc.single_pred import ADME, Tox
from sklearn.model_selection import train_test_split


class ADMETDataLoader:
    """
    Loader for ADMET property prediction datasets from TDC.
    
    Supports both ADME (absorption, distribution, metabolism, excretion)
    and Toxicity endpoints with consistent preprocessing.
    
    Attributes:
        dataset_name: Name of the TDC dataset to load.
        task_type: Either 'regression' or 'classification'.
        random_seed: Seed for reproducible splits.
    """
    
    ADME_DATASETS = {
        "Caco2_Wang": {"task": "regression", "category": "ADME"},
        "HIA_Hou": {"task": "classification", "category": "ADME"},
        "Pgp_Broccatelli": {"task": "classification", "category": "ADME"},
        "Bioavailability_Ma": {"task": "classification", "category": "ADME"},
        "BBB_Martins": {"task": "classification", "category": "ADME"},
        "PPBR_AZ": {"task": "regression", "category": "ADME"},
        "VDss_Lombardo": {"task": "regression", "category": "ADME"},
        "CYP2D6_Veith": {"task": "classification", "category": "ADME"},
        "CYP3A4_Veith": {"task": "classification", "category": "ADME"},
        "CYP2C9_Veith": {"task": "classification", "category": "ADME"},
        "Half_Life_Obach": {"task": "regression", "category": "ADME"},
        "Clearance_Hepatocyte_AZ": {"task": "regression", "category": "ADME"},
    }
    
    TOX_DATASETS = {
        "hERG": {"task": "classification", "category": "Tox"},
        "AMES": {"task": "classification", "category": "Tox"},
        "DILI": {"task": "classification", "category": "Tox"},
        "LD50_Zhu": {"task": "regression", "category": "Tox"},
    }
    
    def __init__(
        self,
        dataset_name: str,
        task_type: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset_name: Name of the TDC dataset.
            task_type: Override task type (regression/classification).
            random_seed: Random seed for splits.
            
        Raises:
            ValueError: If dataset_name is not supported.
        """
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        
        all_datasets = {**self.ADME_DATASETS, **self.TOX_DATASETS}
        
        if dataset_name not in all_datasets:
            available = list(all_datasets.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. "
                f"Available datasets: {available}"
            )
        
        dataset_info = all_datasets[dataset_name]
        self.category = dataset_info["category"]
        self.task_type = task_type or dataset_info["task"]
        
        self.data: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
    
    def load(self, data_dir: str = "./data/raw") -> pd.DataFrame:
        """
        Download and load the dataset.
        
        Args:
            data_dir: Directory to store downloaded data.
            
        Returns:
            DataFrame with columns: Drug_ID, Drug (SMILES), Y (target).
        """
        os.makedirs(data_dir, exist_ok=True)
        
        if self.category == "ADME":
            dataset = ADME(name=self.dataset_name, path=data_dir)
        else:
            dataset = Tox(name=self.dataset_name, path=data_dir)
        
        self.data = dataset.get_data()
        
        print(f"Loaded {self.dataset_name}: {len(self.data)} samples")
        print(f"Task type: {self.task_type}")
        print(f"Target statistics:")
        print(self.data["Y"].describe())
        
        return self.data
    
    def split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Uses stratified splitting for classification tasks.
        
        Args:
            test_size: Fraction for test set.
            val_size: Fraction for validation set (from remaining after test).
            
        Returns:
            Tuple of (train_df, val_df, test_df).
            
        Raises:
            RuntimeError: If data not loaded.
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")
        
        stratify = None
        if self.task_type == "classification":
            stratify = self.data["Y"]
        
        train_val, self.test_df = train_test_split(
            self.data,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=stratify
        )
        
        val_adjusted = val_size / (1 - test_size)
        stratify_val = None
        if self.task_type == "classification":
            stratify_val = train_val["Y"]
        
        self.train_df, self.val_df = train_test_split(
            train_val,
            test_size=val_adjusted,
            random_state=self.random_seed,
            stratify=stratify_val
        )
        
        print(f"Split sizes - Train: {len(self.train_df)}, "
              f"Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        return self.train_df, self.val_df, self.test_df
    
    def get_smiles_and_labels(
        self,
        split: str = "train"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get SMILES strings and target values for a split.
        
        Args:
            split: One of 'train', 'val', or 'test'.
            
        Returns:
            Tuple of (smiles_array, labels_array).
        """
        split_map = {
            "train": self.train_df,
            "val": self.val_df,
            "test": self.test_df
        }
        
        if split not in split_map:
            raise ValueError(f"Split must be one of {list(split_map.keys())}")
        
        df = split_map[split]
        if df is None:
            raise RuntimeError(f"Split '{split}' not available. Call split() first.")
        
        smiles = df["Drug"].values
        labels = df["Y"].values.astype(np.float32)
        
        return smiles, labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics and configuration.
        """
        info = {
            "name": self.dataset_name,
            "category": self.category,
            "task_type": self.task_type,
            "total_samples": len(self.data) if self.data is not None else 0,
        }
        
        if self.data is not None:
            info["target_mean"] = float(self.data["Y"].mean())
            info["target_std"] = float(self.data["Y"].std())
            
            if self.task_type == "classification":
                info["class_distribution"] = self.data["Y"].value_counts().to_dict()
        
        return info


def get_available_datasets() -> Dict[str, Dict[str, str]]:
    """
    Get all available ADMET datasets with their task types.
    
    Returns:
        Dictionary mapping dataset names to their metadata.
    """
    return {
        **ADMETDataLoader.ADME_DATASETS,
        **ADMETDataLoader.TOX_DATASETS
    }
