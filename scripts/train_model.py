#!/usr/bin/env python
"""
Training script for ADMET prediction models.

Usage:
    python scripts/train_model.py --config config/config.yaml
    python scripts/train_model.py --download-only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.loader import DataLoader

from src.data import ADMETDataLoader, MolecularFeaturizer
from src.data.molecular_featurizer import MoleculeDataset
from src.models import GNNPredictor, Trainer
from src.utils import load_config, set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ADMET prediction model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download dataset without training"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="GPU device ID"
    )
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    config = load_config(args.config)
    
    set_seed(config["data"].get("random_seed", 42))
    
    print("=" * 60)
    print("ADMET Prediction Model Training")
    print("=" * 60)
    
    data_config = config["data"]
    loader = ADMETDataLoader(
        dataset_name=data_config["dataset_name"],
        task_type=data_config.get("task_type"),
        random_seed=data_config.get("random_seed", 42)
    )
    
    print(f"\nLoading dataset: {data_config['dataset_name']}")
    loader.load(data_dir=config["paths"]["raw_data"])
    
    if args.download_only:
        print("\nDataset downloaded successfully.")
        print(loader.get_dataset_info())
        return
    
    print("\nSplitting data...")
    train_df, val_df, test_df = loader.split(
        test_size=data_config.get("test_split", 0.2),
        val_size=data_config.get("val_split", 0.1)
    )
    
    print("\nFeaturizing molecules...")
    featurizer = MolecularFeaturizer()
    
    train_smiles, train_labels = loader.get_smiles_and_labels("train")
    val_smiles, val_labels = loader.get_smiles_and_labels("val")
    test_smiles, test_labels = loader.get_smiles_and_labels("test")
    
    train_graphs = featurizer.featurize_dataset(train_smiles, train_labels)
    val_graphs = featurizer.featurize_dataset(val_smiles, val_labels)
    test_graphs = featurizer.featurize_dataset(test_smiles, test_labels)
    
    print(f"\nFeaturized: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")
    
    model_config = config["model"]
    batch_size = model_config.get("batch_size", 32)
    
    train_loader = DataLoader(
        MoleculeDataset(train_graphs),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        MoleculeDataset(val_graphs),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        MoleculeDataset(test_graphs),
        batch_size=batch_size,
        shuffle=False
    )
    
    print("\nInitializing model...")
    input_dim = featurizer.atom_feature_dim
    edge_dim = featurizer.bond_feature_dim
    
    model = GNNPredictor(
        input_dim=input_dim,
        hidden_dim=model_config.get("hidden_dim", 128),
        output_dim=model_config.get("output_dim", 1),
        num_layers=model_config.get("num_layers", 3),
        dropout=model_config.get("dropout", 0.2),
        architecture=model_config.get("architecture", "GCN"),
        pooling=model_config.get("pooling", "mean"),
        edge_dim=edge_dim
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    device = f"cuda:{args.gpu}" if model_config.get("device") == "cuda" else "cpu"
    
    trainer = Trainer(
        model=model,
        task_type=data_config.get("task_type", "regression"),
        learning_rate=model_config.get("learning_rate", 0.001),
        weight_decay=model_config.get("weight_decay", 0.0001),
        device=device
    )
    
    print("\nStarting training...")
    print("-" * 60)
    
    mlflow_config = config.get("mlflow", {})
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=model_config.get("epochs", 100),
        patience=model_config.get("early_stopping_patience", 15),
        checkpoint_dir=config["paths"].get("models", "./models"),
        experiment_name=mlflow_config.get("experiment_name", "admet_prediction"),
        run_name=f"{data_config['dataset_name']}_{model_config['architecture']}"
    )
    
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    checkpoint_path = os.path.join(
        config["paths"].get("models", "./models"),
        "best_model.pt"
    )
    trainer.load_checkpoint(checkpoint_path)
    
    test_loss, test_metrics = trainer.evaluate(test_loader)
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
