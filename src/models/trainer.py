"""
Training loop for GNN models with MLflow experiment tracking.

Implements training, validation, early stopping, and checkpointing.
"""

import os
from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import mlflow
from tqdm import tqdm


class Trainer:
    """
    Trainer for GNN molecular property prediction models.
    
    Handles training loop, validation, metrics computation,
    early stopping, and MLflow logging.
    
    Attributes:
        model: GNN model to train.
        task_type: Either 'regression' or 'classification'.
        device: Training device (cuda or cpu).
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_type: Literal["regression", "classification"] = "regression",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: GNN model instance.
            task_type: Type of prediction task.
            learning_rate: Optimizer learning rate.
            weight_decay: L2 regularization strength.
            device: Device to train on.
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.task_type = task_type
        
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        if task_type == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def train_epoch(self, loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            loader: Training data loader.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in loader:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output.squeeze(), batch.y.squeeze())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        
        return total_loss / len(loader.dataset)
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on a dataset.
        
        Args:
            loader: Evaluation data loader.
            
        Returns:
            Tuple of (loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            loss = self.criterion(output.squeeze(), batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            
            all_preds.append(output.cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        
        preds = np.concatenate(all_preds).squeeze()
        labels = np.concatenate(all_labels).squeeze()
        
        metrics = self._compute_metrics(preds, labels)
        metrics["loss"] = avg_loss
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute task-specific metrics.
        
        Args:
            predictions: Model predictions.
            labels: Ground truth labels.
            
        Returns:
            Dictionary of metric names to values.
        """
        metrics = {}
        
        if self.task_type == "regression":
            metrics["rmse"] = np.sqrt(mean_squared_error(labels, predictions))
            metrics["mae"] = mean_absolute_error(labels, predictions)
            metrics["r2"] = r2_score(labels, predictions)
        else:
            probs = 1 / (1 + np.exp(-predictions))
            try:
                metrics["auroc"] = roc_auc_score(labels, probs)
            except ValueError:
                metrics["auroc"] = 0.5
            
            binary_preds = (probs > 0.5).astype(int)
            metrics["accuracy"] = np.mean(binary_preds == labels)
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        checkpoint_dir: str = "./models",
        experiment_name: str = "admet_prediction",
        run_name: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping and MLflow logging.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Maximum number of epochs.
            patience: Early stopping patience.
            checkpoint_dir: Directory to save model checkpoints.
            experiment_name: MLflow experiment name.
            run_name: Optional MLflow run name.
            
        Returns:
            Dictionary with training history.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "task_type": self.task_type,
                "epochs": epochs,
                "patience": patience,
                "optimizer": "Adam",
                "model_params": sum(p.numel() for p in self.model.parameters()),
            })
            
            for epoch in range(epochs):
                train_loss = self.train_epoch(train_loader)
                val_loss, val_metrics = self.evaluate(val_loader)
                
                self.scheduler.step(val_loss)
                
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items() if k != "loss"}
                }, step=epoch)
                
                primary_metric = (
                    val_metrics.get("rmse", val_loss) if self.task_type == "regression"
                    else -val_metrics.get("auroc", -val_loss)
                )
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    checkpoint_path = os.path.join(
                        checkpoint_dir, "best_model.pt"
                    )
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    }, checkpoint_path)
                else:
                    self.patience_counter += 1
                
                if self.task_type == "regression":
                    metric_str = f"RMSE: {val_metrics['rmse']:.4f}, R2: {val_metrics['r2']:.4f}"
                else:
                    metric_str = f"AUROC: {val_metrics['auroc']:.4f}, Acc: {val_metrics['accuracy']:.4f}"
                
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"{metric_str}"
                )
                
                if self.patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            mlflow.log_artifact(os.path.join(checkpoint_dir, "best_model.pt"))
        
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Validation metrics: {checkpoint['val_metrics']}")
    
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        """
        Generate predictions for a dataset.
        
        Args:
            loader: Data loader for inference.
            
        Returns:
            Array of predictions.
        """
        self.model.eval()
        all_preds = []
        
        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            all_preds.append(output.cpu().numpy())
        
        predictions = np.concatenate(all_preds).squeeze()
        
        if self.task_type == "classification":
            predictions = 1 / (1 + np.exp(-predictions))
        
        return predictions
    
    @torch.no_grad()
    def predict_single(self, data) -> Tuple[float, np.ndarray]:
        """
        Predict for a single molecule.
        
        Args:
            data: PyG Data object for one molecule.
            
        Returns:
            Tuple of (prediction, embedding).
        """
        self.model.eval()
        data = data.to(self.device)
        
        prediction = self.model(data).cpu().numpy().squeeze()
        embedding = self.model(data, return_embedding=True).cpu().numpy().squeeze()
        
        if self.task_type == "classification":
            prediction = 1 / (1 + np.exp(-prediction))
        
        return float(prediction), embedding
