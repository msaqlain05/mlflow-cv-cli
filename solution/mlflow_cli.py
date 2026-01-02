# solution/mlflow_cli.py
#!/usr/bin/env python3
"""
MLflow CLI for Resumable 5-Fold Cross-Validation
"""

import click
import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import hashlib


class MLflowCVPipeline:
    """Manages 5-fold CV with MLflow tracking and resumability."""
    
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.n_folds = 5
        self.random_state = 42
        
    def generate_trial_id(self, params: Dict) -> str:
        """Generate deterministic trial ID from parameters."""
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def check_run_exists(self, parent_run_id: str, trial_id: str, fold: int) -> bool:
        """Check if a specific trial/fold combination has been completed."""
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name(self.experiment_name).experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' and tags.trial_id = '{trial_id}' and tags.fold = '{fold}'",
            max_results=1
        )
        return len(runs) > 0
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       trial_id: str, fold: int) -> str:
        """Generate and save ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - Trial {trial_id} Fold {fold}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = f"/tmp/roc_trial_{trial_id}_fold_{fold}.png"
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      trial_id: str, fold: int) -> str:
        """Generate and save Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Trial {trial_id} Fold {fold}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = f"/tmp/pr_trial_{trial_id}_fold_{fold}.png"
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             trial_id: str, fold: int) -> str:
        """Generate and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Trial {trial_id} Fold {fold}')
        
        filepath = f"/tmp/cm_trial_{trial_id}_fold_{fold}.png"
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        return filepath
    
    def train_and_evaluate_fold(self, X_train: np.ndarray, X_val: np.ndarray,
                                y_train: np.ndarray, y_val: np.ndarray,
                                params: Dict, trial_id: str, fold: int) -> Dict:
        """Train model on one fold and return metrics."""
        # Train model
        model = RandomForestClassifier(**params, random_state=self.random_state)
        model.fit(X_train, y_train)
        
        # Generate predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_val, y_pred_proba),
            'average_precision': average_precision_score(y_val, y_pred_proba),
            'accuracy': (y_pred == y_val).mean()
        }
        
        # Generate artifacts
        roc_path = self.plot_roc_curve(y_val, y_pred_proba, trial_id, fold)
        pr_path = self.plot_pr_curve(y_val, y_pred_proba, trial_id, fold)
        cm_path = self.plot_confusion_matrix(y_val, y_pred, trial_id, fold)
        
        # Serialize model
        model_path = f"/tmp/model_trial_{trial_id}_fold_{fold}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        artifacts = {
            'roc_curve': roc_path,
            'pr_curve': pr_path,
            'confusion_matrix': cm_path,
            'model': model_path
        }
        
        return metrics, artifacts, model
    
    def run_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                            param_grid: List[Dict]) -> pd.DataFrame:
        """Execute hyperparameter search with 5-fold CV."""
        leaderboard_data = []
        
        # Start parent run
        with mlflow.start_run(run_name="hyperparameter_search") as parent_run:
            parent_run_id = parent_run.info.run_id
            mlflow.log_param("n_folds", self.n_folds)
            mlflow.log_param("n_trials", len(param_grid))
            
            # Iterate through hyperparameter combinations
            for trial_idx, params in enumerate(param_grid):
                trial_id = self.generate_trial_id(params)
                
                # Start trial run
                with mlflow.start_run(run_name=f"trial_{trial_id}", nested=True) as trial_run:
                    mlflow.set_tags({
                        "trial_id": trial_id,
                        "trial_index": trial_idx
                    })
                    mlflow.log_params(params)
                    
                    fold_metrics = []
                    skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                                         random_state=self.random_state)
                    
                    # Iterate through folds
                    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                        # Check if fold already completed (resumability)
                        if self.check_run_exists(parent_run_id, trial_id, fold):
                            print(f"Skipping Trial {trial_id} Fold {fold} - already completed")
                            continue
                        
                        # Start fold run
                        with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                            mlflow.set_tags({
                                "trial_id": trial_id,
                                "fold": fold
                            })
                            
                            # Split data
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            # Train and evaluate
                            metrics, artifacts, model = self.train_and_evaluate_fold(
                                X_train, X_val, y_train, y_val, params, trial_id, fold
                            )
                            
                            # Log to MLflow
                            mlflow.log_metrics(metrics)
                            mlflow.log_artifact(artifacts['roc_curve'])
                            mlflow.log_artifact(artifacts['pr_curve'])
                            mlflow.log_artifact(artifacts['confusion_matrix'])
                            mlflow.log_artifact(artifacts['model'])
                            
                            fold_metrics.append(metrics)
                            print(f"Completed Trial {trial_id} Fold {fold} - ROC AUC: {metrics['roc_auc']:.4f}")
                    
                    # Aggregate metrics across folds
                    if fold_metrics:
                        avg_metrics = {
                            key: np.mean([m[key] for m in fold_metrics])
                            for key in fold_metrics[0].keys()
                        }
                        mlflow.log_metrics({f"avg_{k}": v for k, v in avg_metrics.items()})
                        
                        leaderboard_data.append({
                            'trial_id': trial_id,
                            'params': json.dumps(params),
                            **{f"avg_{k}": v for k, v in avg_metrics.items()}
                        })
        
        # Create leaderboard
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('avg_roc_auc', ascending=False)
        leaderboard.to_csv('/app/leaderboard.csv', index=False)
        
        return leaderboard
    
    def register_best_model(self, X: np.ndarray, y: np.ndarray, 
                           best_params: Dict, model_name: str):
        """Train and register the best model."""
        from mlflow.models import infer_signature
        
        # Train final model on full data
        model = RandomForestClassifier(**best_params, random_state=self.random_state)
        model.fit(X, y)
        
        # Create input example
        input_example = pd.DataFrame(X[:1])
        
        # Infer signature
        predictions = model.predict(X[:5])
        signature = infer_signature(X[:5], predictions)
        
        # Register model
        with mlflow.start_run(run_name="best_model_registration"):
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name
            )
            mlflow.log_params(best_params)
        
        print(f"Best model registered as '{model_name}'")


@click.command()
@click.option('--data-path', required=True, help='Absolute path to dataset CSV')
@click.option('--tracking-uri', default='/app/mlruns', help='MLflow tracking URI')
@click.option('--experiment-name', default='cv_experiment', help='MLflow experiment name')
@click.option('--model-name', default='best_cv_model', help='Registered model name')
def main(data_path: str, tracking_uri: str, experiment_name: str, model_name: str):
    """MLflow CLI for resumable 5-fold cross-validation."""
    
    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Define hyperparameter grid
    param_grid = [
        {'n_estimators': 50, 'max_depth': 5},
        {'n_estimators': 100, 'max_depth': 10},
        {'n_estimators': 150, 'max_depth': 15}
    ]
    
    # Initialize pipeline
    pipeline = MLflowCVPipeline(tracking_uri, experiment_name)
    
    # Run cross-validation
    print("Starting hyperparameter search with 5-fold CV...")
    leaderboard = pipeline.run_cross_validation(X, y, param_grid)
    
    print("\n=== Leaderboard ===")
    print(leaderboard.head())
    
    # Register best model
    best_params = json.loads(leaderboard.iloc[0]['params'])
    pipeline.register_best_model(X, y, best_params, model_name)
    
    print("\n✓ Pipeline completed successfully!")


if __name__ == '__main__':
    main()