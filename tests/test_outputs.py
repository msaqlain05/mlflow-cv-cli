# tests/test_outputs.py
"""
Test suite for MLflow CV CLI
"""

import pytest
import mlflow
import pandas as pd
from pathlib import Path
import json


class TestMLflowPipeline:
    """Test the MLflow CV pipeline outputs."""
    
    def test_leaderboard_exists(self):
        """Verify leaderboard.csv is generated."""
        leaderboard_path = Path('/app/leaderboard.csv')
        assert leaderboard_path.exists(), "leaderboard.csv not found"
    
    def test_leaderboard_structure(self):
        """Verify leaderboard has correct columns and data."""
        df = pd.read_csv('/app/leaderboard.csv')
        
        required_columns = ['trial_id', 'params', 'avg_roc_auc', 
                           'avg_average_precision', 'avg_accuracy']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
        
        assert len(df) > 0, "Leaderboard is empty"
        assert df['avg_roc_auc'].max() <= 1.0, "ROC AUC out of bounds"
        assert df['avg_roc_auc'].min() >= 0.0, "ROC AUC out of bounds"
    
    def test_mlflow_experiment_exists(self):
        """Verify MLflow experiment was created."""
        mlflow.set_tracking_uri('/app/mlruns')
        experiment = mlflow.get_experiment_by_name('cv_experiment')
        assert experiment is not None, "Experiment not found"
    
    def test_parent_run_exists(self):
        """Verify parent run was logged."""
        mlflow.set_tracking_uri('/app/mlruns')
        experiment = mlflow.get_experiment_by_name('cv_experiment')
        
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'hyperparameter_search'"
        )
        assert len(runs) > 0, "Parent run not found"
    
    def test_nested_runs_exist(self):
        """Verify nested trial and fold runs exist."""
        mlflow.set_tracking_uri('/app/mlruns')
        experiment = mlflow.get_experiment_by_name('cv_experiment')
        
        client = mlflow.tracking.MlflowClient()
        all_runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        
        # Check for trial runs
        trial_runs = [r for r in all_runs if 'trial_' in r.info.run_name]
        assert len(trial_runs) > 0, "No trial runs found"
        
        # Check for fold runs
        fold_runs = [r for r in all_runs if 'fold_' in r.info.run_name]
        assert len(fold_runs) >= 5, "Expected at least 5 fold runs"
    
    def test_artifacts_logged(self):
        """Verify required artifacts are logged."""
        mlflow.set_tracking_uri('/app/mlruns')
        experiment = mlflow.get_experiment_by_name('cv_experiment')
        
        client = mlflow.tracking.MlflowClient()
        fold_runs = [r for r in client.search_runs(experiment_ids=[experiment.experiment_id])
                    if 'fold_' in r.info.run_name]
        
        assert len(fold_runs) > 0, "No fold runs to check artifacts"
        
        # Check first fold run for required artifacts
        run = fold_runs[0]
        artifacts = client.list_artifacts(run.info.run_id)
        artifact_names = [a.path for a in artifacts]
        
        required_patterns = ['roc_', 'pr_', 'cm_', 'model_']
        for pattern in required_patterns:
            assert any(pattern in name for name in artifact_names), \
                f"Missing artifact matching pattern: {pattern}"
    
    def test_model_registered(self):
        """Verify best model is registered."""
        mlflow.set_tracking_uri('/app/mlruns')
        client = mlflow.tracking.MlflowClient()
        
        try:
            model_versions = client.search_model_versions("name='best_cv_model'")
            assert len(model_versions) > 0, "Model not registered"
        except Exception as e:
            pytest.fail(f"Failed to find registered model: {e}")
    
    def test_resumability(self):
        """Verify pipeline can resume without duplicating runs."""
        # This test would run the pipeline twice and verify
        # the second run skips existing trial/fold combinations
        # Implementation depends on test execution strategy
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])