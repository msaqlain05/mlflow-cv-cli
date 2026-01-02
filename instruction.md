# MLflow CLI for Resumable 5-Fold Cross-Validation

## Objective
Develop a Python CLI that performs hyperparameter search using 5-fold cross-validation
with MLflow tracking and automatic model registration.

## Requirements

### Input
- Dataset path (absolute path to CSV file)
- Hyperparameter grid (JSON or command-line arguments)
- MLflow tracking URI

### Output
- MLflow parent run with nested child runs
- leaderboard.csv with aggregated results
- Best model registered in MLflow Model Registry

### Key Features
1. **5-Fold Cross-Validation**: Split data into 5 folds
2. **MLflow Tracking**: Log parameters, metrics, and artifacts
3. **Resumability**: Skip completed trial/fold combinations
4. **Artifact Logging**: ROC curves, PR curves, confusion matrices, models
5. **Automatic Registration**: Register best model with signature

### Success Criteria
- All runs properly nested in MLflow
- leaderboard.csv contains correct aggregated metrics
- Best model registered with valid signature and input example
- Script can resume interrupted jobs without duplication