#!/bin/bash
# solution/solve.sh

set -e  # Exit on error

# Set absolute paths
export PYTHONPATH=/app:$PYTHONPATH
export MLFLOW_TRACKING_URI=/app/mlruns

# Run the CLI
python3 /app/solution/mlflow_cli.py \
    --data-path /app/data/dataset.csv \
    --tracking-uri /app/mlruns \
    --experiment-name cv_experiment \
    --model-name best_cv_model

echo "Solution execution completed"