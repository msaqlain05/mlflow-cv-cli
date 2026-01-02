#!/bin/bash
# tests/test.sh

set -e

echo "Running MLflow CV CLI tests..."

# Run pytest
pytest /app/tests/test_outputs.py -v --tb=short

echo "All tests passed!"