# create_dataset.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
    flip_y=0.1  # Add some noise
)

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
df.to_csv('data/dataset.csv', index=False)
print(f"✓ Dataset created: {len(df)} rows, {len(df.columns)} columns")
print(f"✓ Target distribution:\n{df['target'].value_counts()}")