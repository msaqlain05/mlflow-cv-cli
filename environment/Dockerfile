FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt /app/environment/requirements.txt
RUN pip install --no-cache-dir -r /app/environment/requirements.txt

# Copy project files
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Default command
CMD ["/bin/bash"]