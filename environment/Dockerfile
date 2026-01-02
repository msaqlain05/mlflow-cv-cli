FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment/requirements.txt /app/environment/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/environment/requirements.txt

COPY . /app/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns

CMD ["/bin/bash"]
