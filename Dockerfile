FROM python:3.11-slim

LABEL maintainer="AgHealth+"
LABEL description="Agentic AI Nutrition and Healthcare Monitor"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create required directories
RUN mkdir -p data/synthetic results/logs results/graphs results/tables

# Expose API port
EXPOSE 8000

# Default: run FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
