# Use a slim Python base
FROM python:3.11-slim

# System deps (build + runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code + data (optional)
COPY fastapi_phoenix_agent.py .
# If your app expects Sales.parquet locally, copy it. Alternatively, load from S3 at runtime instead.
COPY Sales.parquet ./Sales.parquet

# Expose port used by uvicorn
EXPOSE 8000

# Env to make uvicorn address/port explicit
ENV HOST=0.0.0.0
ENV PORT=8000

# Start server
CMD ["uvicorn", "fastapi_phoenix_agent:app", "--host", "0.0.0.0", "--port", "8000"]
