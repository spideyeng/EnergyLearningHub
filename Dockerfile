FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ChromaDB and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Render uses PORT env var (default 10000)
ENV PORT=10000
ENV GRADIO_SERVER_NAME="0.0.0.0"

EXPOSE ${PORT}

CMD ["python", "app.py"]
