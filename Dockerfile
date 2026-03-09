FROM python:3.11-slim

WORKDIR /app

# System deps for ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and data
COPY . .

# Render binds to PORT env var (default 10000)
ENV PORT=10000
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=10000

EXPOSE 10000

CMD ["python", "app.py"]
