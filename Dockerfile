# --- Base Image ---
FROM python:3.10-slim

# --- Working Directory ---
WORKDIR /app

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Copy project files ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# --- Default command ---
CMD ["python", "main.py"]
