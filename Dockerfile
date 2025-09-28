# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (optional: git, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command (runs main.py)
CMD ["python", "main.py"]
