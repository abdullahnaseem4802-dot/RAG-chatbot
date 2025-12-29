FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY yolov8_detector.py .
COPY mock_rag_local.py .

# Copy model file and create models directory
RUN mkdir -p models
COPY models/best.pt models/best.pt

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
