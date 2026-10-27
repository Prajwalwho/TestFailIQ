# Multi-stage Dockerfile for all-in-one deployment (Hugging Face Spaces, Render, Koyeb, etc.)

# Stage 1: Build the React Frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# Stage 2: Setup Python Backend and serve Frontend
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    PORT=7860

WORKDIR /app

# Install CPU-only PyTorch to minimize image size and memory usage
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

# Pre-download the embedding model so no network calls are needed on cold start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY backend/ ./
# Copy built static frontend files into backend static directory
COPY --from=frontend-builder /app/frontend/dist ./static

EXPOSE 7860
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
