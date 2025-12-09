# Dockerfile cho Hugging Face Space

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY data_processing/ ./data_processing/
COPY models/ ./models/
COPY training/ ./training/
COPY app/ ./app/

# Download model weights from Hugging Face Hub
RUN mkdir -p /app/weights
ENV MODEL_PATH=/app/weights/best_model.pth

# Expose port
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the app
CMD ["python", "app/gradio_app.py"]