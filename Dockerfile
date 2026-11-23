FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU sürümü PyTorch; ekran kartı olan sunucuda CUDA sürümü kurulabilir
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY models/yolo11s_1024.pt models/

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
