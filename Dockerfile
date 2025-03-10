FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Gerekli paketleri yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Uygulama dosyalarını kopyala
COPY requirements.txt .
COPY main.py .
COPY modules/ ./modules/

# PDF dizinini oluştur
RUN mkdir -p pdf_docs storage/json storage/pdf

# Gerekli Python paketlerini yükle
RUN pip3 install --no-cache-dir -r requirements.txt

# GPU bellek optimizasyonu için ortam değişkenleri
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=256

# Uygulama portunu aç
EXPOSE 5000

# Uygulamayı başlat - worker sayısını azalttık
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "main:app"] 