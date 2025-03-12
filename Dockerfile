FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CUDA bellek yönetimi için çevre değişkenleri
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# İki GPU'yu da kullan
ENV CUDA_VISIBLE_DEVICES=0,1

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY rag_app.py .
COPY Book1.json .

# Model ve storage dizinlerini oluştur
RUN mkdir -p model_cache embedding_cache storage

# Uygulama portunu aç
EXPOSE 5000

# Çalışma komutu
CMD ["python", "rag_app.py"] 