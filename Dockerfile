FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# CUDA bellek yönetimi için çevre değişkenleri
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
# İki GPU'yu da kullan
ENV CUDA_VISIBLE_DEVICES=0,1

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Çalışma komutu
CMD ["python", "rag_app.py"] 