FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# CUDA bellek yönetimi için çevre değişkenleri
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
# İki GPU'yu da kullan
ENV CUDA_VISIBLE_DEVICES=0,1
# CUDA ortam değişkenleri
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# NVIDIA sürücü bilgilerini kontrol et
RUN nvidia-smi || echo "nvidia-smi komutu çalıştırılamadı, ancak bu normal olabilir. Docker çalıştırılırken NVIDIA sürücüleri kullanılabilir olacaktır."

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CUDA kullanılabilirliğini kontrol et
RUN python -c "import torch; print('CUDA kullanılabilir:', torch.cuda.is_available()); print('CUDA sürümü:', torch.version.cuda if torch.cuda.is_available() else 'Yok')" || echo "PyTorch CUDA kontrolü başarısız oldu, ancak bu normal olabilir."

# Uygulama dosyalarını kopyala
COPY rag_app.py .
COPY api.py .
COPY db_processor.py .
COPY pdf_processor.py .
COPY prompts.py .
COPY system_monitor.py .
COPY websockets.py .
COPY Book1.json .
COPY document.pdf .

# Model ve storage dizinlerini oluştur
RUN mkdir -p model_cache embedding_cache storage pdf_storage

# Uygulama portunu aç
EXPOSE 5000

# Çalışma komutu (Gunicorn + Eventlet ile)
CMD ["gunicorn", "-k", "eventlet", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "120", "rag_app:app"]
# Eski CMD: CMD ["python", "rag_app.py"] 