FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# NVIDIA paket deposunu ekle - NCCL için
RUN wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA bellek yönetimi için çevre değişkenleri
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
# İki GPU'yu da kullan
ENV CUDA_VISIBLE_DEVICES=0,1
# CUDA ortam değişkenleri
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# vLLM yapılandırması
ENV VLLM_USE_PAGED_ATTENTION=true
ENV VLLM_ATTENTION_SHARD_SIZE=1024
ENV VLLM_MAX_PARALLEL_LOADING_WORKERS=2
ENV VLLM_GPU_MEMORY_UTILIZATION=0.85
ENV VLLM_TENSOR_PARALLEL_SIZE=2

# NVIDIA sürücü bilgilerini kontrol et
RUN nvidia-smi || echo "nvidia-smi komutu çalıştırılamadı, ancak bu normal olabilir. Docker çalıştırılırken NVIDIA sürücüleri kullanılabilir olacaktır."

# Python bağımlılıklarını kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyTorch CUDA kullanılabilirliğini kontrol et
RUN python -c "import torch; print('CUDA kullanılabilir:', torch.cuda.is_available()); print('CUDA sürümü:', torch.version.cuda if torch.cuda.is_available() else 'Yok')" || echo "PyTorch CUDA kontrolü başarısız oldu, ancak bu normal olabilir."

# vLLM kurulumunu kontrol et
RUN python -c "from vllm import LLM; print('vLLM başarıyla kuruldu.')" || echo "vLLM kontrolü başarısız oldu, ancak bu normal olabilir."

# Uygulama dosyalarını kopyala
COPY rag_app.py .
COPY Book1.json .
COPY document.pdf .

# Model ve storage dizinlerini oluştur
RUN mkdir -p model_cache embedding_cache storage pdf_storage

# Uygulama portunu aç
EXPOSE 5000

# Çalışma komutu
CMD ["python", "rag_app.py"] 