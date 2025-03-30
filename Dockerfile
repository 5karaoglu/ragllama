FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Etkileşimsiz yapılandırma ve timezone ayarları
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Istanbul

# NVIDIA apt repo ekle
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    && apt-key del 7fa2af80 || true \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - \
    && echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-cuda.list

# Timezone paketini önceden yükle ve yapılandır
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    software-properties-common \
    build-essential \
    procps \
    libnccl2 \
    libnccl-dev \
    && rm -rf /var/lib/apt/lists/*

# CUDA ve vLLM için ortam değişkenlerini ayarla
ENV CUDA_DEVICE_MAX_CONNECTIONS=1 \
    NCCL_DEBUG=INFO \
    NCCL_SOCKET_IFNAME=^lo,docker \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=0 \
    NCCL_CROSS_NIC=1 \
    NCCL_ASYNC_ERROR_HANDLING=1 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024 \
    # Ray ve vLLM bellek ve performans ayarları
    OMP_NUM_THREADS=8 \
    TOKENIZERS_PARALLELISM=true \
    RAY_DEDUP_LOGS=0 \
    RAY_memory_monitor_refresh_ms=0

# Ray için geçici klasörleri oluştur ve izinleri ayarla
RUN mkdir -p /tmp/ray && \
    chmod 777 /tmp/ray && \
    mkdir -p /tmp/shm && \
    chmod 777 /tmp/shm

# VLLM ve diğer bağımlılıkları yükle
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    vllm==0.3.0 \
    ray==2.7.1 \
    pynccl \
    psutil \
    Flask==2.3.3 \
    llama-index==0.9.16 \
    pypdf \
    pandas==2.1.1 \
    langchain==0.0.312 \
    sentence_transformers==2.2.2 \
    faiss-gpu==1.7.2 \
    llama-index-vector-stores-faiss==0.1.2 \
    llama-index-llms-openai==0.1.4 \
    llama-index-llms-huggingface==0.1.4 \
    llama-index-embeddings-huggingface==0.1.3 \
    llama-index-readers-file==0.1.4 \
    && python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" \
    && python -c "from vllm import LLM; print('vLLM imported successfully')"

# Ray önbelleği için geçici dizini oluştur
RUN mkdir -p /tmp/ray/session_files && \
    chmod -R 777 /tmp/ray

# Uygulama klasörü oluştur ve dosyaları kopyala
WORKDIR /app
COPY . /app

# Uygulama için gereken klasörleri oluştur
RUN mkdir -p /app/model_cache /app/embedding_cache /app/storage /app/pdf_storage && \
    chmod -R 777 /app

# Sistem bilgilerini kontrol et ve eğer varsa GPU'ları listele
RUN nvidia-smi -L || echo "NVIDIA driver not found - will use CPU mode" && \
    echo "Checking NCCL installation:" && ls -la /usr/lib/x86_64-linux-gnu/libnccl* || echo "NCCL not found in standard location"

# GPU bellek kullanımını sınırla
ENV VLLM_GPU_MEMORY_UTILIZATION=0.75 \
    VLLM_MAX_MODEL_LEN=4096 \
    VLLM_ENFORCE_EAGER=1 \
    VLLM_USE_PAGED_ATTENTION=true \
    VLLM_ENABLE_DISK_CACHE=true

# Docker build tamamlandıktan sonra etkileşimli modu tekrar etkinleştir
ENV DEBIAN_FRONTEND=dialog

# Port yapılandırması ve girdi noktası
EXPOSE 5000
CMD ["python", "rag_app.py"] 