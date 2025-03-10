FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# Gerekli paketleri yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Uygulama dosyalarını kopyala
COPY requirements.txt .

# Gerekli Python paketlerini yükle
RUN pip3 install --no-cache-dir -r requirements.txt

# Model cache dizinini oluştur ve izinleri ayarla
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Modeli önceden indir ve önbelleğe al - model_type parametresi eklendi
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', cache_dir='/app/model_cache'); \
    model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', cache_dir='/app/model_cache', \
    torch_dtype='auto', device_map='auto', low_cpu_mem_usage=True)"

# Embedding modelini önceden indir - trust_remote_code parametresi kaldırıldı
RUN python3 -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/model_cache')"

# Uygulama dosyalarını kopyala
COPY main.py .
COPY modules/ ./modules/

# PDF dizinini oluştur
RUN mkdir -p pdf_docs storage/json storage/pdf

# GPU bellek optimizasyonu için ortam değişkenleri
ENV TRANSFORMERS_CACHE=/app/model_cache
ENV HF_HOME=/app/model_cache
ENV TORCH_HOME=/app/model_cache

# Uygulama portunu aç
EXPOSE 5000

# Uygulamayı başlat - tek worker kullanarak
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "600", "--preload", "main:app"] 