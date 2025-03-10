FROM python:3.10-slim

WORKDIR /app

# Gerekli paketleri yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Uygulama dosyalarını kopyala
COPY requirements.txt .
COPY main.py .
COPY modules/ ./modules/

# PDF dizinini oluştur
RUN mkdir -p pdf_docs storage/json storage/pdf

# Gerekli Python paketlerini yükle
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama portunu aç
EXPOSE 5000

# Uygulamayı başlat
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "main:app"] 