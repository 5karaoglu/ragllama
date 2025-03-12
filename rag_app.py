#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import colorlog
import torch
from typing import List, Dict, Any, Tuple
from pathlib import Path
from flask import Flask, request, jsonify
import PyPDF2
import faiss
import numpy as np

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Document
)
from llama_index.core.indices.loading import load_index_from_storage
from llama_index.core.query_engine import JSONalyzeQueryEngine, RetrieverQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

# Flask uygulaması
app = Flask(__name__)

# Global değişkenler
db_query_engine = None
pdf_query_engine = None

# Loglama yapılandırması
def setup_logging():
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    # Diğer kütüphanelerin loglarını azalt
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

# Model yapılandırması
def setup_llm():
    logger.info("DeepSeek modeli yapılandırılıyor...")
    
    # DeepSeek modelini kullanmaya devam ediyoruz
    model_name = "deepseek-ai/deepseek-llm-7b-chat"
    cache_dir = "./model_cache"
    
    # Cache dizinini oluştur
    os.makedirs(cache_dir, exist_ok=True)
    
    # GPU kullanılabilirliğini kontrol et
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Cihaz: {device}")
    
    # RTX 4090 GPU'lar için bellek optimizasyonu
    if device == "cuda":
        try:
            # CUDA önbelleğini temizle
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
            # GPU sayısını kontrol et
            gpu_count = torch.cuda.device_count()
            logger.info(f"Kullanılabilir GPU sayısı: {gpu_count}")
            
        except Exception as e:
            logger.warning(f"GPU yapılandırması yapılamadı: {str(e)}")
    
    # Önce model ve tokenizer'ı manuel olarak yükleyelim
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    try:
        logger.info(f"Model yükleniyor: {model_name}")
        
        # Tokenizer'ı yükle
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # Model'i yükle
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "low_cpu_mem_usage": True,
            "cache_dir": cache_dir
        }
        
        # device_map'i sadece burada kullan
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        logger.info("Model başarıyla yüklendi")
        
        # HuggingFaceLLM oluştur, model ve tokenizer'ı doğrudan geç
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=4096,
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.7, "do_sample": True}
        )
        
        return llm
        
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
        logger.error("Daha küçük bir model kullanmaya çalışılıyor...")
        
        # Daha küçük bir model dene
        fallback_model = "google/flan-t5-base"
        logger.info(f"Yedek model yükleniyor: {fallback_model}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            fallback_model,
            cache_dir=cache_dir
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            fallback_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            cache_dir=cache_dir,
            device_map="auto" if device == "cuda" else None
        )
        
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.7, "do_sample": True}
        )
        
        return llm

def setup_embedding_model():
    logger.info("Embedding modeli yapılandırılıyor...")
    
    # Basit bir HuggingFaceEmbedding kullan
    try:
        # safe_serialization parametresini kaldırarak HuggingFaceEmbedding'i yapılandır
        from transformers import AutoModel
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Modeli manuel olarak yükle, safe_serialization parametresi olmadan
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            model=model,
            max_length=512
        )
        logger.info("Embedding modeli başarıyla yüklendi.")
        return embed_model
    except Exception as e:
        logger.error(f"Embedding modeli yüklenirken hata oluştu: {str(e)}")
        logger.warning("Varsayılan embedding modeli kullanılacak.")
        return None

# JSON veri yükleme
def load_json_data(file_path: str) -> Dict[str, Any]:
    logger.info(f"'{file_path}' dosyası yükleniyor...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON verisi başarıyla yüklendi. {len(data)} sayfa bulundu.")
        return data
    
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {str(e)}")
        raise

# JSON indeksi oluşturma veya yükleme
def create_or_load_json_index(json_data: Dict[str, Any], persist_dir: str = "./storage"):
    # Dizin varsa yükle, yoksa oluştur
    if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "docstore.json")):
        logger.info(f"Mevcut indeks '{persist_dir}' konumundan yükleniyor...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info("İndeks başarıyla yüklendi.")
            return index
        except Exception as e:
            logger.error(f"İndeks yüklenirken hata oluştu: {str(e)}")
            logger.info("Yeni indeks oluşturuluyor...")
            # Hata durumunda yeni indeks oluştur
            return create_new_json_index(json_data, persist_dir)
    else:
        logger.info("Yeni JSON indeksi oluşturuluyor...")
        return create_new_json_index(json_data, persist_dir)

# Yeni JSON indeksi oluşturma
def create_new_json_index(json_data: Dict[str, Any], persist_dir: str):
    # JSON verilerini düzleştir - tüm satırları tek bir listede topla
    json_rows = []
    
    for sheet_name, rows in json_data.items():
        logger.info(f"'{sheet_name}' sayfası işleniyor, {len(rows)} satır bulundu.")
        
        for i, row in enumerate(rows):
            # Her satıra sayfa bilgisini ekle
            row_with_metadata = row.copy()
            row_with_metadata["_sheet"] = sheet_name
            row_with_metadata["_row_index"] = i
            row_with_metadata["_row_id"] = f"{sheet_name}_{i}"
            json_rows.append(row_with_metadata)
    
    logger.info(f"Toplam {len(json_rows)} satır işlendi.")
    
    # JSON indeksini oluştur ve kaydet
    os.makedirs(persist_dir, exist_ok=True)
    
    # JSON satırlarını dosyaya kaydet
    json_file_path = os.path.join(persist_dir, "json_rows.json")
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)
    
    logger.info(f"JSON satırları '{json_file_path}' konumuna kaydedildi.")
    
    return json_rows

# PDF işleme fonksiyonları
def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """PDF dosyasından metin çıkartır."""
    logger.info(f"'{file_path}' dosyasından metin çıkartılıyor...")
    text_chunks = []
    
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():  # Boş sayfaları atla
                    # Sayfa numarasını metne ekle
                    chunk = {
                        "content": text,
                        "metadata": {
                            "page": i + 1,
                            "source": file_path
                        }
                    }
                    text_chunks.append(chunk)
        
        logger.info(f"PDF dosyasından {len(text_chunks)} sayfa metni çıkartıldı.")
        return text_chunks
    
    except Exception as e:
        logger.error(f"PDF metni çıkartılırken hata oluştu: {str(e)}")
        raise

def create_or_load_pdf_index(pdf_file: str, persist_dir: str = "./pdf_storage") -> VectorStoreIndex:
    """PDF verilerini işleyip indeks oluşturma veya yükleme işlemleri."""
    # Dizin varsa yükle, yoksa oluştur
    if os.path.exists(persist_dir) and os.path.exists(os.path.join(persist_dir, "docstore.json")):
        logger.info(f"Mevcut PDF indeksi '{persist_dir}' konumundan yükleniyor...")
        try:
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
            logger.info("PDF indeksi başarıyla yüklendi.")
            return index
        except Exception as e:
            logger.error(f"PDF indeksi yüklenirken hata oluştu: {str(e)}")
            logger.info("Yeni PDF indeksi oluşturuluyor...")
            # Hata durumunda yeni indeks oluştur
            return create_new_pdf_index(pdf_file, persist_dir)
    else:
        logger.info("Yeni PDF indeksi oluşturuluyor...")
        return create_new_pdf_index(pdf_file, persist_dir)

def create_new_pdf_index(pdf_file: str, persist_dir: str) -> VectorStoreIndex:
    """Yeni bir PDF indeksi oluşturur."""
    # PDF dosyasından metin çıkart
    text_chunks = extract_text_from_pdf(pdf_file)
    
    # Belgeleri oluştur
    documents = []
    for chunk in text_chunks:
        doc = Document(
            text=chunk["content"],
            metadata=chunk["metadata"]
        )
        documents.append(doc)
    
    logger.info(f"PDF'den {len(documents)} belge oluşturuldu.")
    
    # Vektör indeksi oluştur
    index = VectorStoreIndex.from_documents(documents)
    
    # İndeksi kaydet
    os.makedirs(persist_dir, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    
    logger.info(f"PDF indeksi '{persist_dir}' konumuna kaydedildi.")
    return index

# API endpoint'leri
@app.route('/api/status', methods=['GET'])
def status():
    """API durumunu kontrol eder."""
    return jsonify({
        "status": "online",
        "db_query_engine_ready": db_query_engine is not None,
        "pdf_query_engine_ready": pdf_query_engine is not None
    })

@app.route('/api/query', methods=['POST'])
def query():
    """Kullanıcı sorgusunu işler ve yanıt döndürür."""
    global db_query_engine, pdf_query_engine
    
    try:
        data = request.json
        user_query = data.get('query')
        module = data.get('module', 'db')  # Varsayılan olarak 'db' kullan
        
        if not user_query:
            return jsonify({"error": "Sorgu parametresi gerekli"}), 400
        
        # Modüle göre sorguyu işle
        if module == 'db':
            if db_query_engine is None:
                return jsonify({"error": "DB query engine henüz hazır değil"}), 503
                
            logger.info(f"DB modülü ile soru işleniyor: {user_query}")
            response = db_query_engine.query(user_query)
        elif module == 'pdf':
            if pdf_query_engine is None:
                return jsonify({"error": "PDF query engine henüz hazır değil"}), 503
                
            logger.info(f"PDF modülü ile soru işleniyor: {user_query}")
            response = pdf_query_engine.query(user_query)
        else:
            return jsonify({"error": "Geçersiz modül parametresi"}), 400
        
        return jsonify({
            "query": user_query,
            "module": module,
            "response": str(response)
        })
    
    except Exception as e:
        logger.error(f"Sorgu işlenirken hata oluştu: {str(e)}")
        logger.exception("Hata detayları:")
        return jsonify({"error": str(e)}), 500

# Uygulama başlatma
def initialize_app():
    """Uygulamayı başlatır ve gerekli bileşenleri yükler."""
    global db_query_engine, pdf_query_engine
    
    logger.info("RAG uygulaması başlatılıyor...")
    
    # Modelleri yapılandır
    llm = setup_llm()
    embed_model = setup_embedding_model()
    
    # Global ayarları yapılandır
    Settings.llm = llm
    if embed_model:
        Settings.embed_model = embed_model
    
    # JSON verisini yükle ve işle
    try:
        json_file = "Book1.json"
        json_data = load_json_data(json_file)
        
        # Önce storage dizinini temizle (isteğe bağlı)
        storage_dir = "./storage"
        if os.path.exists(storage_dir):
            import shutil
            try:
                logger.info(f"Eski storage dizini temizleniyor: {storage_dir}")
                shutil.rmtree(storage_dir)
                logger.info("Storage dizini temizlendi.")
            except Exception as e:
                logger.warning(f"Storage dizini temizlenirken hata oluştu: {str(e)}")
        
        # JSON verilerini düzleştirilmiş liste olarak al
        json_rows = create_or_load_json_index(json_data)
        
        # JSONalyzeQueryEngine oluştur
        db_query_engine = JSONalyzeQueryEngine(
            list_of_dict=json_rows,
            llm=llm,
            verbose=True
        )
        
        logger.info("DB modülü başarıyla yüklendi.")
    except Exception as e:
        logger.error(f"DB modülü yüklenirken hata oluştu: {str(e)}")
        logger.warning("DB modülü atlanıyor.")
    
    # PDF verisini yükle ve işle
    try:
        pdf_file = "document.pdf"  # PDF dosyasının adını buraya yazın
        
        # PDF dosyasının varlığını kontrol et
        if os.path.exists(pdf_file):
            # PDF indeksini oluştur veya yükle
            pdf_index = create_or_load_pdf_index(pdf_file)
            
            # PDF sorgu motorunu oluştur
            retriever = VectorIndexRetriever(
                index=pdf_index,
                similarity_top_k=3
            )
            
            pdf_query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=llm
            )
            
            logger.info("PDF modülü başarıyla yüklendi.")
        else:
            logger.warning(f"PDF dosyası bulunamadı: {pdf_file}")
            logger.warning("PDF modülü atlanıyor.")
    except Exception as e:
        logger.error(f"PDF modülü yüklenirken hata oluştu: {str(e)}")
        logger.warning("PDF modülü atlanıyor.")
    
    logger.info("RAG uygulaması başarıyla başlatıldı ve API hazır.")
    return True

if __name__ == "__main__":
    # Uygulamayı başlat
    success = initialize_app()
    
    if success:
        # Flask uygulamasını başlat
        logger.info("API sunucusu başlatılıyor...")
        app.run(host='0.0.0.0', port=5000)
    else:
        logger.error("Uygulama başlatılamadı") 