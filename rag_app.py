#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import colorlog
import torch
from typing import List, Dict, Any
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.indices.loading import load_index_from_storage
# Kullanılmayan importlar kaldırıldı
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
            
            # RTX 4090'lar güçlü olduğu için quantization'a gerek yok
            use_quantization = False
                
        except Exception as e:
            logger.warning(f"GPU yapılandırması yapılamadı: {str(e)}")
            use_quantization = True
    else:
        use_quantization = False
    
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
        
        # 8-bit quantization kullan
        if use_quantization and device == "cuda":
            logger.info("8-bit quantization kullanılıyor")
            model_kwargs["load_in_8bit"] = True
        
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
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir = "./embedding_cache"
    
    # Cache dizinini oluştur
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Sentence Transformers modelini manuel olarak yükle
        from sentence_transformers import SentenceTransformer
        
        # Model yükleme seçenekleri
        model_kwargs = {
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # Modeli yükle
        logger.info(f"Embedding modeli yükleniyor: {model_name}")
        
        # Modeli önce manuel olarak yükle
        model = SentenceTransformer(model_name, cache_folder=cache_dir, **model_kwargs)
        
        # HuggingFaceEmbedding'e modeli doğrudan geç
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            model=model
        )
        
        logger.info("Embedding modeli başarıyla yüklendi.")
        return embed_model
        
    except Exception as e:
        logger.error(f"Embedding modeli yüklenirken hata oluştu: {str(e)}")
        logger.info("Varsayılan embedding modeli kullanılıyor...")
        
        # Hata durumunda basit bir yapılandırma kullan
        embed_model = HuggingFaceEmbedding(
            model_name=model_name
        )
        
        return embed_model

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
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        logger.info(f"Mevcut indeks '{persist_dir}' konumundan yükleniyor...")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        logger.info("İndeks başarıyla yüklendi.")
        return index
    else:
        logger.info("Yeni JSON indeksi oluşturuluyor...")
        
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

# Ana uygulama
def main():
    logger.info("RAG uygulaması başlatılıyor...")
    
    # Modelleri yapılandır
    llm = setup_llm()
    embed_model = setup_embedding_model()
    
    # Global ayarları yapılandır
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # JSON verisini yükle
    json_file = "Book1.json"
    json_data = load_json_data(json_file)
    
    # JSON verilerini düzleştirilmiş liste olarak al
    json_rows = create_or_load_json_index(json_data)
    
    # JSONalyzeQueryEngine oluştur
    query_engine = JSONalyzeQueryEngine(
        list_of_dict=json_rows,
        llm=llm,
        verbose=True
    )
    
    logger.info("RAG uygulaması hazır. Sorularınızı sorun (çıkmak için 'q' veya 'exit' yazın):")
    
    # Komut satırı arayüzü
    while True:
        try:
            user_input = input("\nSoru: ")
            
            if user_input.lower() in ['q', 'exit', 'quit', 'çıkış']:
                logger.info("Uygulama kapatılıyor...")
                break
            
            if not user_input.strip():
                continue
            
            logger.info(f"Soru işleniyor: {user_input}")
            response = query_engine.query(user_input)
            
            print("\nCevap:")
            print(response)
        
        except KeyboardInterrupt:
            logger.info("Kullanıcı tarafından sonlandırıldı.")
            break
        except Exception as e:
            logger.error(f"Hata oluştu: {str(e)}")
            logger.exception("Hata detayları:")

if __name__ == "__main__":
    main() 