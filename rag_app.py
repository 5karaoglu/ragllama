#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import colorlog
import torch
import gc
from typing import List, Dict, Any, Optional
from pathlib import Path
from flask import Flask
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from bitsandbytes.nn import Linear4bit
from transformers import BitsAndBytesConfig

from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Yerel modülleri içe aktar
from pdf_processor import setup_pdf_query_engine
from db_processor import setup_db_query_engine
from prompts import get_system_prompt
from api import setup_routes

# Flask uygulaması
app = Flask(__name__)

# Global değişkenler
sql_database = None
global_llm = None
pdf_query_engine = None
llama_debug_handler = None

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
    logger.setLevel(logging.DEBUG)  # DEBUG seviyesine değiştirdik
    logger.addHandler(handler)
    
    # Diğer kütüphanelerin loglarını azalt
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logger

logger = setup_logging()

# LlamaDebugHandler kurulumu
def setup_debug_handler():
    global llama_debug_handler
    # Yeni bir LlamaDebugHandler oluştur
    llama_debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    # CallbackManager ile entegre et
    callback_manager = CallbackManager([llama_debug_handler])
    # Global Settings'e ata
    Settings.callback_manager = callback_manager
    logger.info("LlamaDebugHandler başarıyla kuruldu.")
    return llama_debug_handler

# Model yapılandırması
def setup_llm():
    logger.info("DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit modeli yapılandırılıyor...")
    
    # Model adı ve cache dizini
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit"
    cache_dir = "./model_cache"
    
    # Cache dizinini oluştur
    os.makedirs(cache_dir, exist_ok=True)
    
    # GPU kullanılabilirliğini kontrol et
    device = "cpu"
    try:
        if torch.cuda.is_available():
            device = "cuda"
            # CUDA bilgilerini logla
            logger.info(f"CUDA kullanılabilir: {torch.cuda.is_available()}")
            logger.info(f"CUDA sürümü: {torch.version.cuda}")
            logger.info(f"CUDA cihaz sayısı: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"CUDA cihaz {i}: {torch.cuda.get_device_name(i)}")
            
            # CUDA önbelleğini temizle
            torch.cuda.empty_cache()
            gc.collect()
            
            # CUDA ortam değişkenlerini kontrol et
            logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Ayarlanmamış')}")
            logger.info(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Ayarlanmamış')}")
        else:
            logger.warning("CUDA kullanılamıyor! CPU kullanılacak.")
            logger.warning("NVIDIA sürücülerini ve CUDA kurulumunu kontrol edin.")
    except Exception as e:
        logger.error(f"GPU kontrolü sırasında hata oluştu: {str(e)}")
        logger.warning("CPU kullanılacak.")
    
    logger.info(f"Cihaz: {device}")
    
    try:
        logger.info(f"Model yükleniyor: {model_name}")
        
        # 4-bit quantization yapılandırması
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Tokenizer'ı yükle
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            use_fast=False  # Fast tokenizer'ı devre dışı bırak
        )
        
        # Model'i yükle
        model_kwargs = {
            "device_map": "auto",
            "quantization_config": quantization_config,
            "trust_remote_code": True,
            "cache_dir": cache_dir,
            "torch_dtype": torch.float16  # float16 kullan
        }
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        logger.info("Model başarıyla yüklendi")
        logger.info(f"Model cihazı: {next(model.parameters()).device}")
        
        # HuggingFaceLLM oluştur, model ve tokenizer'ı doğrudan geç
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=8192,  # Daha uzun bağlam penceresi
            max_new_tokens=1024,  # Daha uzun yanıtlar için
            generate_kwargs={"temperature": 0.7, "do_sample": True, "top_p": 0.95}
        )
        
        return llm
        
    except Exception as e:
        logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
        raise

def setup_embedding_model():
    logger.info("Embedding modeli yapılandırılıyor...")
    
    try:
        # BGE-large-en-v1.5 modelini yapılandır
        model_name = "BAAI/bge-large-en-v1.5"
        cache_dir = "./embedding_cache"
        
        # Cache dizinini oluştur
        os.makedirs(cache_dir, exist_ok=True)
        
        # Önce SentenceTransformer ile modeli yükle
        st_model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        # Sonra HuggingFaceEmbedding ile sarmala
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=512,
            # BGE modelleri için query prefix ekleme
            query_instruction="Represent this sentence for searching relevant passages:"
        )
        
        logger.info("BGE-large-en-v1.5 embedding modeli başarıyla yüklendi.")
        return embed_model
    except Exception as e:
        logger.error(f"Embedding modeli yüklenirken hata oluştu: {str(e)}")
        raise

# Uygulama başlatma
def initialize_app():
    """Uygulamayı başlatır ve gerekli bileşenleri yükler."""
    global sql_database, global_llm, pdf_query_engine
    
    logger.info("RAG uygulaması başlatılıyor...")
    
    # Debug handler'ı kur
    setup_debug_handler()
    
    # Modelleri yapılandır
    llm = setup_llm()
    global_llm = llm
    embed_model = setup_embedding_model()
    
    # Global ayarları yapılandır
    Settings.embed_model = embed_model
    logger.info("Embedding modeli global ayarlara atandı.")
    
    # DB SQLDatabase nesnesini oluştur
    logger.info("DB SQLDatabase nesnesi oluşturuluyor...")
    sql_database = setup_db_query_engine("Book1.json")
    logger.info("DB SQLDatabase nesnesi başarıyla oluşturuldu.")
    
    # PDF sorgu motorunu oluştur
    logger.info("PDF sorgu motoru oluşturuluyor...")
    pdf_query_engine = setup_pdf_query_engine("document.pdf", global_llm, get_system_prompt('pdf'))
    logger.info("PDF sorgu motoru başarıyla oluşturuldu.")
    
    # API rotalarını ayarla
    setup_routes(app, pdf_query_engine, llama_debug_handler)
    logger.info("API Rotaları ayarlandı.")
    
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