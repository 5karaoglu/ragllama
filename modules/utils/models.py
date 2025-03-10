#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import torch
from typing import Optional

from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

def setup_llm():
    """LLM modelini yapılandırır ve döndürür."""
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
    """Embedding modelini yapılandırır ve döndürür."""
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

def configure_settings():
    """Global LlamaIndex ayarlarını yapılandırır."""
    llm = setup_llm()
    embed_model = setup_embedding_model()
    
    # Global ayarları yapılandır
    Settings.llm = llm
    if embed_model:
        Settings.embed_model = embed_model
    
    return llm, embed_model 