#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import colorlog
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
from flask import Flask, request, jsonify
import PyPDF2
import re
import sqlite3

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
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

# Flask uygulaması
app = Flask(__name__)

# Global değişkenler
db_query_engine = None
pdf_query_engine = None
llama_debug_handler = None

# Sistem promptları
SYSTEM_PROMPT = """
Bu bir RAG (Retrieval-Augmented Generation) sistemidir. Lütfen aşağıdaki kurallara uygun yanıtlar verin:

1. Her zaman Türkçe yanıt verin.
2. Yalnızca verilen belgelerden elde edilen bilgilere dayanarak yanıt verin.
3. Eğer yanıt verilen belgelerde bulunmuyorsa, "Bu konuda belgelerde yeterli bilgi bulamadım" deyin.
4. Kişisel görüş veya yorum eklemeyin.
5. Verilen konunun dışına çıkmayın.
6. Yanıtlarınızı kapsamlı, detaylı ve anlaşılır tutun.
7. Emin olmadığınız bilgileri paylaşmayın.
8. Belgelerdeki bilgileri çarpıtmadan, doğru şekilde aktarın.
9. Yanıtlarınızı yapılandırırken, önemli bilgileri vurgulayın ve gerektiğinde maddeler halinde sunun.
10. Teknik terimleri açıklayın ve gerektiğinde örnekler verin.

ÖNEMLİ - SQL SORGULARI İÇİN KRİTİK KURALLAR:
1. SQLite sözdizimi kurallarına kesinlikle uyun.
2. SQL sorgularınızda HİÇBİR ŞEKİLDE '#' karakterini KULLANMAYIN! 
3. SQL sorgularınıza KESİNLİKLE yorum satırı EKLEMEYİN!
4. Gerekiyorsa yorum SADECE '--' (iki tire) ile başlamalıdır.
5. Örnek: SELECT * FROM users WHERE id = 1; -- bu şekilde.
6. SQL sorgularını mümkün olduğunca basit tutun.
7. SQL sorgularını tek bir ifade olarak yazın.
8. SQL sorgusunu doğrudan çalıştırılabilir formatta döndürün - açıklama olmadan.
9. SQL sorgusu gönderilirken ASLA açıklama yazmayın - sadece SQL kodunu gönderin.
10. DİKKAT: 'selecting the specific column' gibi doğal dil ifadeleri değil, 'SELECT column_name FROM table' gibi SQL kodları yazın!
11. SQL sorgusu oluşturma düşünce sürecinizi dökümanlara yansıtmayın, SADECE nihai SQL kodunu yazın.
12. Düşünme sürecinizi tamamlayın ve SADECE çalıştırmaya hazır SQL kodunu döndürün.
13. Eğer sorgu oluşturmakta zorlanırsanız, verilere doğrudan bakıp analiz yapın.

ÖRNEKLER:
DOĞRU: SELECT * FROM users WHERE name = 'Ali';
YANLIŞ: selecting the specific column which is name = 'Ali'
YANLIŞ: SELECT * FROM users WHERE name = 'Ali'; # Bu Ali'yi bulan sorgu
YANLIŞ: İşte Ali'yi bulmak için bir sorgu yazıyorum: SELECT * FROM users WHERE name = 'Ali';

Göreviniz, kullanıcının sorularını belgelerden elde ettiğiniz bilgilerle detaylı ve doğru bir şekilde yanıtlamaktır.
"""

# Özel SQL filtre fonksiyonu
def filter_llm_response_for_sql(llm_response: str) -> str:
    """
    LLM'in ürettiği yanıttan SQL sorgusunu ayıklar.
    
    Args:
        llm_response: LLM'in ürettiği yanıt
        
    Returns:
        Temizlenmiş SQL sorgusu
    """
    try:
        logger.info(f"LLM YANIT FİLTRESİ GİRİŞİ: {llm_response}")
        
        # Markdown kod bloklarını ara
        sql_code_pattern = r"```sql(.*?)```"
        sql_code_matches = re.findall(sql_code_pattern, llm_response, re.DOTALL)
        
        if sql_code_matches:
            # Markdown kod bloğu içindeki SQL'i al
            sql_query = sql_code_matches[0].strip()
            logger.info(f"Markdown kod bloğundan SQL sorgusu ayıklandı: {sql_query}")
            return sql_query
        
        # Alternatif: Markdown kod bloğunu farklı formatta ara
        general_code_pattern = r"```(.*?)```"
        code_matches = re.findall(general_code_pattern, llm_response, re.DOTALL)
        
        if code_matches:
            for code_block in code_matches:
                # Kod bloğunun içeriğinden SELECT ifadesi içeren bir şey var mı?
                if re.search(r'(?i)SELECT\s+', code_block):
                    sql_query = code_block.strip()
                    if not sql_query.endswith(";"):
                        sql_query += ";"
                    logger.info(f"Genel kod bloğundan SQL sorgusu ayıklandı: {sql_query}")
                    return sql_query
        
        # Alternatif: Markdown blok olmadan SELECT ifadesini ara
        select_pattern = r"(?i)SELECT\s+.*?(?:;|$)"
        select_matches = re.findall(select_pattern, llm_response, re.DOTALL)
        
        if select_matches:
            # SELECT ile başlayan ilk sorguyu al
            sql_query = select_matches[0].strip()
            if not sql_query.endswith(";"):
                sql_query += ";"
            logger.info(f"SELECT deseninden SQL sorgusu ayıklandı: {sql_query}")
            return sql_query
        
        # Tüm yorum satırlarını kaldır
        cleaned_sql = re.sub(r"#.*", "", llm_response)  # # ile başlayan yorumları kaldır
        cleaned_sql = re.sub(r"--.*", "", cleaned_sql)  # -- ile başlayan yorumları kaldır
        cleaned_sql = re.sub(r"/\*.*?\*/", "", cleaned_sql, flags=re.DOTALL)  # /* */ yorumlarını kaldır
        
        # Boşlukları kırp
        cleaned_sql = cleaned_sql.strip()
        
        # SELECT ile başlayan kısmı bul
        if "SELECT" in cleaned_sql.upper():
            sql_query = cleaned_sql[cleaned_sql.upper().find("SELECT"):]
            sql_query = sql_query.strip()
            if not sql_query.endswith(";"):
                sql_query += ";"
            logger.info(f"Temizlenmiş metinden SQL sorgusu ayıklandı: {sql_query}")
            return sql_query
        
        # Hiçbir şey bulunamazsa güvenli bir sorgu döndür
        logger.warning("SQL sorgusu bulunamadı, güvenli sorgu döndürülüyor")
        return "SELECT * FROM table_data LIMIT 5;"
    
    except Exception as e:
        logger.error(f"SQL filtreleme hatası: {str(e)}")
        return "SELECT * FROM table_data LIMIT 5;"

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
    logger.info("DeepSeek-R1-Distill-Qwen-14B modeli yapılandırılıyor...")
    
    # DeepSeek-R1-Distill-Qwen-14B modelini kullanacağız
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
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
            import gc
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
    
    # vLLM kullanarak daha hızlı ve bellek verimli çıkarım yapmayı deneyelim
    try:
        logger.info("vLLM ile model yükleme deneniyor...")
        
        # LLM tipi için çevre değişkenini kontrol et
        llm_type = os.environ.get("LLM_TYPE", "vllm").lower()
        
        if llm_type == "vllm" and device == "cuda":
            logger.info("vLLM kullanılarak model yükleniyor...")
            
            try:
                # vLLM entegrasyonunu LlamaIndex üzerinden yükleyelim
                from llama_index.llms.vllm import VLLMLangChainCompatibility

                # vLLM'i yapılandır - PagedAttention özelliğinden faydalanır
                from vllm import LLM as VLLM
                vllm_model = VLLM(
                    model=model_name,
                    tensor_parallel_size=torch.cuda.device_count(),  # Tüm GPU'ları kullan
                    dtype="float16" if device == "cuda" else "float32",
                    trust_remote_code=True,
                    max_model_len=8192,  # Maksimum bağlam penceresi
                    download_dir=cache_dir,
                    gpu_memory_utilization=0.85,  # GPU belleği kullanım oranı
                    enforce_eager=False,  # Daha yüksek verimlilik için eager modu kapatın
                    enable_lora=False,  # LoRA desteğini devre dışı bırak
                )
                
                # LlamaIndex uyumlu wrapper oluştur
                llm = VLLMLangChainCompatibility(
                    client=vllm_model,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    streaming=False
                )
                
                logger.info("vLLM ile model başarıyla yüklendi!")
                return llm
                
            except ImportError as ie:
                logger.error(f"vLLM yüklenirken ImportError: {str(ie)}")
                logger.warning("vLLM yüklenemedi, klasik HuggingFace modeline geçiliyor...")
            except Exception as ve:
                logger.error(f"vLLM model yükleme hatası: {str(ve)}")
                logger.warning("vLLM hatası nedeniyle klasik HuggingFace modeline geçiliyor...")
        else:
            if llm_type != "vllm":
                logger.info(f"LLM_TYPE={llm_type} olarak ayarlandı, HuggingFace kullanılıyor...")
            else:
                logger.info("CUDA kullanılamadığı için vLLM atlanıyor, HuggingFace kullanılıyor...")
                
        # Klasik HuggingFace modelini yükle (vLLM başarısız olursa veya istenmezse)
        logger.info("Klasik HuggingFace modeli yükleniyor...")
        
        # Önce model ve tokenizer'ı manuel olarak yükleyelim
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
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
            logger.info("GPU kullanılacak: device_map=auto")
        else:
            logger.warning("GPU kullanılamıyor, CPU kullanılacak!")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        logger.info("HuggingFace modeli başarıyla yüklendi")
        logger.info(f"Model cihazı: {next(model.parameters()).device}")
        
        # HuggingFaceLLM oluştur, model ve tokenizer'ı doğrudan geç
        from llama_index.llms.huggingface import HuggingFaceLLM
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
        logger.error("Daha küçük bir model kullanmaya çalışılıyor...")
        
        # Daha küçük bir model dene
        fallback_model = "deepseek-ai/deepseek-llm-7b-chat"
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
        
        from llama_index.llms.huggingface import HuggingFaceLLM
        llm = HuggingFaceLLM(
            model=model,
            tokenizer=tokenizer,
            context_window=4096,
            max_new_tokens=512,
            generate_kwargs={"temperature": 0.7, "do_sample": True, "top_p": 0.95}
        )
        
        return llm

def setup_embedding_model():
    logger.info("Embedding modeli yapılandırılıyor...")
    
    try:
        # BGE-large-en-v1.5 modelini yapılandır
        model_name = "BAAI/bge-large-en-v1.5"
        cache_dir = "./embedding_cache"
        
        # Cache dizinini oluştur
        os.makedirs(cache_dir, exist_ok=True)
        
        # Sentence-transformers kullanarak modeli yükle
        from sentence_transformers import SentenceTransformer
        
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
        logger.warning("Daha basit bir embedding modeli kullanmaya çalışılıyor...")
        
        try:
            # Daha basit bir model dene
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            embed_model = HuggingFaceEmbedding(
                model_name=fallback_model,
                max_length=512
            )
            logger.info(f"Yedek embedding modeli {fallback_model} başarıyla yüklendi.")
            return embed_model
        except Exception as e:
            logger.error(f"Yedek embedding modeli yüklenirken hata oluştu: {str(e)}")
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
    # JSON verilerini düzleştirilmiş liste olarak al
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
    
    # Vektör indeksi oluştur - embed_model parametresini açıkça belirt
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model  # Global embed_model'i kullan
    )
    
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

@app.route('/api/system-info', methods=['GET'])
def system_info():
    """Sistem durumu ve LLM konfigürasyonu hakkında detaylı bilgi sağlar."""
    try:
        # Kullanılan LLM tipini belirle
        llm_type = os.environ.get("LLM_TYPE", "vllm").lower()
        is_vllm = "vllm" in llm_type and hasattr(Settings.llm, "client")
        
        # Bellek kullanımını hesapla
        gpu_info = []
        total_memory_used = 0
        total_memory_available = 0
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_used = torch.cuda.memory_allocated(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                gpu_info.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_used_gb": round(memory_used, 2),
                    "memory_total_gb": round(memory_total, 2),
                    "utilization_percent": round((memory_used / memory_total) * 100, 2)
                })
                
                total_memory_used += memory_used
                total_memory_available += memory_total
        
        # vLLM konfigürasyonu (eğer kullanılıyorsa)
        vllm_config = None
        if is_vllm:
            vllm_config = {
                "paged_attention": os.environ.get("VLLM_USE_PAGED_ATTENTION", "true") == "true",
                "tensor_parallel_size": int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "1")),
                "gpu_memory_utilization": float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
                "max_parallel_loading_workers": int(os.environ.get("VLLM_MAX_PARALLEL_LOADING_WORKERS", "2")),
                "attention_shard_size": int(os.environ.get("VLLM_ATTENTION_SHARD_SIZE", "1024"))
            }
        
        # Yanıt hazırla
        response = {
            "status": "online",
            "llm_type": "vllm" if is_vllm else "huggingface",
            "llm_model": os.environ.get("LLM_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
            "api_modules": {
                "db_query_engine_ready": db_query_engine is not None,
                "pdf_query_engine_ready": pdf_query_engine is not None
            },
            "gpu": {
                "available": torch.cuda.is_available(),
                "count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "devices": gpu_info,
                "total_memory_used_gb": round(total_memory_used, 2),
                "total_memory_available_gb": round(total_memory_available, 2),
                "overall_utilization_percent": round((total_memory_used / total_memory_available) * 100, 2) if total_memory_available > 0 else 0
            }
        }
        
        # vLLM konfigürasyonunu ekle (eğer varsa)
        if vllm_config:
            response["vllm_config"] = vllm_config
            
            # vLLM performans iyileştirmeleri hakkında bilgi ver
            response["vllm_optimizations"] = {
                "paged_attention": "KV cache bellek parçalanmasını azaltır ve bellek verimliliğini artırır.",
                "continuous_batching": "Farklı uzunluktaki sorguları dinamik olarak gruplar ve GPU kullanımını optimize eder.",
                "tensor_parallelism": "Büyük modelleri birden fazla GPU'ya dağıtarak bellek sınırlamalarını aşar.",
                "estimated_speedup": "Geleneksel HuggingFace'e göre 3-24x arası hızlanma sağlar."
            }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Sistem bilgisi oluşturulurken hata: {str(e)}")
        logger.exception("Hata detayları:")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def query():
    """Kullanıcı sorgusunu işler ve yanıt döndürür."""
    global db_query_engine, pdf_query_engine, llama_debug_handler
    
    try:
        data = request.json
        user_query = data.get('query')
        module = data.get('module', 'db')  # Varsayılan olarak 'db' kullan
        show_thoughts = data.get('show_thoughts', True)  # Düşünce sürecini gösterme seçeneği
        
        if not user_query:
            return jsonify({"error": "Sorgu parametresi gerekli"}), 400
        
        # Sorgu öncesi event loglarını temizle
        if llama_debug_handler:
            llama_debug_handler.flush_event_logs()
            logger.info("Sorgu öncesi debug handler event logları temizlendi.")
        
        # Modüle göre sorguyu işle
        thought_process = []
        llm_response = None
        
        if module == 'db':
            if db_query_engine is None:
                return jsonify({"error": "DB query engine henüz hazır değil"}), 503
                
            logger.info(f"DB modülü ile soru işleniyor: {user_query}")
            
            # LLM sorgu takibini etkinleştir
            from llama_index.core.callbacks import CallbackManager
            import types
            
            # JSONalyzeQueryEngine sorgularını gözlemleme
            original_query = db_query_engine.query
            
            def logging_query_wrapper(self, query_str, **kwargs):
                logger.info(f"LLM'e gönderilen sorgu: {query_str}")
                response = original_query(query_str, **kwargs)
                
                # SQL kodunu ve LLM'in düşüncelerini logla
                try:
                    if hasattr(response, 'metadata') and response.metadata is not None:
                        if 'sql_query' in response.metadata:
                            logger.info(f"ÜRETİLEN SQL SORGUSU: {response.metadata['sql_query']}")
                        if 'result' in response.metadata:
                            logger.info(f"SQL SORGU SONUCU: {response.metadata['result']}")
                except Exception as log_error:
                    logger.error(f"Yanıt log hatası: {str(log_error)}")
                
                return response
            
            # Orijinal query fonksiyonunu geçici olarak değiştir
            db_query_engine.query = types.MethodType(logging_query_wrapper, db_query_engine)
            
            # Sorguyu çalıştır
            response = db_query_engine.query(user_query)
            llm_response = str(response)
            
            # Fonksiyonu eski haline getir
            db_query_engine.query = original_query
            
        elif module == 'pdf':
            if pdf_query_engine is None:
                return jsonify({"error": "PDF query engine henüz hazır değil"}), 503
                
            logger.info(f"PDF modülü ile soru işleniyor: {user_query}")
            
            response = pdf_query_engine.query(user_query)
            llm_response = str(response)
            
        else:
            return jsonify({"error": "Geçersiz modül parametresi"}), 400
        
        # LLM giriş/çıkışlarını ve düşünce sürecini topla
        if llama_debug_handler and show_thoughts:
            event_pairs = llama_debug_handler.get_llm_inputs_outputs()
            if event_pairs:
                for i, (start_event, end_event) in enumerate(event_pairs):
                    logger.info(f"LLM Çağrısı #{i+1}:")
                    
                    # Çıkış yanıtındaki düşünme sürecini al
                    if 'response' in end_event.payload:
                        response_text = end_event.payload['response']
                        # Cevabı olduğu gibi alalım ama paragrafları ayıralım
                        thought_process.append({
                            "step": i+1,
                            "thought": response_text
                        })
                        logger.info(f"Çıkış yanıtı: {response_text}")
                    
                    # Girdi mesajlarını da alabilirsiniz (isteğe bağlı)
                    if 'messages' in start_event.payload:
                        messages = start_event.payload['messages']
                        for msg in messages:
                            if msg.get('role') == 'system' or msg.get('role') == 'user':
                                logger.info(f"Giriş mesajı ({msg.get('role')}): {msg.get('content')}")
            
            # İşlem bittikten sonra event loglarını temizle
            llama_debug_handler.flush_event_logs()
            logger.info("Sorgu sonrası debug handler event logları temizlendi.")
        
        # Yanıtı ve düşünce sürecini içeren JSON'ı döndür
        response_data = {
            "query": user_query,
            "module": module,
            "response": llm_response
        }
        
        # Düşünce sürecini ekle (istenirse)
        if show_thoughts and thought_process:
            response_data["thoughts"] = thought_process
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Sorgu işlenirken hata oluştu: {str(e)}")
        logger.exception("Hata detayları:")
        return jsonify({"error": str(e)}), 500

# Uygulama başlatma
def initialize_app():
    """Uygulamayı başlatır ve gerekli bileşenleri yükler."""
    global db_query_engine, pdf_query_engine
    
    logger.info("RAG uygulaması başlatılıyor...")
    
    # LLM tipi için çevre değişkenini kontrol et
    llm_type = os.environ.get("LLM_TYPE", "vllm").lower()
    logger.info(f"Kullanılacak LLM tipi: {llm_type}")
    
    # Debug handler'ı kur
    setup_debug_handler()
    
    # Modelleri yapılandır
    llm = setup_llm()
    embed_model = setup_embedding_model()
    
    # LLM tipi ve konfigürasyon bilgilerini logla
    try:
        llm_info = f"LLM tipi: {type(llm).__name__}"
        if "vllm" in llm_type and hasattr(llm, "client"):
            # vLLM konfigürasyon bilgilerini logla
            llm_info += f" (vLLM, Tensor Paralel Boyutu: {os.environ.get('VLLM_TENSOR_PARALLEL_SIZE', 'Tanımlanmamış')})"
            logger.info(f"vLLM başarıyla yapılandırıldı, PagedAttention etkin: {os.environ.get('VLLM_USE_PAGED_ATTENTION', 'Tanımlanmamış')}")
            logger.info(f"vLLM GPU Bellek Kullanım Oranı: {os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', 'Tanımlanmamış')}")
        else:
            llm_info += " (HuggingFace)"
        logger.info(llm_info)
    except Exception as e:
        logger.error(f"LLM bilgilerini loglarken hata: {str(e)}")
    
    # Global ayarları yapılandır
    Settings.llm = llm
    if embed_model:
        Settings.embed_model = embed_model
        logger.info("Embedding modeli global ayarlara atandı.")
    else:
        logger.warning("Embedding modeli bulunamadı, varsayılan model kullanılacak.")
        # Varsayılan olarak local embedding modeli kullan
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Varsayılan embedding modeli (all-MiniLM-L6-v2) global ayarlara atandı.")
    
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
        try:
            # SQL Parser'ı devre dışı bırak ve kendi filtreleme fonksiyonumuzu ekle
            logger.info("JSONalyzeQueryEngine özel filtre ile oluşturuluyor")
            
            # JSONalyzeQueryEngine oluştur
            db_query_engine = JSONalyzeQueryEngine(
                list_of_dict=json_rows,
                llm=llm,
                verbose=True,
                system_prompt=SYSTEM_PROMPT,
                synthesize_response=True,  # SQL sorgusu çalıştırılsa bile yanıtı sentezle
                sql_optimizer=True,  # SQL sorgularını optimize et
                sql_parser=None,  # SQL Parser'ı devre dışı bırak
                infer_schema=True,  # Şema çıkarımını etkinleştir
                enforce_sql_syntax=True,  # SQL sözdizimi kontrolünü zorla
                output_direct_sql=True,  # LLM'in doğrudan SQL sorgusu döndürmesini sağla
                allow_multiple_queries=False  # Birden fazla sorguya izin verme
            )
            
            # Monkey patching yoluyla JSONalyzeQueryEngine'in _analyzer metodunu düzenleyelim
            # Bu, sorguyu execute etmeden önce filtrelemek için kullanılacak
            original_analyzer = db_query_engine._analyzer
            
            def filtered_analyzer(*args, **kwargs):
                try:
                    sql_query, table_schema, results = original_analyzer(*args, **kwargs)
                    # Şimdi SQL sorgusunu filtreleme fonksiyonumuzdan geçirelim
                    if sql_query and isinstance(sql_query, str):
                        filtered_sql = filter_llm_response_for_sql(sql_query)
                        logger.info(f"ORİJİNAL SQL: {sql_query}")
                        logger.info(f"FİLTRELENMİŞ SQL: {filtered_sql}")
                        return filtered_sql, table_schema, results
                    return sql_query, table_schema, results
                except Exception as e:
                    logger.error(f"Analyzer düzeltme hatası: {str(e)}")
                    return args[0], None, []
            
            # Analyzer metodunu düzenlenmiş versiyonuyla değiştirelim
            db_query_engine._analyzer = filtered_analyzer
            
            logger.info("DB modülü başarıyla yüklendi. Özel SQL filtreleme etkin.")
        except Exception as sql_error:
            logger.error(f"JSONalyzeQueryEngine oluşturulurken SQL hatası: {str(sql_error)}")
            logger.exception("SQL hata detayları:")
            logger.warning("JSONalyzeQueryEngine oluşturulurken hata, varsayılan yapılandırma deneniyor...")
            
            # Hata durumunda daha basit yapılandırmayı dene
            try:
                logger.info("Basit yapılandırma ile JSONalyzeQueryEngine oluşturuluyor")
                
                db_query_engine = JSONalyzeQueryEngine(
                    list_of_dict=json_rows,
                    llm=llm,
                    verbose=True,
                    system_prompt=SYSTEM_PROMPT,
                    infer_schema=False,  # Şema çıkarımını devre dışı bırak
                    sql_parser=None,  # SQL Parser'ı devre dışı bırak
                    output_direct_sql=True,  # LLM'in doğrudan SQL sorgusu döndürmesini sağla
                    enforce_sql_syntax=True  # SQL sözdizimi kontrolünü zorla
                )
                
                # Basitleştirilmiş filtrelemeyi yine de ekleyelim
                original_analyzer = db_query_engine._analyzer
                
                def simple_filtered_analyzer(*args, **kwargs):
                    try:
                        sql_query, table_schema, results = original_analyzer(*args, **kwargs)
                        # Şimdi SQL sorgusunu filtreleme fonksiyonumuzdan geçirelim
                        if sql_query and isinstance(sql_query, str):
                            filtered_sql = filter_llm_response_for_sql(sql_query)
                            logger.info(f"ORİJİNAL SQL (BASİT): {sql_query}")
                            logger.info(f"FİLTRELENMİŞ SQL (BASİT): {filtered_sql}")
                            return filtered_sql, table_schema, results
                        return sql_query, table_schema, results
                    except Exception as e:
                        logger.error(f"Basit analyzer düzeltme hatası: {str(e)}")
                        return args[0], None, []
                
                # Analyzer metodunu düzenlenmiş versiyonuyla değiştirelim
                db_query_engine._analyzer = simple_filtered_analyzer
                
                logger.info("DB modülü basit yapılandırma ile yüklendi. Özel SQL filtreleme etkin.")
            except Exception as fallback_error:
                logger.error(f"Alternatif JSONalyzeQueryEngine yapılandırması da başarısız oldu: {str(fallback_error)}")
                logger.exception("Fallback hata detayları:")
                logger.warning("DB modülü atlanıyor.")
                db_query_engine = None
    except Exception as e:
        logger.error(f"DB modülü yüklenirken hata oluştu: {str(e)}")
        logger.exception("Hata detayları:")
        logger.warning("DB modülü atlanıyor.")
        db_query_engine = None
    
    # PDF verisini yükle ve işle
    try:
        pdf_file = "document.pdf"  # PDF dosyasının adını buraya yazın
        
        # PDF dosyasının varlığını kontrol et
        if os.path.exists(pdf_file):
            # PDF indeksini oluştur veya yükle
            pdf_index = create_or_load_pdf_index(pdf_file)
            
            try:
                # PDF sorgu motorunu oluştur
                retriever = VectorIndexRetriever(
                    index=pdf_index,
                    similarity_top_k=5  # Daha fazla ilgili belge getir
                )
                
                # RetrieverQueryEngine oluştur
                pdf_query_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    llm=llm,
                    verbose=True,  # Ayrıntılı günlükleri etkinleştir
                    system_prompt=SYSTEM_PROMPT,
                    node_postprocessors=None  # Hata yapabilecek özel işleyicileri kaldır
                )
                
                logger.info("PDF modülü başarıyla yüklendi.")
            except Exception as retriever_error:
                logger.error(f"PDF sorgu motoru oluşturulurken hata: {str(retriever_error)}")
                logger.exception("Retriever hata detayları:")
                logger.warning("PDF modülü için basit yapılandırma deneniyor...")
                
                # Daha basit yapılandırma dene
                try:
                    # Basit sorgu motoru
                    pdf_query_engine = pdf_index.as_query_engine(
                        llm=llm,
                        system_prompt=SYSTEM_PROMPT
                    )
                    logger.info("PDF modülü basit yapılandırma ile yüklendi.")
                except Exception as fallback_error:
                    logger.error(f"Alternatif PDF sorgu motoru yapılandırması da başarısız oldu: {str(fallback_error)}")
                    logger.exception("PDF fallback hata detayları:")
                    logger.warning("PDF modülü atlanıyor.")
                    pdf_query_engine = None
        else:
            logger.warning(f"PDF dosyası bulunamadı: {pdf_file}")
            logger.warning("PDF modülü atlanıyor.")
            pdf_query_engine = None
    except Exception as e:
        logger.error(f"PDF modülü yüklenirken hata oluştu: {str(e)}")
        logger.exception("Hata detayları:")
        logger.warning("PDF modülü atlanıyor.")
        pdf_query_engine = None
    
    # Bellek kullanımını raporla
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} Bellek Kullanımı: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # vLLM ise ek bilgiler göster
            if "vllm" in llm_type and hasattr(llm, "client"):
                logger.info("vLLM kullanılıyor - KV cache bellek optimizasyonu aktif.")
                logger.info(f"vLLM PagedAttention: {os.environ.get('VLLM_USE_PAGED_ATTENTION', 'Tanımlanmamış')}")
                logger.info("PagedAttention, KV cache bellek parçalanmasını önemli ölçüde azaltır.")
    except Exception as mem_error:
        logger.error(f"Bellek kullanımı raporlama hatası: {str(mem_error)}")
    
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