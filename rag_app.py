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
9. Sorgu sonucunu ayrı bir yanıt olarak verin, sorguyu yanıta dahil etmeyin.
10. Eğer sorgu oluşturmakta zorlanırsanız, verilere doğrudan bakıp analiz yapın.

ÖRNEKLER:
DOĞRU: SELECT * FROM users WHERE name = 'Ali';
YANLIŞ: SELECT * FROM users WHERE name = 'Ali'; # Bu Ali'yi bulan sorgu

Göreviniz, kullanıcının sorularını belgelerden elde ettiğiniz bilgilerle detaylı ve doğru bir şekilde yanıtlamaktır.
"""

# Özel SQL Parser sınıfı oluşturalım
class CustomSQLParser:
    """Özel SQL ayrıştırıcı - LLM tarafından oluşturulan SQL sorgularını temizler ve düzenler."""
    
    def __init__(self):
        """Özel SQL Parser'ı başlat."""
        pass
    
    def parse_response_to_sql(self, sql_query: str, query_bundle: Optional[Any] = None) -> str:
        """
        LLM tarafından oluşturulan SQL sorgularını temizler ve düzenler.
        
        Args:
            sql_query: LLM tarafından oluşturulan SQL sorgusu
            query_bundle: Query bundle nesnesi (isteğe bağlı)
            
        Returns:
            Temizlenmiş ve düzenlenmiş SQL sorgusu
        """
        try:
            logger.info(f"SQL PARSER GİRİŞ SORGUSU: {sql_query}")
            
            # Boşlukları kırp
            sql_query = sql_query.strip()
            
            # Markdown kod bloklarını kaldır
            sql_query = sql_query.replace("```", "")
            sql_query = sql_query.replace("```sql", "")
            sql_query = sql_query.replace("```SQL", "")
            
            # sql ve SQL kelimelerini sorgulardan kaldır
            sql_query = re.sub(r"^sql\s+", "", sql_query, flags=re.IGNORECASE)
            sql_query = re.sub(r"^SQL\s+", "", sql_query, flags=re.IGNORECASE)
            
            # Yorum satırlarını kaldır
            sql_query = re.sub(r"#.*", "", sql_query)  # # ile başlayan yorumları kaldır
            sql_query = re.sub(r"--.*", "", sql_query)  # -- ile başlayan yorumları kaldır
            sql_query = re.sub(r"/\*.*?\*/", "", sql_query, flags=re.DOTALL)  # /* */ yorumlarını kaldır
            
            # SELECT ile başlayan kısmı al
            if "SELECT" in sql_query.upper():
                sql_query = sql_query[sql_query.upper().find("SELECT"):]
            
            # SQL sözdizimi kontrolü
            sql_query = sql_query.strip()
            if not sql_query.endswith(";"):
                sql_query += ";"
                
            # İsteğe bağlı olarak sorguyu test et
            # sqlite3.connect(":memory:").execute(sql_query)
            
            logger.info(f"SQL PARSER DÖNÜŞ SORGUSU: {sql_query}")
            return sql_query
        except Exception as e:
            # Herhangi bir hata durumunda orijinal sorguyu sadeleştirerek döndür
            logger.error(f"SQL Parser hatası: {str(e)}")
            logger.error(f"Orijinal sorgu: {sql_query}")
            cleaned_query = re.sub(r"#.*", "", sql_query)  # # işaretli yorumları kaldır
            cleaned_query = f"SELECT * FROM table_data LIMIT 5;" # Güvenli sorgu ile değiştir
            logger.info(f"SQL PARSER HATA SONRASI DÖNÜŞ SORGUSU: {cleaned_query}")
            return cleaned_query

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
            logger.info("GPU kullanılacak: device_map=auto")
        else:
            logger.warning("GPU kullanılamıyor, CPU kullanılacak!")
        
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
            
            response = db_query_engine.query(user_query)
            llm_response = str(response)
            
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
    
    # Debug handler'ı kur
    setup_debug_handler()
    
    # Modelleri yapılandır
    llm = setup_llm()
    embed_model = setup_embedding_model()
    
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
            # Özel SQL Parser oluştur
            custom_sql_parser = CustomSQLParser()
            logger.info("Özel SQL Parser oluşturuldu")
            
            # JSONalyzeQueryEngine oluştur
            db_query_engine = JSONalyzeQueryEngine(
                list_of_dict=json_rows,
                llm=llm,
                verbose=True,
                system_prompt=SYSTEM_PROMPT,
                synthesize_response=True,  # SQL sorgusu çalıştırılsa bile yanıtı sentezle
                sql_optimizer=True,  # SQL sorgularını optimize et
                sql_parser=custom_sql_parser  # Özel SQL parser kullan
            )
            
            logger.info("DB modülü başarıyla yüklendi.")
        except Exception as sql_error:
            logger.error(f"JSONalyzeQueryEngine oluşturulurken SQL hatası: {str(sql_error)}")
            logger.exception("SQL hata detayları:")
            logger.warning("JSONalyzeQueryEngine oluşturulurken hata, varsayılan yapılandırma deneniyor...")
            
            # Hata durumunda daha basit yapılandırmayı dene
            try:
                # Özel SQL Parser oluştur (yedek)
                custom_sql_parser = CustomSQLParser()
                
                db_query_engine = JSONalyzeQueryEngine(
                    list_of_dict=json_rows,
                    llm=llm,
                    verbose=True,
                    system_prompt=SYSTEM_PROMPT,
                    infer_schema=False,  # Şema çıkarımını devre dışı bırak
                    sql_parser=custom_sql_parser  # Özel SQL parser kullan
                )
                logger.info("DB modülü basit yapılandırma ile yüklendi.")
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