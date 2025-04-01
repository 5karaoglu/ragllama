"""
Veritabanı işlemleri için yardımcı fonksiyonlar.
"""

import os
import json
import logging
from typing import Dict, Any
from pathlib import Path
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.llms import LLM
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.experimental.query_engine import JSONalyzeQueryEngine

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """JSON dosyasından veri yükler."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"JSON dosyasından {len(data.get('pages', []))} sayfa yüklendi")
            return data
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading JSON from {file_path}: {str(e)}")
        raise

# Artık kullanılmayan fonksiyonlar yorum satırına alındı (JSONalyzeQueryEngine kullanılıyor)
# def create_or_load_json_index(json_data: Dict[str, Any], persist_dir: str = "./storage") -> VectorStoreIndex:
#     """JSON indeksini oluşturur veya yükler."""
#     try:
#         # İndeks dosyasının yolu
#         index_path = os.path.join(persist_dir, "json_index")
#         
#         # İndeks zaten varsa yükle
#         if os.path.exists(index_path):
#             logger.info(f"Mevcut JSON indeksi yükleniyor: {index_path}")
#             # VectorStoreIndex yükleme mantığı burada olmalıydı
#             raise NotImplementedError("VectorStoreIndex loading needs adjustment if this function is kept.")
#         
#         # Yeni indeks oluştur
#         logger.info("Yeni JSON indeksi oluşturuluyor...")
#         # return create_new_json_index(json_data, persist_dir)
#         raise NotImplementedError("create_new_json_index call needs adjustment if this function is kept.")
#         
#     except Exception as e:
#         logger.error(f"JSON indeksi oluşturulurken/yüklenirken hata oluştu: {str(e)}")
#         raise
#
# def create_new_json_index(json_data: Dict[str, Any], persist_dir: str) -> VectorStoreIndex:
#     """JSON verilerinden yeni bir indeks oluşturur."""
#     logger.info("Yeni JSON indeksi oluşturuluyor...")
#     
#     try:
#         # JSON verilerini düz metin olarak dönüştür
#         documents = []
#         for key, value in json_data.items():
#             if isinstance(value, dict):
#                 text = f"Key: {key}\n"
#                 for k, v in value.items():
#                     text += f"{k}: {v}\n"
#             else:
#                 text = f"Key: {key}\nValue: {value}\n"
#             # Document importu burada gerekliydi
#             # documents.append(Document(text=text))
#         
#         # Doğrudan VectorStoreIndex.from_documents() metodunu kullan
#         # index = VectorStoreIndex.from_documents(
#         #     documents,
#         #     show_progress=True
#         # )
#         
#         # İndeksi kaydet
#         # index.storage_context.persist(persist_dir=persist_dir)
#         logger.info("JSON indeksi başarıyla oluşturuldu ve kaydedildi.")
#         # return index
#         raise NotImplementedError("VectorStoreIndex creation needs adjustment if this function is kept.")
#         
#     except Exception as e:
#         logger.error(f"Yeni JSON indeksi oluşturulurken hata oluştu: {str(e)}")
#         raise

def filter_llm_response_for_sql(llm_response: str) -> str:
    """LLM yanıtından SQL sorgusunu çıkarır."""
    try:
        # SQL sorgusunu bul
        if "```sql" in llm_response:
            sql = llm_response.split("```sql")[1].split("```")[0].strip()
        elif "```" in llm_response:
            sql = llm_response.split("```")[1].strip()
        else:
            sql = llm_response.strip()
        
        # SQL sorgusunu temizle
        sql = sql.replace("```", "").strip()
        
        # Sorgu sonunda noktalama işareti varsa kaldır
        if sql.endswith((".", "!", "?")):
            sql = sql[:-1].strip()
        
        return sql
        
    except Exception as e:
        logger.error(f"SQL sorgusu çıkarılırken hata oluştu: {str(e)}")
        raise

def setup_db_query_engine(json_file: str, llm: LLM, system_prompt: str) -> JSONalyzeQueryEngine:
    """Veritabanı sorgu motorunu JSONalyzeQueryEngine kullanarak oluşturur."""
    try:
        # JSON dosyasının varlığını kontrol et
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON dosyası bulunamadı: {json_file}")
        
        # JSON verilerini yükle
        json_data = load_json_data(json_file)
        
        # QA Şablonunu oluştur (sistem prompt'u ile birlikte)
        qa_template_str = f"""{system_prompt}

Aşağıdaki JSON verilerini kullanarak soruyu yanıtla:
---------------------
{{context_str}}
---------------------
Soru: {{query_str}}
Yanıt:"""

        # JSONalyzeQueryEngine oluştur (özelleştirilmiş QA şablonu ile)
        query_engine = JSONalyzeQueryEngine(
            json_value=json_data,
            llm=llm,
            text_qa_template=qa_template_str, # Özel QA şablonunu ekle
            verbose=True
        )
        
        logger.info("JSONalyzeQueryEngine başarıyla oluşturuldu (özel QA şablonu ile)")
        return query_engine
        
    except Exception as e:
        logger.error(f"JSONalyzeQueryEngine oluşturulurken hata oluştu: {str(e)}")
        raise 