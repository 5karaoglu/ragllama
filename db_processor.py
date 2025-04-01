"""
Veritabanı işlemleri için yardımcı fonksiyonlar.
"""

import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# LlamaIndex Core Imports (Keep Settings if needed elsewhere)
from llama_index.core import Settings
from llama_index.core.llms import LLM

# New Imports for NLSQLTableQueryEngine approach
from sqlalchemy import create_engine
import sqlite_utils
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """JSON dosyasından ham sözlük verisini yükler."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"JSON dosyası başarıyla yüklendi: {file_path}")
            return data
    except FileNotFoundError:
        logger.error(f"JSON dosyası bulunamadı: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON dosyası çözümlenirken hata oluştu: {file_path}, Hata: {e}")
        raise
    except Exception as e:
        logger.error(f"JSON yüklenirken beklenmedik hata ({file_path}): {e}")
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
    """LLM yanıtından SQL sorgusunu çıkarır (NLSQLTableQueryEngine bunu genellikle kendi yönetir)."""
    try:
        # SQL sorgusunu bul (Basit arama)
        sql_match = None
        if "```sql" in llm_response:
            sql_match = llm_response.split("```sql")[1].split("```")[0].strip()
        elif "SELECT " in llm_response and ";" in llm_response: # Daha genel arama
             start = llm_response.find("SELECT ")
             end = llm_response.find(";", start)
             if start != -1 and end != -1:
                 sql_match = llm_response[start:end+1].strip()

        if sql_match:
             # Temizleme (varsa ``` kaldır)
             sql = sql_match.replace("```", "").strip()
             # Sadece tek bir ifade olduğundan emin ol (ilkini al)
             sql = sql.split(';')[0].strip() + ';'
             # Sorgu sonunda gereksiz karakter varsa kaldır (opsiyonel)
             if sql.endswith((".", "!", "?", ";;")):
                  sql = sql[:-1].strip()
                  if not sql.endswith(';'):
                      sql += ';' # Tek bir ; ile bitmesini sağla
             logger.debug(f"Ayıklanan SQL: {sql}")
             return sql
        else:
             # SQL bulunamazsa orijinal yanıtı (veya boş string) döndür
             logger.warning("Yanıt içinde SQL sorgusu bulunamadı.")
             return "" # Veya llm_response

    except Exception as e:
        logger.error(f"SQL sorgusu çıkarılırken hata oluştu: {str(e)}")
        return "" # Hata durumunda boş string

def setup_db_query_engine(json_file: str, llm: LLM, system_prompt: str) -> NLSQLTableQueryEngine:
    """Veritabanı sorgu motorunu NLSQLTableQueryEngine kullanarak oluşturur."""
    logger.info(f"NLSQLTableQueryEngine için kurulum başlatılıyor: {json_file}")
    try:
        # --- 1. JSON Verisini Yükle ve Hazırla ---
        json_key = "Sheet1" # JSON dosyasındaki liste anahtarı
        table_name = "muhasebe_kayitlari" # SQL tablosuna verilecek ad

        logger.debug(f"'{json_file}' dosyasından ham veri yükleniyor...")
        data_wrapper = load_json_data(json_file)

        logger.debug(f"'{json_key}' anahtarından veri listesi alınıyor...")
        data_list = data_wrapper.get(json_key)
        if data_list is None:
             raise ValueError(f"'{json_key}' anahtarı JSON dosyasında bulunamadı.")
        if not isinstance(data_list, list):
            raise ValueError(f"'{json_key}' anahtarı altında bir liste bekleniyordu, fakat {type(data_list)} bulundu.")
        if not data_list:
            # Boş liste durumu - Hata ver veya boş motor döndür? Şimdilik hata verelim.
            raise ValueError(f"'{json_key}' anahtarı altındaki liste boş, sorgu motoru oluşturulamaz.")
        logger.info(f"'{json_key}' anahtarından {len(data_list)} kayıt başarıyla alındı.")

        # --- 2. In-Memory SQLite DB Oluştur ve Veriyi Yükle ---
        db_uri = "sqlite:///:memory:" # In-memory veritabanı URI'si
        logger.debug(f"SQLAlchemy motoru oluşturuluyor: {db_uri}")
        engine = create_engine(db_uri)

        logger.debug(f"'{table_name}' tablosuna (in-memory) veri yükleniyor...")
        try:
            db = sqlite_utils.Database(engine)
            # sqlite-utils'un PK'yı otomatik yönetmesine izin ver (rowid)
            db[table_name].insert_all(data_list)
            logger.info(f"{len(data_list)} kayıt '{table_name}' tablosuna başarıyla yüklendi.")
            # Doğrulama için sütunları logla
            logger.debug(f"'{table_name}' tablosunun sütunları: {db[table_name].columns_dict}")
        except Exception as e:
            logger.error(f"sqlite-utils ile veri yüklenirken hata oluştu: {e}")
            raise

        # --- 3. LlamaIndex SQLDatabase Nesnesi Oluştur ---
        logger.debug("LlamaIndex SQLDatabase nesnesi oluşturuluyor...")
        sql_database = SQLDatabase(engine, include_tables=[table_name])
        logger.info(f"SQLDatabase nesnesi '{table_name}' tablosu için oluşturuldu.")
        # İsteğe bağlı: Şemayı logla
        try:
            table_info = sql_database.get_table_info([table_name])
            logger.debug(f"Alınan tablo bilgisi:\n{table_info}")
        except Exception as e:
            logger.warning(f"Tablo bilgisi alınırken hata oluştu (devam ediliyor): {e}")


        # --- 4. NLSQLTableQueryEngine Oluştur ---
        # system_prompt burada doğrudan kullanılmaz. NLSQLTableQueryEngine şemayı kullanır.
        logger.debug("NLSQLTableQueryEngine oluşturuluyor...")
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[table_name],
            llm=llm, # LLM'i doğrudan ver
            verbose=True # Debugging için loglamayı etkin tut
        )
        logger.info("NLSQLTableQueryEngine başarıyla oluşturuldu.")
        return query_engine

    except Exception as e:
        logger.error(f"NLSQLTableQueryEngine kurulumunda genel hata: {e}", exc_info=True) # Hatanın izini sür
        raise # Hatayı tekrar fırlat 