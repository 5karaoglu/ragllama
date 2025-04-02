"""
Veritabanı işlemleri için yardımcı fonksiyonlar.
"""

import os
import json
import logging
from typing import Dict, Any, List
from pathlib import Path

# LlamaIndex Core Imports
from llama_index.core import Settings
from llama_index.core.llms import LLM

# SQLAlchemy Imports for DB creation and interaction
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, inspect

# LlamaIndex SQL Utilities
from llama_index.core.utilities.sql_wrapper import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts import PromptTemplate

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
        sql_match = None
        if "```sql" in llm_response:
            sql_match = llm_response.split("```sql")[1].split("```")[0].strip()
        elif "SELECT " in llm_response and ";" in llm_response:
             start = llm_response.find("SELECT ")
             end = llm_response.find(";", start)
             if start != -1 and end != -1:
                 sql_match = llm_response[start:end+1].strip()

        if sql_match:
             sql = sql_match.replace("```", "").strip()
             sql = sql.split(';')[0].strip() + ';'
             if sql.endswith((".", "!", "?", ";;")):
                  sql = sql[:-1].strip()
                  if not sql.endswith(';'):
                      sql += ';'
             logger.debug(f"Ayıklanan SQL: {sql}")
             return sql
        else:
             logger.warning("Yanıt içinde SQL sorgusu bulunamadı.")
             return ""

    except Exception as e:
        logger.error(f"SQL sorgusu çıkarılırken hata oluştu: {str(e)}")
        return ""

# --- Rewritten setup_db_query_engine using SQLAlchemy exclusively ---
def setup_db_query_engine(json_file: str, llm: LLM, system_prompt: str) -> NLSQLTableQueryEngine:
    """Veritabanı sorgu motorunu NLSQLTableQueryEngine kullanarak oluşturur (SQLAlchemy ile)."""
    logger.info(f"NLSQLTableQueryEngine için kurulum başlatılıyor (SQLAlchemy): {json_file}")
    try:
        # --- 1. JSON Verisini Yükle ve Hazırla ---
        json_key = "Sheet1"
        table_name = "muhasebe_kayitlari"

        logger.debug(f"'{json_file}' dosyasından ham veri yükleniyor...")
        data_wrapper = load_json_data(json_file)
        logger.debug(f"'{json_key}' anahtarından veri listesi alınıyor...")
        data_list = data_wrapper.get(json_key)
        if data_list is None:
             raise ValueError(f"'{json_key}' anahtarı JSON dosyasında bulunamadı.")
        if not isinstance(data_list, list):
            raise ValueError(f"'{json_key}' anahtarı altında bir liste bekleniyordu, fakat {type(data_list)} bulundu.")
        if not data_list:
            raise ValueError(f"'{json_key}' anahtarı altındaki liste boş, sorgu motoru oluşturulamaz.")
        logger.info(f"'{json_key}' anahtarından {len(data_list)} kayıt başarıyla alındı.")

        # --- 2. SQLAlchemy Engine Oluştur ---
        db_uri = "sqlite:///file::memory:?cache=shared"
        logger.debug(f"SQLAlchemy motoru oluşturuluyor: {db_uri}")
        engine = create_engine(db_uri)
        metadata = MetaData()

        # --- 3. Tablo Şemasını Dinamik Olarak Tanımla ---
        logger.debug("Tablo şeması JSON verisinden dinamik olarak belirleniyor...")
        columns = []
        first_record = data_list[0]
        for key, value in first_record.items():
            col_type = String # Default to String
            if isinstance(value, int):
                col_type = Integer
            elif isinstance(value, float):
                col_type = Float # Use Float for potential decimals
            elif isinstance(value, str):
                col_type = String
            # Add more type checks if needed (e.g., for dates)
            columns.append(Column(key, col_type))
            logger.debug(f" Sütun: {key} -> Tür: {col_type}")

        logger.debug(f"SQLAlchemy Table nesnesi tanımlanıyor: {table_name}")
        dynamic_table = Table(table_name, metadata, *columns)

        # --- 4. Tabloyu Veritabanında Oluştur ---
        logger.debug(f"'{table_name}' tablosu veritabanında oluşturuluyor (eğer yoksa)...")
        try:
             metadata.create_all(bind=engine)
             logger.info(f"'{table_name}' tablosu başarıyla oluşturuldu/kontrol edildi.")
             # Verify table creation
             inspector = inspect(engine)
             if not inspector.has_table(table_name):
                 raise RuntimeError(f"Tablo '{table_name}' oluşturulduktan sonra veritabanında bulunamadı!")
             logger.debug(f"Tablo '{table_name}' varlığı doğrulandı.")
        except Exception as e:
             logger.error(f"SQLAlchemy ile tablo oluşturulurken hata: {e}", exc_info=True)
             raise

        # --- 5. Veriyi Tabloya Yükle ---
        logger.debug(f"'{table_name}' tablosuna veri yükleniyor (SQLAlchemy)...")
        try:
            with engine.connect() as connection:
                # Begin transaction
                with connection.begin(): 
                    connection.execute(dynamic_table.insert(), data_list)
                # Transaction is automatically committed here by connection.begin()
                logger.info(f"{len(data_list)} kayıt '{table_name}' tablosuna başarıyla yüklendi (SQLAlchemy).")
        except Exception as e:
            logger.error(f"SQLAlchemy ile veri yüklenirken hata: {e}", exc_info=True)
            raise

        # --- 6. LlamaIndex SQLDatabase Nesnesi Oluştur ---
        logger.debug("LlamaIndex SQLDatabase nesnesi oluşturuluyor...")
        sql_database = SQLDatabase(engine, include_tables=[table_name])
        logger.info(f"SQLDatabase nesnesi '{table_name}' tablosu için oluşturuldu.")
        try:
            table_info = sql_database.get_table_info([table_name])
            logger.debug(f"Alınan tablo bilgisi:\n{table_info}")
        except Exception as e:
            logger.warning(f"Tablo bilgisi alınırken hata oluştu (devam ediliyor): {e}")

        # --- 7. NLSQLTableQueryEngine Oluştur ---
        logger.debug("NLSQLTableQueryEngine oluşturuluyor...")

        # Özel Text-to-SQL Prompt Şablonu (Tam sütun adlarını kullanmaya zorla)
        custom_sql_prompt_str = (
            "Verilen bir girdi sorusu için, önce çalıştırılacak sözdizimsel olarak doğru bir {dialect} "
            "sorgusu oluştur, ardından sorgunun sonuçlarına bak ve yanıtı döndür.\n"
            "Doğru sütun adlarını kullanmak için aşağıda sağlanan tablo şemasını KULLANMALISIN.\n"
            "Şemada sağlanan TAM sütun adlarına (büyük/küçük harf dahil) ÇOK DİKKAT ET.\n"
            "Sütun adlarını snake_case veya başka bir formata dönüştürme. Tam adları kullan.\n"
            "Tablo Şeması:\n"
            "---------------------\n"
            "{schema}\n"
            "---------------------\n"
            "Soru: {query_str}\n"
            "SQL Sorgusu (YALNIZCA tek bir geçerli SQL SELECT ifadesi yaz, başına veya sonuna başka HİÇBİR metin veya yorum EKLEME): "
        )
        custom_text_to_sql_prompt = PromptTemplate(custom_sql_prompt_str)

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=[table_name],
            llm=llm,
            text_to_sql_prompt=custom_text_to_sql_prompt,
            verbose=True
        )
        logger.info("NLSQLTableQueryEngine başarıyla oluşturuldu (SQLAlchemy tabanlı, özel prompt ile).")
        return query_engine

    except Exception as e:
        logger.error(f"NLSQLTableQueryEngine kurulumunda genel hata: {e}", exc_info=True)
        raise 