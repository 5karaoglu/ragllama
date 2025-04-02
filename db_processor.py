"""
Veritabanı işlemleri için yardımcı fonksiyonlar.
"""

import os
import json
import logging
import re
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
    """LLM yanıtından ilk geçerli SQL SELECT ifadesini regex kullanarak çıkarır."""
    if not llm_response or not isinstance(llm_response, str):
        logger.warning("filter_llm_response_for_sql: Geçersiz veya boş yanıt alındı.")
        return ""

    try:
        # SELECT ile başlayıp ; ile biten ilk ifadeyi ara (case-insensitive SELECT)
        # .*? -> non-greedy match for anything between SELECT and ;
        # re.DOTALL -> . karakterinin newline dahil her şeyi eşlemesini sağlar
        # re.IGNORECASE -> SELECT ifadesini büyük/küçük harf duyarsız arar
        # Corrected regex string literal and flags
        match = re.search(r"SELECT.*?\;", llm_response, re.IGNORECASE | re.DOTALL)

        if match:
            sql = match.group(0).strip()
            # Başında/sonunda olabilecek ```sql ve ``` gibi işaretleri temizle (ekstra güvenlik)
            # Corrected regex string literal for cleaning
            sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
            logger.debug(f"Regex ile ayıklanan SQL: {sql}")
            return sql
        else: # Corrected indentation/placement if needed, or just ensure it's correct
            logger.warning(f"Yanıt içinde 'SELECT ... ;' kalıbında SQL sorgusu bulunamadı. Yanıt: {llm_response[:500]}...") # Log a snippet
            return ""

    except Exception as e:
        logger.error(f"SQL sorgusu regex ile çıkarılırken hata oluştu: {str(e)}")
        return ""

# --- Rewritten setup_db_query_engine using SQLAlchemy exclusively ---
def setup_db_query_engine(json_file: str) -> SQLDatabase:
    """JSON verisini yükler, SQLAlchemy ile in-memory DB oluşturur ve SQLDatabase nesnesini döndürür."""
    logger.info(f"SQLDatabase kurulumu başlatılıyor (SQLAlchemy): {json_file}")
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

        # --- 6. LlamaIndex SQLDatabase Nesnesi Oluştur ve Döndür ---
        logger.debug("LlamaIndex SQLDatabase nesnesi oluşturuluyor...")
        sql_database = SQLDatabase(engine, include_tables=[table_name])
        logger.info(f"SQLDatabase nesnesi '{table_name}' tablosu için başarıyla oluşturuldu ve döndürülüyor.")
        # Optional: Log schema info (already present)
        try:
            table_info = sql_database.get_table_info([table_name])
            logger.debug(f"Alınan tablo bilgisi:\n{table_info}")
        except Exception as e:
            logger.warning(f"Tablo bilgisi alınırken hata oluştu (devam ediliyor): {e}")
            
        return sql_database # Return the SQLDatabase object

    except Exception as e:
        # Adjusted error message to reflect function's new purpose
        logger.error(f"SQLDatabase kurulumunda genel hata: {e}", exc_info=True)
        raise 

# --- New Function for Direct SQL Execution ---
def execute_natural_language_query(sql_database: SQLDatabase, llm: LLM, user_query: str) -> str:
    """Doğal dil sorgusunu alır, LLM ile SQL'e çevirir, filtreler ve SQLDatabase üzerinde çalıştırır."""
    logger.info(f"Doğal dil sorgusu alınıyor: {user_query}")

    try:
        # --- 1. Get Table Schema using SQLAlchemy Inspect --- 
        if not sql_database.get_usable_table_names():
            raise ValueError("SQLDatabase içinde kullanılabilir tablo bulunamadı.")
        table_name = list(sql_database.get_usable_table_names())[0]
        logger.debug(f"Kullanılacak tablo: {table_name}")

        # Use SQLAlchemy inspect to get schema
        engine = sql_database.engine
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        
        # Format schema string for the prompt
        column_descriptions = []
        for column in columns:
            column_descriptions.append(f"{column['name']} ({str(column['type'])})")
        context = f"Table '{table_name}' has columns: {', '.join(column_descriptions)}."
        logger.debug(f"SQLAlchemy inspect ile alınan şema bilgisi:\n{context}")

        # --- 2. Create SQL Generation Prompt --- 
        custom_sql_prompt_str = (
            "Verilen bir girdi sorusu için, önce çalıştırılacak sözdizimsel olarak doğru bir {dialect} "
            "sorgusu oluştur.\n"
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
        sql_generation_prompt = PromptTemplate(template=custom_sql_prompt_str)

        # Access dialect name via the engine attribute
        dialect = sql_database.engine.dialect.name
        formatted_prompt = sql_generation_prompt.format(
            dialect=dialect, 
            schema=context, 
            query_str=user_query
        )
        logger.debug(f"SQL üretimi için LLM'e gönderilecek prompt:\n{formatted_prompt}")
        logger.info("SQL sorgusu üretmek için LLM çağrılıyor...")
        llm_response = llm.complete(formatted_prompt)
        raw_sql_response = llm_response.text
        logger.debug(f"LLM'den ham SQL yanıtı alındı:\n{raw_sql_response}")
        logger.info("LLM yanıtından SQL sorgusu ayıklanıyor...")
        clean_sql = filter_llm_response_for_sql(raw_sql_response)
        if not clean_sql:
            logger.error("LLM yanıtından geçerli SQL sorgusu ayıklanamadı.")
            return f"Üzgünüm, sorgunuzu SQL'e çeviremedim. LLM yanıtı: {raw_sql_response}"
        logger.info(f"Ayıklanan SQL sorgusu çalıştırılıyor: {clean_sql}")
        try:
            sql_result = sql_database.run_sql(clean_sql)
            logger.info("SQL sorgu sonucu başarıyla alındı.")
            logger.debug(f"SQL Sonucu: {sql_result}")
            return str(sql_result)
        except Exception as sql_exec_error:
            logger.error(f"SQL sorgusu çalıştırılırken hata oluştu: {clean_sql}, Hata: {sql_exec_error}", exc_info=True)
            return f"SQL sorgusu ({clean_sql}) çalıştırılırken bir hata oluştu: {sql_exec_error}"
            
    except Exception as e:
        logger.error(f"Doğal dil sorgusu işlenirken genel hata: {e}", exc_info=True)
        return f"Sorgunuz işlenirken beklenmedik bir hata oluştu: {e}"


# --- setup_db_query_engine remains the same, returning SQLDatabase --- 