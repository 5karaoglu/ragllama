"""
Veritabanı işlemleri için yardımcı fonksiyonlar.
"""

import os
import json
import logging
import re
from typing import Dict, Any, List

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Document
)
from llama_index.core.query_engine import JSONalyzeQueryEngine
from llama_index.llms.base import LLM

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """JSON verisini yükler."""
    logger.info(f"'{file_path}' dosyası yükleniyor...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON verisi başarıyla yüklendi. {len(data)} sayfa bulundu.")
        return data
    
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {str(e)}")
        raise

def create_or_load_json_index(json_data: Dict[str, Any], persist_dir: str = "./storage"):
    """JSON indeksini oluşturur veya yükler."""
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

def create_new_json_index(json_data: Dict[str, Any], persist_dir: str):
    """Yeni bir JSON indeksi oluşturur."""
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

def setup_db_query_engine(json_file: str, llm: LLM, system_prompt: str) -> JSONalyzeQueryEngine:
    """Veritabanı sorgu motorunu oluşturur."""
    try:
        # JSON verisini yükle
        json_data = load_json_data(json_file)
        
        # Önce storage dizinini temizle
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
        logger.info("JSONalyzeQueryEngine özel filtre ile oluşturuluyor")
        
        # JSONalyzeQueryEngine oluştur
        db_query_engine = JSONalyzeQueryEngine(
            list_of_dict=json_rows,
            llm=llm,
            verbose=True,
            system_prompt=system_prompt,
            synthesize_response=True,  # SQL sorgusu çalıştırılsa bile yanıtı sentezle
            sql_optimizer=True,  # SQL sorgularını optimize et
            sql_parser=None,  # SQL Parser'ı devre dışı bırak
            infer_schema=True,  # Şema çıkarımını etkinleştir
            enforce_sql_syntax=True,  # SQL sözdizimi kontrolünü zorla
            output_direct_sql=True,  # LLM'in doğrudan SQL sorgusu döndürmesini sağla
            allow_multiple_queries=False  # Birden fazla sorguya izin verme
        )
        
        # Monkey patching yoluyla JSONalyzeQueryEngine'in _analyzer metodunu düzenleyelim
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
        return db_query_engine
        
    except Exception as e:
        logger.error(f"DB sorgu motoru oluşturulurken hata: {str(e)}")
        logger.exception("Hata detayları:")
        raise 