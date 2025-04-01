"""
Veritabanı işlemleri için yardımcı fonksiyonlar.
"""

import os
import json
import logging
from typing import Dict, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.vector_stores import FaissVectorStore
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Dict[str, Any]:
    """JSON dosyasından veri yükler."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"JSON dosyasından {len(data.get('pages', []))} sayfa yüklendi")
            return data
    except Exception as e:
        logger.error(f"JSON dosyası yüklenirken hata oluştu: {str(e)}")
        raise

def create_or_load_json_index(json_data: Dict[str, Any], persist_dir: str = "./storage") -> VectorStoreIndex:
    """JSON indeksini oluşturur veya yükler."""
    try:
        # İndeks dosyasının yolu
        index_path = os.path.join(persist_dir, "json_index")
        
        # İndeks zaten varsa yükle
        if os.path.exists(index_path):
            logger.info(f"Mevcut JSON indeksi yükleniyor: {index_path}")
            return VectorStoreIndex.load(index_path)
        
        # Yeni indeks oluştur
        logger.info("Yeni JSON indeksi oluşturuluyor...")
        return create_new_json_index(json_data, persist_dir)
        
    except Exception as e:
        logger.error(f"JSON indeksi oluşturulurken/yüklenirken hata oluştu: {str(e)}")
        raise

def create_new_json_index(json_data: Dict[str, Any], persist_dir: str) -> VectorStoreIndex:
    """JSON verilerinden yeni bir indeks oluşturur."""
    try:
        # Document nesneleri oluştur
        documents = []
        for page in json_data.get('pages', []):
            doc = Document(
                text=page.get('content', ''),
                metadata={
                    'page_number': page.get('page_number', 0),
                    'source': 'Book1.json'
                }
            )
            documents.append(doc)
        
        # Node parser oluştur
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        
        # Vector store oluştur
        vector_store = FaissVectorStore.from_documents(
            documents,
            embed_model=Settings.embed_model
        )
        
        # İndeks oluştur
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            nodes=nodes
        )
        
        # İndeksi kaydet
        index_path = os.path.join(persist_dir, "json_index")
        index.storage_context.persist(persist_dir=index_path)
        logger.info(f"JSON indeksi kaydedildi: {index_path}")
        
        return index
        
    except Exception as e:
        logger.error(f"Yeni JSON indeksi oluşturulurken hata oluştu: {str(e)}")
        raise

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

def setup_db_query_engine(json_file: str, llm: LLM, system_prompt: str) -> RetrieverQueryEngine:
    """Veritabanı sorgu motorunu oluşturur."""
    try:
        # JSON dosyasının varlığını kontrol et
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON dosyası bulunamadı: {json_file}")
        
        # JSON verilerini yükle
        json_data = load_json_data(json_file)
        
        # İndeksi oluştur veya yükle
        index = create_or_load_json_index(json_data)
        
        # Retriever oluştur
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3
        )
        
        # Query engine oluştur
        query_engine = RetrieverQueryEngine.from_defaults(
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt,
            response_mode="tree_summarize"  # Yanıtları özetle
        )
        
        logger.info("Veritabanı sorgu motoru başarıyla oluşturuldu")
        return query_engine
        
    except Exception as e:
        logger.error(f"Veritabanı sorgu motoru oluşturulurken hata oluştu: {str(e)}")
        raise 