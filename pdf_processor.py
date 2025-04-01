"""
PDF işleme ve indeksleme için yardımcı fonksiyonlar.
"""

import os
import logging
import PyPDF2
from typing import List, Dict, Any
from pathlib import Path
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Document
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.base import LLM

logger = logging.getLogger(__name__)

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

def setup_pdf_query_engine(pdf_file: str, llm: LLM, system_prompt: str) -> RetrieverQueryEngine:
    """PDF sorgu motorunu oluşturur."""
    try:
        # PDF dosyasının varlığını kontrol et
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_file}")
        
        # PDF indeksini oluştur veya yükle
        pdf_index = create_or_load_pdf_index(pdf_file)
        
        # Retriever oluştur
        retriever = VectorIndexRetriever(
            index=pdf_index,
            similarity_top_k=3  # En benzer 3 belgeyi getir
        )
        
        # RetrieverQueryEngine oluştur
        pdf_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt,
            response_mode="tree_summarize",  # Yanıtları özetle
            verbose=True
        )
        
        logger.info("PDF sorgu motoru başarıyla oluşturuldu.")
        return pdf_query_engine
        
    except Exception as e:
        logger.error(f"PDF sorgu motoru oluşturulurken hata: {str(e)}")
        logger.exception("Hata detayları:")
        raise 