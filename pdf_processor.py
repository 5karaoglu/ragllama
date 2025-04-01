"""
PDF işleme ve indeksleme için yardımcı fonksiyonlar.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """PDF dosyasından metin çıkarır."""
    try:
        from PyPDF2 import PdfReader
        logger.info(f"PDF dosyasından metin çıkarılıyor: {file_path}")
        
        reader = PdfReader(file_path)
        pages = []
        
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text = page.extract_text()
            pages.append({
                "text": text,
                "metadata": {
                    "page_number": page_num + 1,
                    "source": file_path
                }
            })
        
        logger.info(f"{len(pages)} sayfa başarıyla çıkarıldı")
        return pages
        
    except Exception as e:
        logger.error(f"PDF'den metin çıkarılırken hata oluştu: {str(e)}")
        raise

def create_or_load_pdf_index(pdf_file: str, persist_dir: str = "./pdf_storage") -> VectorStoreIndex:
    """PDF indeksini oluşturur veya yükler."""
    try:
        # İndeks dosyasının yolu
        index_path = os.path.join(persist_dir, "pdf_index")
        
        # İndeks zaten varsa yükle
        if os.path.exists(index_path):
            logger.info(f"Mevcut PDF indeksi yükleniyor: {index_path}")
            return VectorStoreIndex.load(index_path)
        
        # Yeni indeks oluştur
        logger.info("Yeni PDF indeksi oluşturuluyor...")
        return create_new_pdf_index(pdf_file, persist_dir)
        
    except Exception as e:
        logger.error(f"PDF indeksi oluşturulurken/yüklenirken hata oluştu: {str(e)}")
        raise

def create_new_pdf_index(pdf_file: str, persist_dir: str) -> VectorStoreIndex:
    """PDF'den yeni bir indeks oluşturur."""
    try:
        # PDF'den metin çıkar
        pages = extract_text_from_pdf(pdf_file)
        
        # Document nesneleri oluştur
        documents = []
        for page in pages:
            doc = Document(
                text=page["text"],
                metadata=page["metadata"]
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
        index_path = os.path.join(persist_dir, "pdf_index")
        index.storage_context.persist(persist_dir=index_path)
        logger.info(f"PDF indeksi kaydedildi: {index_path}")
        
        return index
        
    except Exception as e:
        logger.error(f"Yeni PDF indeksi oluşturulurken hata oluştu: {str(e)}")
        raise

def setup_pdf_query_engine(pdf_file: str, llm: LLM, system_prompt: str) -> RetrieverQueryEngine:
    """PDF sorgu motorunu oluşturur."""
    try:
        # PDF dosyasının varlığını kontrol et
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_file}")
        
        # İndeksi oluştur veya yükle
        index = create_or_load_pdf_index(pdf_file)
        
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
        
        logger.info("PDF sorgu motoru başarıyla oluşturuldu")
        return query_engine
        
    except Exception as e:
        logger.error(f"PDF sorgu motoru oluşturulurken hata oluştu: {str(e)}")
        raise 