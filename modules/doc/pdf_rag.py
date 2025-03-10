#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.huggingface import HuggingFaceLLM

from ..utils.logging_utils import setup_logging

logger = setup_logging("pdf_rag")

class PdfRagModule:
    """PDF dosyaları ile çalışan RAG modülü."""
    
    def __init__(self, pdf_dir: str = "./pdf_docs", persist_dir: str = "./storage/pdf"):
        """
        PDF RAG modülünü başlatır.
        
        Args:
            pdf_dir (str): PDF dosyalarının bulunduğu dizin
            persist_dir (str): İndeksin kaydedileceği dizin
        """
        self.pdf_dir = pdf_dir
        self.persist_dir = persist_dir
        self.documents = None
        self.index = None
        self.query_engine = None
        
        logger.info("PDF RAG modülü başlatılıyor...")
        
        # PDF dizinini oluştur
        os.makedirs(pdf_dir, exist_ok=True)
    
    def initialize(self, llm: Optional[HuggingFaceLLM] = None):
        """
        RAG modülünü başlatır ve gerekli bileşenleri yükler.
        
        Args:
            llm (HuggingFaceLLM, optional): Kullanılacak LLM modeli
        """
        # PDF dosyalarını yükle
        self.documents = self.load_pdf_documents(self.pdf_dir)
        
        # İndeksi oluştur veya yükle
        self.index = self.create_or_load_pdf_index(self.documents, self.persist_dir)
        
        # Retriever oluştur
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=3
        )
        
        # Query engine oluştur
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            llm=llm or Settings.llm,
            response_mode="compact"
        )
        
        logger.info("PDF RAG modülü başarıyla başlatıldı.")
        return self
    
    def query(self, query_text: str) -> str:
        """
        Kullanıcı sorgusunu işler ve yanıt döndürür.
        
        Args:
            query_text (str): Kullanıcı sorgusu
            
        Returns:
            str: Sorguya verilen yanıt
        """
        if not self.query_engine:
            raise ValueError("RAG modülü henüz başlatılmadı. Önce initialize() metodunu çağırın.")
        
        logger.info(f"PDF RAG sorgusu işleniyor: {query_text}")
        response = self.query_engine.query(query_text)
        
        return str(response)
    
    def load_pdf_documents(self, pdf_dir: str) -> List[Document]:
        """
        PDF dosyalarını yükler.
        
        Args:
            pdf_dir (str): PDF dosyalarının bulunduğu dizin
            
        Returns:
            List[Document]: Yüklenen PDF dokümanları
        """
        # PDF dizininde sample.pdf var mı kontrol et
        sample_pdf_path = os.path.join(pdf_dir, "sample.pdf")
        
        if not os.path.exists(sample_pdf_path):
            logger.warning(f"'{sample_pdf_path}' dosyası bulunamadı.")
            return []
        
        # PDF dosyasını manuel olarak yükle
        try:
            logger.info(f"'{sample_pdf_path}' dosyası yükleniyor...")
            
            # PyPDF kullanarak PDF dosyasını oku
            from pypdf import PdfReader
            
            reader = PdfReader(sample_pdf_path)
            text = ""
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"Sayfa {i+1}:\n{page_text}\n\n"
            
            # Document nesnesi oluştur
            document = Document(
                text=text,
                metadata={
                    "filename": "sample.pdf",
                    "file_path": sample_pdf_path
                }
            )
            
            logger.info(f"PDF dokümanı başarıyla yüklendi: {len(text)} karakter.")
            return [document]
            
        except Exception as e:
            logger.error(f"PDF dosyası yüklenirken hata oluştu: {str(e)}")
            return []
    
    def create_or_load_pdf_index(self, documents: List[Document], persist_dir: str):
        """
        PDF indeksini oluşturur veya yükler.
        
        Args:
            documents (List[Document]): PDF dokümanları
            persist_dir (str): İndeksin kaydedileceği dizin
            
        Returns:
            VectorStoreIndex: Oluşturulan veya yüklenen indeks
        """
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
                return self.create_new_pdf_index(documents, persist_dir)
        else:
            logger.info("Yeni PDF indeksi oluşturuluyor...")
            return self.create_new_pdf_index(documents, persist_dir)
    
    def create_new_pdf_index(self, documents: List[Document], persist_dir: str):
        """
        Yeni bir PDF indeksi oluşturur.
        
        Args:
            documents (List[Document]): PDF dokümanları
            persist_dir (str): İndeksin kaydedileceği dizin
            
        Returns:
            VectorStoreIndex: Oluşturulan indeks
        """
        if not documents:
            logger.warning("Doküman bulunamadı. Boş bir indeks oluşturuluyor.")
            # Boş bir indeks oluştur
            index = VectorStoreIndex([])
            
            # İndeksi kaydet
            os.makedirs(persist_dir, exist_ok=True)
            index.storage_context.persist(persist_dir=persist_dir)
            
            logger.info(f"Boş PDF indeksi '{persist_dir}' konumuna kaydedildi.")
            return index
        
        # Dokümanları cümlelere ayır
        splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=256
        )
        
        # İndeksi oluştur
        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[splitter]
        )
        
        # İndeksi kaydet
        os.makedirs(persist_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=persist_dir)
        
        logger.info(f"PDF indeksi '{persist_dir}' konumuna kaydedildi.")
        return index 