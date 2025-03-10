#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

from ..db import JsonRagModule
from ..doc import PdfRagModule
from ..utils.logging_utils import setup_logging

logger = setup_logging("rag_router")

class RagRouter:
    """
    Kullanıcı isteklerini ilgili RAG modülüne yönlendiren router sınıfı.
    """
    
    def __init__(self):
        """Router sınıfını başlatır ve RAG modüllerini yükler."""
        self.json_rag = None
        self.pdf_rag = None
        self.initialized = False
        
        logger.info("RAG Router başlatılıyor...")
    
    def initialize(self):
        """RAG modüllerini başlatır."""
        from ..utils.models import configure_settings
        
        # Global ayarları yapılandır
        llm, _ = configure_settings()
        
        # JSON RAG modülünü başlat
        logger.info("JSON RAG modülü başlatılıyor...")
        self.json_rag = JsonRagModule().initialize(llm)
        
        # PDF RAG modülünü başlat
        logger.info("PDF RAG modülü başlatılıyor...")
        self.pdf_rag = PdfRagModule().initialize(llm)
        
        self.initialized = True
        logger.info("RAG Router başarıyla başlatıldı.")
        return self
    
    def route_query(self, query_text: str, module_type: str) -> str:
        """
        Kullanıcı sorgusunu belirtilen RAG modülüne yönlendirir.
        
        Args:
            query_text (str): Kullanıcı sorgusu
            module_type (str): Kullanılacak modül tipi ('json' veya 'pdf')
            
        Returns:
            str: Sorguya verilen yanıt
        """
        if not self.initialized:
            raise ValueError("Router henüz başlatılmadı. Önce initialize() metodunu çağırın.")
        
        logger.info(f"Sorgu '{module_type}' modülüne yönlendiriliyor: {query_text}")
        
        # İlgili modüle yönlendir
        if module_type == "json":
            if not self.json_rag:
                raise ValueError("JSON RAG modülü başlatılmadı.")
            response = self.json_rag.query(query_text)
        elif module_type == "pdf":
            if not self.pdf_rag:
                raise ValueError("PDF RAG modülü başlatılmadı.")
            response = self.pdf_rag.query(query_text)
        else:
            raise ValueError(f"Geçersiz modül tipi: {module_type}. 'json' veya 'pdf' olmalıdır.")
        
        return response 