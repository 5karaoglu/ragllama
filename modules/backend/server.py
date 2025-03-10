#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

from .router import RagRouter
from ..utils.logging_utils import setup_logging

logger = setup_logging("rag_server")

# Flask uygulaması
app = Flask(__name__)
CORS(app)  # CORS desteği

# Global router nesnesi
router = None

@app.route('/api/query', methods=['POST'])
def query():
    """
    Kullanıcı sorgusunu işler ve yanıt döndürür.
    
    Request:
    {
        "query": "Kullanıcı sorgusu",
        "module": "json|pdf|db|doc" (zorunlu)
    }
    
    Response:
    {
        "response": "Yanıt metni",
        "module": "json|pdf"
    }
    """
    global router
    
    # Router'ın başlatıldığından emin ol
    if router is None or not router.initialized:
        return jsonify({
            "error": "RAG sistemi henüz başlatılmadı."
        }), 500
    
    # İstek verilerini al
    data = request.json
    
    if not data or 'query' not in data:
        return jsonify({
            "error": "Geçersiz istek. 'query' alanı gereklidir."
        }), 400
    
    if not data or 'module' not in data:
        return jsonify({
            "error": "Geçersiz istek. 'module' alanı gereklidir."
        }), 400
    
    query_text = data['query']
    module_type = data['module']
    
    if not query_text or not isinstance(query_text, str):
        return jsonify({
            "error": "Geçersiz sorgu. Sorgu metni bir string olmalıdır."
        }), 400
    
    if not module_type or not isinstance(module_type, str):
        return jsonify({
            "error": "Geçersiz modül. Modül tipi bir string olmalıdır."
        }), 400
    
    try:
        # Sorguyu işle
        logger.info(f"Gelen sorgu: {query_text}, Belirtilen modül: {module_type}")
        
        # Modül tipini standartlaştır
        if module_type.lower() in ['json', 'db']:
            module_type = 'json'
        elif module_type.lower() in ['pdf', 'doc']:
            module_type = 'pdf'
        else:
            return jsonify({
                "error": f"Geçersiz modül tipi: {module_type}. 'json', 'db', 'pdf' veya 'doc' olmalıdır."
            }), 400
        
        # Sorguyu yönlendir
        response = router.route_query(query_text, module_type)
        
        return jsonify({
            "response": response,
            "module": module_type
        })
    
    except Exception as e:
        logger.error(f"Sorgu işlenirken hata oluştu: {str(e)}")
        logger.exception("Hata detayları:")
        
        return jsonify({
            "error": f"Sorgu işlenirken hata oluştu: {str(e)}"
        }), 500

@app.route('/api/status', methods=['GET'])
def status():
    """
    Sistemin durumunu döndürür.
    """
    global router
    
    # Router'ın başlatıldığından emin ol
    if router is None:
        return jsonify({
            "status": "not_initialized",
            "message": "RAG sistemi henüz başlatılmadı."
        })
    
    # PDF dosyalarını listele
    pdf_files = []
    if router.pdf_rag:
        pdf_dir = router.pdf_rag.pdf_dir
        if os.path.exists(pdf_dir):
            pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    return jsonify({
        "status": "ready" if router.initialized else "initializing",
        "modules": {
            "json": router.json_rag is not None,
            "pdf": router.pdf_rag is not None
        },
        "pdf_files": pdf_files
    })

def init_app():
    """
    Flask uygulamasını başlatır ve yapılandırır.
    Gunicorn ile kullanım için.
    
    Returns:
        Flask: Yapılandırılmış Flask uygulaması
    """
    global router
    
    # Router'ı başlat
    if router is None or not router.initialized:
        logger.info("RAG API sunucusu başlatılıyor...")
        
        # Gerekli dizinleri oluştur
        os.makedirs("./storage", exist_ok=True)
        os.makedirs("./storage/json", exist_ok=True)
        os.makedirs("./storage/pdf", exist_ok=True)
        os.makedirs("./pdf_docs", exist_ok=True)
        
        # Router'ı başlat
        router = RagRouter().initialize()
        
        logger.info("RAG API sunucusu başarıyla başlatıldı.")
    
    return app

def start_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """
    API sunucusunu başlatır.
    
    Args:
        host (str): Sunucu host adresi
        port (int): Sunucu port numarası
        debug (bool): Debug modu
    """
    global router
    
    logger.info("RAG API sunucusu başlatılıyor...")
    logger.info(f"Host: {host}, Port: {port}, Debug: {debug}")
    
    # Router'ı başlat
    if router is None or not router.initialized:
        # Gerekli dizinleri oluştur
        os.makedirs("./storage", exist_ok=True)
        os.makedirs("./storage/json", exist_ok=True)
        os.makedirs("./storage/pdf", exist_ok=True)
        os.makedirs("./pdf_docs", exist_ok=True)
        
        # Router'ı başlat
        router = RagRouter().initialize()
    
    # Kullanılabilir endpointleri göster
    logger.info("Kullanılabilir API endpointleri:")
    logger.info(f"  POST {host}:{port}/api/query - Sorgu gönderme")
    logger.info(f"  GET  {host}:{port}/api/status - Sistem durumunu kontrol etme")
    
    # Sunucuyu başlat
    logger.info(f"Sunucu {host}:{port} adresinde başlatılıyor...")
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    start_server(debug=True) 