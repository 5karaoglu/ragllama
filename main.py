#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Sistemi Ana Uygulaması
=========================

Bu uygulama, JSON verileri ve PDF dosyaları ile çalışan bir RAG (Retrieval-Augmented Generation) sistemi sunar.
Kullanıcı istekleri API endpointleri üzerinden alınır ve belirtilen modüle yönlendirilir.

Kullanım:
    python main.py [--host HOST] [--port PORT] [--debug]

Parametreler:
    --host: API sunucusu host adresi (varsayılan: 0.0.0.0)
    --port: API sunucusu port numarası (varsayılan: 5000)
    --debug: Debug modunu etkinleştirir (varsayılan: False)
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, Optional

from modules.utils.logging_utils import setup_logging
from modules.backend.server import app, init_app, start_server

logger = setup_logging("main")

# Gunicorn için uygulama
app = init_app()

def parse_args():
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(description="RAG Sistemi API Sunucusu")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API sunucusu host adresi")
    parser.add_argument("--port", type=int, default=5000, help="API sunucusu port numarası")
    parser.add_argument("--debug", action="store_true", help="Debug modunu etkinleştirir")
    
    return parser.parse_args()

def main():
    """Ana uygulama fonksiyonu."""
    args = parse_args()
    
    logger.info("RAG sistemi API sunucusu başlatılıyor...")
    
    # Gerekli dizinleri oluştur
    os.makedirs("./storage", exist_ok=True)
    os.makedirs("./storage/json", exist_ok=True)
    os.makedirs("./storage/pdf", exist_ok=True)
    os.makedirs("./pdf_docs", exist_ok=True)
    
    # API sunucusunu başlat
    start_server(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 