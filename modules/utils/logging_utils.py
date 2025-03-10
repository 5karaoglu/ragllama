#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import colorlog

def setup_logging(logger_name=None):
    """
    Renkli loglama yapılandırması oluşturur.
    
    Args:
        logger_name (str, optional): Logger adı. None ise root logger kullanılır.
        
    Returns:
        logging.Logger: Yapılandırılmış logger nesnesi.
    """
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)
    
    # Eğer handler zaten eklenmediyse ekle
    if not logger.handlers:
        logger.addHandler(handler)
    
    # Diğer kütüphanelerin loglarını azalt
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("llama_index").setLevel(logging.WARNING)
    
    return logger 