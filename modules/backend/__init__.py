#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Sistemi Backend Modülü
=========================

Bu modül, kullanıcı isteklerini işleyen ve ilgili RAG modülüne yönlendiren backend sistemini içerir.

Modüller:
- router: Kullanıcı isteklerini ilgili modüle yönlendiren router
- server: API sunucusu
"""

from .router import RagRouter
from .server import start_server

__all__ = ['RagRouter', 'start_server'] 