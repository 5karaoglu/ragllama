"""
API endpoint'leri için Flask rotaları.
"""

import logging
from typing import Dict, Any, List
from flask import Flask, request, jsonify
from llama_index.core.callbacks import CallbackManager
import types

logger = logging.getLogger(__name__)

def setup_routes(app: Flask, db_query_engine, pdf_query_engine, llama_debug_handler):
    """
    Flask uygulamasına API rotalarını ekler.
    
    Args:
        app: Flask uygulaması
        db_query_engine: Veritabanı sorgu motoru
        pdf_query_engine: PDF sorgu motoru
        llama_debug_handler: LLM debug handler'ı
    """
    
    @app.route('/api/status', methods=['GET'])
    def status():
        """API durumunu kontrol eder."""
        return jsonify({
            "status": "online",
            "db_query_engine_ready": db_query_engine is not None,
            "pdf_query_engine_ready": pdf_query_engine is not None
        })

    @app.route('/api/query', methods=['POST'])
    def query():
        """Kullanıcı sorgusunu işler ve yanıt döndürür."""
        try:
            data = request.json
            user_query = data.get('query')
            module = data.get('module', 'db')  # Varsayılan olarak 'db' kullan
            show_thoughts = data.get('show_thoughts', True)  # Düşünce sürecini gösterme seçeneği
            
            if not user_query:
                return jsonify({"error": "Sorgu parametresi gerekli"}), 400
            
            # Sorgu öncesi event loglarını temizle
            if llama_debug_handler:
                llama_debug_handler.flush_event_logs()
                logger.info("Sorgu öncesi debug handler event logları temizlendi.")
            
            # Modüle göre sorguyu işle
            thought_process = []
            llm_response = None
            
            if module == 'db':
                if db_query_engine is None:
                    return jsonify({"error": "DB query engine henüz hazır değil"}), 503
                    
                logger.info(f"DB modülü ile soru işleniyor: {user_query}")
                
                # JSONalyzeQueryEngine sorgularını gözlemleme
                original_query = db_query_engine.query
                
                def logging_query_wrapper(self, query_str, **kwargs):
                    logger.info(f"LLM'e gönderilen sorgu: {query_str}")
                    response = original_query(query_str, **kwargs)
                    
                    # SQL kodunu ve LLM'in düşüncelerini logla
                    try:
                        if hasattr(response, 'metadata') and response.metadata is not None:
                            if 'sql_query' in response.metadata:
                                logger.info(f"ÜRETİLEN SQL SORGUSU: {response.metadata['sql_query']}")
                            if 'result' in response.metadata:
                                logger.info(f"SQL SORGU SONUCU: {response.metadata['result']}")
                    except Exception as log_error:
                        logger.error(f"Yanıt log hatası: {str(log_error)}")
                    
                    return response
                
                # Orijinal query fonksiyonunu geçici olarak değiştir
                db_query_engine.query = types.MethodType(logging_query_wrapper, db_query_engine)
                
                # Sorguyu çalıştır
                response = db_query_engine.query(user_query)
                llm_response = str(response)
                
                # Fonksiyonu eski haline getir
                db_query_engine.query = original_query
                
            elif module == 'pdf':
                if pdf_query_engine is None:
                    return jsonify({"error": "PDF query engine henüz hazır değil"}), 503
                    
                logger.info(f"PDF modülü ile soru işleniyor: {user_query}")
                
                response = pdf_query_engine.query(user_query)
                llm_response = str(response)
                
            else:
                return jsonify({"error": "Geçersiz modül parametresi"}), 400
            
            # LLM giriş/çıkışlarını ve düşünce sürecini topla
            if llama_debug_handler and show_thoughts:
                event_pairs = llama_debug_handler.get_llm_inputs_outputs()
                if event_pairs:
                    for i, (start_event, end_event) in enumerate(event_pairs):
                        logger.info(f"LLM Çağrısı #{i+1}:")
                        
                        # Çıkış yanıtındaki düşünme sürecini al
                        if 'response' in end_event.payload:
                            response_text = end_event.payload['response']
                            # Cevabı olduğu gibi alalım ama paragrafları ayıralım
                            thought_process.append({
                                "step": i+1,
                                "thought": response_text
                            })
                            logger.info(f"Çıkış yanıtı: {response_text}")
                        
                        # Girdi mesajlarını da alabilirsiniz (isteğe bağlı)
                        if 'messages' in start_event.payload:
                            messages = start_event.payload['messages']
                            for msg in messages:
                                if msg.get('role') == 'system' or msg.get('role') == 'user':
                                    logger.info(f"Giriş mesajı ({msg.get('role')}): {msg.get('content')}")
                
                # İşlem bittikten sonra event loglarını temizle
                llama_debug_handler.flush_event_logs()
                logger.info("Sorgu sonrası debug handler event logları temizlendi.")
            
            # Yanıtı ve düşünce sürecini içeren JSON'ı döndür
            response_data = {
                "query": user_query,
                "module": module,
                "response": llm_response
            }
            
            # Düşünce sürecini ekle (istenirse)
            if show_thoughts and thought_process:
                response_data["thoughts"] = thought_process
            
            return jsonify(response_data)
        
        except Exception as e:
            logger.error(f"Sorgu işlenirken hata oluştu: {str(e)}")
            logger.exception("Hata detayları:")
            return jsonify({"error": str(e)}), 500 