"""
API endpoint'leri için Flask rotaları.
"""

import logging
from typing import Dict, Any, List
from flask import Flask, request, jsonify
# Removed CallbackManager, types - no longer needed here for DB part

# Import the execution function and potentially the globals from rag_app
from db_processor import execute_natural_language_query
# Attempt to import globals from rag_app (adjust path if needed)
try:
    from rag_app import sql_database, global_llm
except ImportError:
    sql_database = None
    global_llm = None
    logging.warning("Could not import globals from rag_app directly in api.py")

logger = logging.getLogger(__name__)

def setup_routes(app: Flask, pdf_query_engine, llama_debug_handler):
    """
    Flask uygulamasına API rotalarını ekler.
    
    Args:
        app: Flask uygulaması
        pdf_query_engine: PDF sorgu motoru
        llama_debug_handler: LLM debug handler'ı
    """
    
    @app.route('/api/status', methods=['GET'])
    def status():
        """API durumunu kontrol eder."""
        is_sql_db_ready = sql_database is not None
        return jsonify({
            "status": "online",
            "sql_database_ready": is_sql_db_ready,
            "pdf_query_engine_ready": pdf_query_engine is not None
        })

    @app.route('/api/query', methods=['POST'])
    def query():
        """Kullanıcı sorgusunu işler ve yanıt döndürür."""
        try:
            data = request.json
            user_query = data.get('query')
            module = data.get('module', 'db')
            show_thoughts = data.get('show_thoughts', True)
            
            if not user_query:
                return jsonify({"error": "Sorgu parametresi gerekli"}), 400
            
            if llama_debug_handler:
                llama_debug_handler.flush_event_logs()
                logger.info("Sorgu öncesi debug handler event logları temizlendi.")
            
            thought_process = []
            llm_response = None
            
            # Access globals for DB query
            current_sql_db = sql_database
            current_llm = global_llm
            
            if module == 'db':
                if current_sql_db is None or current_llm is None:
                    return jsonify({"error": "Veritabanı veya LLM modülü henüz hazır değil"}), 503
                    
                logger.info(f"DB modülü ile soru işleniyor: {user_query}")
                
                llm_response = execute_natural_language_query(
                    sql_database=current_sql_db,
                    llm=current_llm,
                    user_query=user_query
                )
                
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