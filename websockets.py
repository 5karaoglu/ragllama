import logging
import time
from flask import request
from flask_socketio import SocketIO, emit, join_room, leave_room
from threading import Lock
from system_monitor import get_system_metrics # Import the metrics function

logger = logging.getLogger(__name__)

# Global değişkenler
thread = None
thread_lock = Lock()
socketio_instance = None

def initialize_websockets(app):
    """Flask-SocketIO örneğini oluşturur ve olayları tanımlar."""
    global socketio_instance
    socketio_instance = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    logger.info("Flask-SocketIO başlatıldı.")

    @socketio_instance.on('connect')
    def handle_connect():
        """Yeni bir istemci bağlandığında çağrılır."""
        client_sid = request.sid
        logger.info(f'WebSocket istemcisi bağlandı: {client_sid}')
        # İstemciyi 'system_metrics' odasına ekle
        join_room('system_metrics')
        logger.info(f'İstemci {client_sid} system_metrics odasına katıldı.')
        
        # Arka plan thread'ini başlat (eğer çalışmıyorsa)
        global thread
        with thread_lock:
            if thread is None:
                logger.info("Metrik gönderim thread'i başlatılıyor...")
                thread = socketio_instance.start_background_task(target=background_metrics_emitter)
                logger.info("Metrik gönderim thread'i başlatıldı.")

    @socketio_instance.on('disconnect')
    def handle_disconnect():
        """Bir istemcinin bağlantısı kesildiğinde çağrılır."""
        client_sid = request.sid
        logger.info(f'WebSocket istemcisi ayrıldı: {client_sid}')
        # İstemciyi odadan çıkar (isteğe bağlı ama iyi pratik)
        leave_room('system_metrics')
        logger.info(f'İstemci {client_sid} system_metrics odasından ayrıldı.')
        # Not: Thread'i durdurmak için bağlı istemci sayısını kontrol edebiliriz,
        # ancak basitlik adına şimdilik thread çalışmaya devam edecek.

    return socketio_instance

def background_metrics_emitter():
    """Arka planda periyodik olarak sistem metriklerini gönderir."""
    logger.info('Metrik gönderim arka plan görevi başladı.')
    while True:
        try:
            # Belirli bir aralıkla CPU ölçümü yapmak için 0.5 sn bekle
            # Bu aynı zamanda gönderim sıklığını da belirler
            socketio_instance.sleep(1) # 1 saniyede bir gönderim yapalım
            
            # Metrikleri al (get_system_metrics içindeki CPU ölçümü için interval kullan)
            # Ancak system_monitor'deki interval=None şu an daha basit.
            metrics = get_system_metrics()
            
            # Metrikleri 'system_metrics' odasındaki tüm bağlı istemcilere gönder
            # emit fonksiyonu thread-safe'dir
            socketio_instance.emit('system_metrics_update', metrics, room='system_metrics')
            # logger.debug("Sistem metrikleri WebSocket üzerinden gönderildi.")
            
        except Exception as e:
            logger.error(f"Metrik gönderim thread'inde hata: {e}", exc_info=True)
            # Hata durumunda kısa bir süre bekle ve tekrar dene
            socketio_instance.sleep(5) 