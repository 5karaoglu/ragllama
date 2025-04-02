import psutil
import time
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_SMI_AVAILABLE = True
    logger.info("pynvml başarıyla başlatıldı.")
except pynvml.NVMLError as e:
    NVIDIA_SMI_AVAILABLE = False
    logger.warning(f"NVIDIA SMI (pynvml) başlatılamadı: {e}. GPU metrikleri kullanılamayacak.")
except ImportError:
    NVIDIA_SMI_AVAILABLE = False
    logger.warning("pynvml kütüphanesi bulunamadı. GPU metrikleri kullanılamayacak. `pip install nvidia-ml-py3` ile kurun.")
except Exception as e:
    NVIDIA_SMI_AVAILABLE = False
    logger.error(f"pynvml başlatılırken beklenmedik hata: {e}")

def get_gpu_metrics() -> List[Dict[str, Any]]:
    """Her bir GPU için metrikleri toplar."""
    gpu_metrics = []
    if not NVIDIA_SMI_AVAILABLE:
        return gpu_metrics

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            
            # Bellek bilgisi (bytes)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_total_gb = mem_info.total / (1024**3)
            mem_used_gb = mem_info.used / (1024**3)
            mem_free_gb = mem_info.free / (1024**3)
            mem_percent = (mem_used_gb / mem_total_gb) * 100 if mem_total_gb > 0 else 0
            
            # Kullanım oranları
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            mem_util = utilization.memory # Bellek bant genişliği kullanımı
            
            # Sıcaklık
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError:
                temperature = None # Bazı GPU'lar desteklemeyebilir
                
            # Güç Kullanımı (Watt)
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 # Watt cinsinden
            except pynvml.NVMLError:
                power_usage = None
            
            gpu_metrics.append({
                "index": i,
                "name": gpu_name,
                "memory_total_gb": round(mem_total_gb, 2),
                "memory_used_gb": round(mem_used_gb, 2),
                "memory_free_gb": round(mem_free_gb, 2),
                "memory_percent": round(mem_percent, 1),
                "gpu_utilization_percent": gpu_util,
                "memory_bandwidth_utilization_percent": mem_util,
                "temperature_celsius": temperature,
                "power_usage_watts": power_usage
            })
    except pynvml.NVMLError as e:
        logger.error(f"GPU metrikleri alınırken NVMLError oluştu: {e}")
        # İletişim kesilmiş olabilir, yeniden başlatmayı deneyebiliriz ama şimdilik hatayı loglayalım
        # Belki NVIDIA_SMI_AVAILABLE = False yaparak sonraki denemeleri durdurabiliriz?
    except Exception as e:
        logger.error(f"GPU metrikleri alınırken beklenmedik hata: {e}")
        
    return gpu_metrics

def get_system_metrics(interval: Optional[float] = 0.5) -> Dict[str, Any]:
    """Genel sistem ve GPU metriklerini toplar."""
    
    # CPU Kullanımı (belirli bir aralıkla daha doğru sonuç verir)
    # interval=None anlık değeri, interval > 0 bloke edici ölçümü verir
    # İlk çağrıda None kullanıp sonraki periyodik çağrılarda interval kullanmak daha iyi olabilir
    # Şimdilik basitlik için anlık değeri alıyoruz.
    cpu_percent = psutil.cpu_percent(interval=None) 
    
    # RAM Kullanımı
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_available_gb = ram.available / (1024**3)
    ram_used_gb = ram.used / (1024**3)
    ram_percent = ram.percent
    
    # Disk Kullanımı (Kök dizin için)
    try:
        disk = psutil.disk_usage('/')
        disk_total_gb = disk.total / (1024**3)
        disk_used_gb = disk.used / (1024**3)
        disk_free_gb = disk.free / (1024**3)
        disk_percent = disk.percent
    except FileNotFoundError:
        logger.warning("Kök disk ('/') bulunamadı, disk metrikleri atlanıyor.")
        disk_total_gb, disk_used_gb, disk_free_gb, disk_percent = None, None, None, None

    metrics = {
        "timestamp": time.time(),
        "cpu_percent": cpu_percent,
        "ram": {
            "total_gb": round(ram_total_gb, 2),
            "available_gb": round(ram_available_gb, 2),
            "used_gb": round(ram_used_gb, 2),
            "percent": ram_percent
        },
        "disk": {
            "total_gb": round(disk_total_gb, 2) if disk_total_gb is not None else None,
            "used_gb": round(disk_used_gb, 2) if disk_used_gb is not None else None,
            "free_gb": round(disk_free_gb, 2) if disk_free_gb is not None else None,
            "percent": disk_percent
        },
        "gpus": get_gpu_metrics() # GPU metriklerini ayrı bir fonksiyondan al
    }
    
    return metrics

# Uygulama kapanırken pynvml'yi kapatmak iyi bir pratik
def shutdown_pynvml():
    if NVIDIA_SMI_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
            logger.info("pynvml başarıyla kapatıldı.")
        except pynvml.NVMLError as e:
            logger.error(f"pynvml kapatılırken hata oluştu: {e}")

# Ana thread kapatıldığında çağrılacak şekilde ayarlanabilir
# import atexit
# atexit.register(shutdown_pynvml) 