# Modüler RAG Sistemi

Bu proje, JSON verileri ve PDF dosyaları ile çalışan modüler bir RAG (Retrieval-Augmented Generation) sistemi sunar. Kullanıcı istekleri API endpointleri üzerinden alınır ve belirtilen modüle yönlendirilir.

## Özellikler

- **Modüler Yapı**: JSON ve PDF modülleri ayrı ayrı çalışabilir
- **Zorunlu Modül Seçimi**: Kullanıcı hangi modülün kullanılacağını açıkça belirtmelidir
- **API Desteği**: RESTful API ile dış sistemlerle entegrasyon
- **Sabit PDF Dosyası**: Sistem, pdf_docs dizinindeki sample.pdf dosyasını kullanır

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. PDF dosyasını yerleştirin:

```bash
# sample.pdf dosyasını pdf_docs dizinine yerleştirin
```

3. Uygulamayı başlatın:

```bash
# API sunucusunu başlatmak için
python main.py

# Özel host ve port belirtmek için
python main.py --host 127.0.0.1 --port 8000

# Debug modunda başlatmak için
python main.py --debug
```

## API Kullanımı

### Sorgu Gönderme

```bash
# Modül belirterek sorgu (json, db, pdf veya doc)
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Veri tablosunda kaç satır var?", "module": "json"}'
```

### Sistem Durumunu Kontrol Etme

```bash
curl http://localhost:5000/api/status
```

## API Endpointleri

| Endpoint | Metod | Açıklama | İstek Gövdesi | Başarılı Yanıt |
|----------|-------|----------|--------------|----------------|
| `/api/query` | POST | Sorgu gönderir | `{"query": "...", "module": "json\|pdf"}` | `{"response": "...", "module": "..."}` |
| `/api/status` | GET | Sistem durumunu döndürür | - | `{"status": "ready", "modules": {"json": true, "pdf": true}, "pdf_files": ["sample.pdf"]}` |

## Modüler Yapı

Sistem aşağıdaki modüllerden oluşur:

- **db**: JSON verileri ile çalışan RAG modülü
- **doc**: PDF dosyaları ile çalışan RAG modülü
- **backend**: Kullanıcı isteklerini işleyen ve ilgili modüle yönlendiren backend modülü
- **utils**: Yardımcı fonksiyonlar ve araçlar

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 