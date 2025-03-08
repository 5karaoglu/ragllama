# RAG Uygulaması

Bu proje, LlamaIndex kullanarak bir Retrieval-Augmented Generation (RAG) uygulaması oluşturur. Uygulama, Book1.json veri setini kullanarak sorguları yanıtlar.

## Özellikler

- LlamaIndex ile RAG mimarisi
- JSONalyzeQueryEngine kullanımı ile JSON verilerinin analizi
- DeepSeek-7B dil modeli kullanımı
- Model cacheleme ile performans optimizasyonu
- Docker Compose ile kolay kurulum ve çalıştırma
- Detaylı loglama

## Kurulum ve Çalıştırma

Docker Compose kullanarak uygulamayı başlatmak için:

```bash
docker-compose up --build
```

## Kullanım

Uygulama başladıktan sonra, komut satırı arayüzü üzerinden sorularınızı sorabilirsiniz. Örnek sorular:

- "Hangi faturalar 1000 TL'den fazla tutara sahip?"
- "Volvo Car Turkey ile ilgili kayıtları göster"
- "En yüksek tutarlı 3 faturayı listele"
- "Toplam fatura tutarı nedir?"
- "Hangi müşterinin en çok faturası var?"

## Veri Seti

Uygulama, Book1.json dosyasındaki verileri kullanarak sorguları yanıtlar. JSONalyzeQueryEngine, JSON verilerini SQLite veritabanına yükleyerek SQL sorguları ile analiz yapabilir, bu da istatistiksel sorguları yanıtlamayı kolaylaştırır. 