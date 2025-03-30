# vLLM Performans Optimizasyonu ve Entegrasyonu

## vLLM Nedir?

vLLM, UC Berkeley tarafından geliştirilen ve Büyük Dil Modellerinin (LLM) çıkarım (inference) süreçlerini optimize etmek için tasarlanmış bir kütüphanedir. vLLM, özellikle PagedAttention ve Continuous Batching gibi teknikleri kullanarak, geleneksel LLM çıkarım yöntemlerine göre bellek kullanımını ve hızı önemli ölçüde iyileştirir.

## Temel Avantajları

1. **PagedAttention Teknolojisi**: Bu teknoloji, işletim sistemlerinin sanal bellek yönetimi konseptinden esinlenmiş olup, LLM'lerin KV (Key-Value) önbelleğini daha verimli yönetir. Bu sayede:
   - Geleneksel yöntemlere göre 24 kata kadar hız artışı sağlar
   - GPU bellek kullanımını yarıya indirir
   - Bellek parçalanmasını (fragmentation) azaltır

2. **Continuous Batching**: Farklı uzunluktaki sorguları dinamik olarak birleştirerek işlem verimliliğini artırır ve GPU kullanımını optimize eder.

3. **Tensor Parallelism**: Büyük modelleri birden fazla GPU'ya dağıtarak, tek bir GPU'nun bellek sınırlamalarını aşmayı mümkün kılar.

## RAG Uygulamamıza vLLM Entegrasyonu

Projemize vLLM entegrasyonu aşağıdaki değişiklikleri içerir:

1. **Paket Bağımlılıkları**:
   - `vllm==0.3.2` - vLLM çekirdeği
   - `llama-index-llms-vllm==0.1.3` - LlamaIndex ve vLLM entegrasyonu

2. **Model Yükleme Değişiklikleri**:
   - LlamaIndex uyumlu vLLM entegrasyonu (`VLLMLangChainCompatibility`)
   - Geleneksel HuggingFace modeline fallback mekanizması

3. **Docker Çevre Değişkenleri**:
   - `LLM_TYPE=vllm` - vLLM kullanımını tetikler
   - `VLLM_USE_PAGED_ATTENTION=true` - PagedAttention'ı etkinleştirir
   - `VLLM_GPU_MEMORY_UTILIZATION=0.85` - GPU bellek kullanım oranını kontrol eder
   - `VLLM_TENSOR_PARALLEL_SIZE=2` - İki GPU üzerinde tensor paralelliği sağlar

4. **Dockerfile Güncellemeleri**:
   - CUDA kütüphaneleri (libcublas-dev, libcurand-dev, libcusparse-dev)
   - vLLM için çevre değişkenleri

## Performans İyileştirmeleri

vLLM entegrasyonu ile beklenen iyileştirmeler:

1. **Daha Yüksek Çıkarım Hızı**:
   - Geleneksel HuggingFace modeline göre 3-24x arası hızlanma
   - Özellikle uzun metin üretimi gerektiren görevlerde belirgin hız artışı

2. **Bellek Optimizasyonu**:
   - Daha düşük GPU bellek kullanımı
   - KV cache parçalanmasının azaltılması
   - Daha yüksek batch boyutları desteklenmesi

3. **Ölçeklenebilirlik**:
   - İki GPU üzerinde efektif tensor paralelliği
   - Daha büyük model ve batch boyutlarının desteklenmesi

## Entegrasyon Sonrası Kontrol Edilecekler

vLLM entegrasyonu sonrasında, aşağıdaki metriklere dikkat edilmelidir:

1. **Latency (Gecikme)**: Özellikle ilk token üretme süresi
2. **Throughput (İş hacmi)**: Saniyede işlenen istek/token sayısı
3. **GPU Bellek Kullanımı**: Maksimum ve ortalama bellek kullanımı
4. **Model Yükleme Süresi**: Başlangıçta modelin yüklenme süresi

## Olası Sorunlar ve Çözümleri

1. **Yeterli GPU Belleği Olmaması**:
   - `gpu_memory_utilization` değerini düşürün (örn. 0.7)
   - Daha küçük batch boyutları kullanın

2. **Bazı Modellerin Uyumsuzluğu**:
   - `trust_remote_code=True` ekleyin
   - Modeli `vllm.LLM` yerine doğrudan yükleyin

3. **Tokenizer Hataları**:
   - Özel tokenizer ekleme sorunlarında `tokenizer_mode="auto"` kullanın

## Örnek Performans Karşılaştırması

| Metrik | HuggingFace | vLLM | İyileştirme |
|--------|-------------|------|------------|
| Ortalama İlk Cevap Süresi | ~2-5 saniye | ~0.5-1 saniye | 4-5x |
| Token/saniye | 10-30 | 50-200 | 5-7x |
| GPU Bellek Kullanımı | ~24-30 GB | ~14-18 GB | ~40% azalma |
| Eşzamanlı İstek Kapasitesi | 4-6 | 10-20 | 2-3x |

## İleri Seviye Optimizasyon İpuçları

1. **Kantitatif Optimizasyon**: INT8 veya INT4 kantitasyon ile bellek kullanımını ve hızı daha da optimize edin.
2. **Prefix-Caching**: Sık kullanılan sistem promptlarını önbelleğe alarak prefill aşamasını hızlandırın.
3. **Lora Adaptörleri**: LoRA fine-tuning adaptörleri ile vLLM kullanarak özelleştirilmiş modelleri verimli şekilde çalıştırın.

---

Bu entegrasyon sayesinde, RAG uygulamamız çok daha hızlı ve bellek açısından verimli hale gelecektir. vLLM'in PagedAttention teknolojisi sayesinde, özellikle KV önbellek kullanımını optimize ederek daha fazla eşzamanlı kullanıcıya hizmet verebileceğiz. 