# Transfer Verisi Entegrasyon Stratejisi

## 📊 Durum Analizi

### Mevcut Veri
**FBref Performans Verileri:**
- ✅ 8,693 oyuncu kaydı
- ✅ 5 lig, 3 sezon (2021-22, 2022-23, 2023-24)
- ✅ Detaylı performans metrikleri (goller, asistler, xG, xAG, vb.)
- ✅ Oyuncu bilgileri (isim, yaş, pozisyon, milliyet)

### Eksik Veri
**Transfer Bilgileri:**
- ❌ Transfer bedelleri
- ❌ Piyasa değerleri
- ❌ Transfer tarihleri
- ❌ Önceki/sonraki kulüp bilgileri

---

## 🔍 Araştırma Sonuçları

### İncelenen Veri Kaynakları

| Kaynak | Dönem | 2021-2024 Kapsama | Transfer Bedeli | Piyasa Değeri | Erişim | Durum |
|--------|-------|-------------------|-----------------|---------------|--------|-------|
| **Mexwell (Kaggle)** | 2009-2021 | ⚠️ Sadece 2021 | ✅ Var | ✅ Var | Kaggle API | Kısmi uygun |
| **Davidcariboo (Kaggle)** | Haftalık güncelleme | ✅ Tam kapsama | ❓ Kontrol gerekli | ✅ 400K+ kayıt | Kaggle API / DVC | **En uygun** |
| **Ewenme (GitHub)** | 1992-2022 | ❌ 2023-2024 yok | ✅ Var (%42) | ❌ Yok | ✅ Doğrudan wget | Uygun değil |

### Seçilen Kaynak
**Davidcariboo/transfermarkt-datasets** - En kapsamlı ve güncel veri

**Avantajları:**
- Haftalık otomatik güncelleme
- 2021-2024 tam kapsama
- 400,000+ piyasa değeri kaydı
- 1,200,000+ oyuncu görünüm kaydı
- GitHub'da açık kaynak (kod + dokümantasyon)
- Relational database yapısı

**Dezavantajları:**
- DVC ile yönetiliyor (S3'te depolanıyor)
- Kaggle API gerekiyor veya manuel indirme
- Birden fazla CSV dosyası (entegrasyon gerekli)

---

## 🎯 Önerilen Strateji

### Yaklaşım 1: Davidcariboo Veri Setini Kullanma (ÖNCELİKLİ)

**Adımlar:**

#### 1. Veri İndirme
```bash
# Seçenek A: Kaggle web interface'den manuel indirme
# - www.kaggle.com/datasets/davidcariboo/player-scores
# - Download butonuna tıkla
# - ZIP dosyasını projeye yükle

# Seçenek B: DVC ile indirme (GitHub repo'yu clone edip)
git clone https://github.com/dcaribou/transfermarkt-datasets.git
cd transfermarkt-datasets
dvc pull data/prep
```

#### 2. İlgili CSV Dosyalarını Belirleme
Davidcariboo veri setinde muhtemelen şu dosyalar var:
- `players.csv` - Oyuncu bilgileri
- `player_valuations.csv` - Piyasa değeri geçmişi ✅✅✅
- `transfers.csv` veya `appearances.csv` - Transfer/maç bilgileri
- `clubs.csv` - Kulüp bilgileri
- `competitions.csv` - Lig bilgileri

#### 3. Veri Entegrasyonu
**Eşleştirme Stratejisi:**
```python
# FBref verileri ile Transfermarkt verilerini eşleştirme
# Eşleştirme kriterleri:
# 1. Oyuncu ismi (fuzzy matching ile)
# 2. Yaş (±1 yıl tolerans)
# 3. Pozisyon
# 4. Kulüp adı
# 5. Sezon
```

**Fuzzy Matching:**
```python
from fuzzywuzzy import fuzz, process

# Örnek: "Mohamed Salah" vs "M. Salah" vs "Mo Salah"
# Fuzzy matching ile %85+ benzerlik oranı
```

#### 4. Feature Engineering
**Transfer Öncesi Performans:**
- Son sezon goller, asistler, xG, xAG
- Oynama süresi (dakika, maç sayısı)
- Per 90 normalize metrikler

**Transfer Sonrası Performans:**
- Yeni takımda ilk sezon performansı
- Performans değişimi (delta)

**Transfer Özellikleri:**
- Transfer bedeli (EUR)
- Piyasa değeri (EUR)
- Yaş
- Pozisyon
- Lig geçişi (aynı lig / farklı lig / lig seviyesi değişimi)

---

### Yaklaşım 2: Hibrit Yaklaşım (ALTERNATİF)

Eğer Davidcariboo verisi erişilemezse:

**Kombinasyon:**
1. **Ewenme (2021-2022)** - Mevcut ve kolay erişilebilir
2. **Manuel veri toplama (2023-2024)** - Önemli transferler için
3. **FBref proxy metrikleri** - Piyasa değeri yerine performans metrikleri

**Avantajları:**
- Hemen başlanabilir
- Ewenme verisi zaten indirildi
- 2021-2022 için tam kapsama

**Dezavantajları:**
- 2023-2024 için eksik veri
- Manuel çalışma gerekiyor

---

### Yaklaşım 3: FBref Tabanlı Analiz (EN HIZLI)

Transfer verisi olmadan, mevcut FBref verileriyle başlama:

**Konsept:**
Takım değiştiren oyuncuları FBref verilerinden tespit edip, performans değişimini analiz etme

**Nasıl Tespit Edilir:**
```python
# Bir oyuncu farklı sezonlarda farklı takımlarda oynadıysa = transfer
# Örnek:
# 2021-22: Player X - Team A
# 2022-23: Player X - Team B  → Transfer!
```

**Avantajları:**
- ✅ Hemen başlanabilir
- ✅ Ek veri indirme gerektirmez
- ✅ Transfer bedeli olmasa da performans analizi yapılabilir

**Dezavantajları:**
- ❌ Transfer bedelleri yok
- ❌ Piyasa değerleri yok
- ❌ Ekonomik analiz yapılamaz

**Uygun Olduğu Durum:**
- İlk proje (Transfer Başarı Tahmini) için yeterli
- İkinci proje (Ekonomik Verimlilik) için ek veri gerekecek

---

## 💡 Önerilen Eylem Planı

### Aşama 1: Hızlı Başlangıç (1-2 gün)
**Yaklaşım 3'ü uygula:**
1. ✅ FBref verilerinden takım değişikliklerini tespit et
2. ✅ Transfer öncesi ve sonrası performansı karşılaştır
3. ✅ İlk EDA ve görselleştirmeler yap
4. ✅ Baseline model kur

**Çıktı:**
- Transfer başarı tahmini için çalışan bir prototip
- Performans bazlı analiz
- İlk bulgular ve insights

### Aşama 2: Veri Zenginleştirme (3-5 gün)
**Yaklaşım 1 veya 2'yi uygula:**
1. Davidcariboo veri setini indir (manuel veya DVC)
2. Transfer bedelleri ve piyasa değerlerini entegre et
3. Modeli zenginleştirilmiş verilerle yeniden eğit
4. Ekonomik analiz için hazırlık yap

**Çıktı:**
- Tam özellikli transfer başarı tahmini modeli
- Ekonomik verimlilik analizi için veri altyapısı

### Aşama 3: İkinci Proje (5-7 gün)
**Ekonomik Verimlilik Analizi:**
1. Transfer bedeli vs performans analizi
2. Value-for-Money (VfM) hesaplama
3. DEA (Data Envelopment Analysis) uygulama
4. Regression modelleri ile verimlilik tahmini

---

## 🚀 Hemen Başlamak İçin

### Seçenek A: Hızlı Prototip (Önerilen)
```bash
# FBref verilerini kullanarak hemen başla
# Transfer tespiti ve performans analizi
# 2-3 saat içinde ilk sonuçlar
```

**Avantajlar:**
- Hızlı ilerleme
- Somut sonuçlar
- Veri entegrasyonu sorunları yok

### Seçenek B: Tam Veri Toplama
```bash
# Davidcariboo veri setini manuel indir
# Kaggle'dan ZIP olarak
# Sonra entegrasyon yap
```

**Avantajlar:**
- Kapsamlı analiz
- Ekonomik metrikler
- Her iki proje için hazır

---

## 📝 Karar Noktası

**Soru:** Hangi yaklaşımla devam etmek istersiniz?

**A) Hızlı Başlangıç** (Yaklaşım 3)
- FBref verilerini kullan
- Transfer tespiti yap
- Performans analizi yap
- ⏱️ 2-3 saat

**B) Tam Veri Entegrasyonu** (Yaklaşım 1)
- Davidcariboo veri setini indir
- Transfer bedelleri ekle
- Kapsamlı analiz
- ⏱️ 1-2 gün

**C) Hibrit Yaklaşım** (Yaklaşım 2)
- Ewenme + Manuel toplama
- 2021-2022 için tam veri
- 2023-2024 için kısmi veri
- ⏱️ 3-4 saat

---

**Önerim:** **Seçenek A** ile başlayıp, sonra **Seçenek B**'ye geçmek. Bu şekilde hem hızlı ilerleme sağlarız hem de sonradan veriyi zenginleştirebiliriz.


