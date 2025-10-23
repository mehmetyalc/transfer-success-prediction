# Veri Toplama Raporu
## Transfer Başarı Tahmini Projesi

**Tarih:** 23 Ekim 2025  
**Proje:** Football Transfer Success Prediction

---

## 📊 Toplanan Veri Özeti

### Genel İstatistikler

| Metrik | Değer |
|--------|-------|
| **Toplam Dosya Sayısı** | 90 CSV dosyası |
| **Toplam Veri Boyutu** | 6.6 MB |
| **Toplam Oyuncu Kaydı** | 8,693 |
| **Toplam Takım Kaydı** | 322 |
| **Toplam Maç Kaydı** | 5,415 |
| **Veri Toplama Süresi** | 11 dakika 27 saniye |

### Veri Kaynakları

**Başarıyla Toplanan:**
- ✅ **FBref** - Oyuncu ve takım performans istatistikleri
  - Oyuncu sezon istatistikleri (standard, shooting, passing, playing time)
  - Takım sezon istatistikleri
  - Maç programı ve sonuçları

**Alternatif Kaynak Gereksinimi:**
- ⚠️ **Transfermarkt** - Transfer bedelleri ve piyasa değerleri
  - Doğrudan scraping karmaşık (dinamik içerik)
  - Alternatif: Kaggle/GitHub'dan hazır veri setleri kullanılabilir

---

## 🏆 Lig ve Sezon Bazında Veri

### Premier League (İngiltere)

| Sezon | Oyuncu Sayısı | Takım Sayısı | Maç Sayısı |
|-------|---------------|--------------|------------|
| 2021-22 | 548 | 22 | 380 |
| 2022-23 | 571 | 22 | 380 |
| 2023-24 | 582 | 22 | 380 |
| **Toplam** | **1,701** | **66** | **1,140** |

### La Liga (İspanya)

| Sezon | Oyuncu Sayısı | Takım Sayısı | Maç Sayısı |
|-------|---------------|--------------|------------|
| 2021-22 | 619 | 22 | 380 |
| 2022-23 | 598 | 22 | 380 |
| 2023-24 | 611 | 22 | 380 |
| **Toplam** | **1,828** | **66** | **1,140** |

### Serie A (İtalya)

| Sezon | Oyuncu Sayısı | Takım Sayısı | Maç Sayısı |
|-------|---------------|--------------|------------|
| 2021-22 | 634 | 22 | 380 |
| 2022-23 | 605 | 22 | 381 |
| 2023-24 | 618 | 22 | 380 |
| **Toplam** | **1,857** | **66** | **1,141** |

### Bundesliga (Almanya)

| Sezon | Oyuncu Sayısı | Takım Sayısı | Maç Sayısı |
|-------|---------------|--------------|------------|
| 2021-22 | 525 | 20 | 308 |
| 2022-23 | 517 | 20 | 308 |
| 2023-24 | 509 | 20 | 308 |
| **Toplam** | **1,551** | **60** | **924** |

### Ligue 1 (Fransa)

| Sezon | Oyuncu Sayısı | Takım Sayısı | Maç Sayısı |
|-------|---------------|--------------|------------|
| 2021-22 | 606 | 22 | 382 |
| 2022-23 | 608 | 22 | 380 |
| 2023-24 | 542 | 20 | 308 |
| **Toplam** | **1,756** | **64** | **1,070** |

---

## 📈 Toplanan Metrikler

### Oyuncu İstatistikleri (Player Stats)

Her oyuncu için 4 farklı kategori altında veri toplandı:

#### 1. Standard Stats (Standart İstatistikler)
- **Oynama Süresi:** Maç sayısı (MP), başlangıç (Starts), dakika (Min), 90'lar (90s)
- **Performans:** Goller (Gls), asistler (Ast), G+A, penaltılar (PK, PKatt)
- **Disiplin:** Sarı kart (CrdY), kırmızı kart (CrdR)
- **İleri Metrikler:** xG (Expected Goals), npxG (non-penalty xG), xAG (Expected Assisted Goals)
- **İlerleme:** PrgC (Progressive Carries), PrgP (Progressive Passes), PrgR (Progressive Receptions)
- **Per 90 Normalize Metrikler:** Tüm metrikler 90 dakika başına normalize edilmiş

#### 2. Shooting Stats (Şut İstatistikleri)
- Şut sayısı ve türleri
- Şut isabeti
- Gol başına şut
- xG detayları

#### 3. Passing Stats (Pas İstatistikleri)
- Toplam pas sayısı
- Pas başarı oranı
- Pas mesafeleri
- Asist ve xAG detayları
- Anahtar pas sayısı

#### 4. Playing Time Stats (Oynama Süresi İstatistikleri)
- Maç başına dakika
- Başlangıç/yedek oranı
- Yaş ve pozisyon bilgisi

### Takım İstatistikleri (Team Stats)

- Takım performans metrikleri
- Lig sıralaması
- Gol atılan/yenilen
- xG ve xGA (Expected Goals Against)

### Maç Programı (Schedule)

- Maç tarihleri
- Ev sahibi ve deplasman takımları
- Maç sonuçları
- Skor bilgileri

---

## ✅ Veri Kalitesi

### Pozitif Bulgular

✅ **Duplikasyon Yok:** Hiçbir dosyada tekrarlanan satır bulunmadı  
✅ **Negatif Değer Yok:** Sayısal sütunlarda negatif değer tespit edilmedi  
✅ **Tutarlı Format:** Tüm dosyalar aynı yapıda ve tutarlı  
✅ **Düşük Eksik Veri:** Eksik veri oranı %1'in altında

### Eksik Veri Analizi

En fazla eksik veri içeren sütunlar:

| Sütun | Eksik Veri Oranı |
|-------|------------------|
| nation | 0.5% |
| age | 0.4% |
| born | 0.4% |
| pos | 0.4% |

**Not:** Bu oranlar ihmal edilebilir seviyededir ve veri temizleme aşamasında kolayca ele alınabilir.

---

## 🔧 Kullanılan Teknolojiler

### Veri Toplama

**Python Kütüphaneleri:**
- `soccerdata` (v1.8.7) - FBref scraping için özel kütüphane
- `pandas` - Veri işleme
- `beautifulsoup4` - HTML parsing
- `requests` - HTTP istekleri
- `selenium` - Dinamik içerik için

**Veri Kaynağı:**
- [FBref.com](https://fbref.com/) - Advanced football statistics

### Veri Depolama

- **Format:** CSV (Comma-Separated Values)
- **Encoding:** UTF-8
- **Yapı:** Hierarchical (lig → sezon → veri tipi)

---

## 📁 Dosya Yapısı

```
data/raw/fbref/
├── England_2122_player_stats_standard.csv
├── England_2122_player_stats_shooting.csv
├── England_2122_player_stats_passing.csv
├── England_2122_player_stats_playing_time.csv
├── England_2122_team_stats.csv
├── England_2122_schedule.csv
├── ... (diğer sezonlar ve ligler için benzer)
└── data_collection_summary.csv
```

**Dosya Adlandırma Konvansiyonu:**
```
{League}_{Season}_{DataType}_{StatType}.csv
```

**Örnek:**
- `England_2122_player_stats_standard.csv`
- `Spain_2324_team_stats.csv`
- `Germany_2223_schedule.csv`

---

## 🎯 Sonraki Adımlar

### 1. Transfer Verisi Entegrasyonu

**Önerilen Yaklaşım:**
Transfermarkt'tan doğrudan scraping yerine hazır veri setleri kullanılması önerilir:

**Kaggle Veri Setleri:**
- [Football Transfer Dataset](https://www.kaggle.com/datasets/mexwell/football-transfer-dataset) - 2009-2021 arası transferler
- [Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores) - Haftalık güncellenen kapsamlı veri
- [Football Transfers (Major Leagues)](https://www.kaggle.com/datasets/ashishmotwani/football-transfers) - Büyük ligler

**GitHub Veri Setleri:**
- [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) - Hazır ve temiz transfer verileri
- [ewenme/transfers](https://github.com/ewenme/transfers) - Avrupa ligleri transfer verileri

### 2. Veri Temizleme ve Preprocessing

**Yapılacaklar:**
- Eksik verilerin doldurulması/çıkarılması
- Sütun isimlerinin standardizasyonu
- Veri tiplerinin düzeltilmesi
- Outlier (aykırı değer) analizi ve ele alınması

### 3. Veri Entegrasyonu

**Hedef:**
FBref performans verileri ile transfer verilerinin birleştirilmesi

**Eşleştirme Kriterleri:**
- Oyuncu isimleri (fuzzy matching ile)
- Takım isimleri
- Sezon bilgisi
- Pozisyon bilgisi

### 4. Feature Engineering

**Oluşturulacak Özellikler:**
- Transfer öncesi performans metrikleri (son sezon ortalamaları)
- Transfer sonrası performans metrikleri (yeni takımda ilk sezon)
- Performans değişimi (delta)
- Lig geçiş türü (aynı lig, lig seviyesi değişimi)
- Yaş kategorisi (genç, prime, veteran)
- Transfer bedeli kategorisi

### 5. Model Geliştirme

**Hedef Değişken Seçenekleri:**
1. **Regression:** Transfer sonrası gol/90 dk, asist/90 dk, rating
2. **Classification:** Başarılı/başarısız transfer (eşik değere göre)

**Özellikler (Features):**
- Transfer öncesi performans metrikleri
- Oyuncu özellikleri (yaş, pozisyon, milliyet)
- Transfer detayları (bedel, lig geçişi)
- Takım özellikleri

---

## 📝 Notlar ve Öneriler

### Veri Toplama Hakkında

1. **FBref Verisi Yeterli:** FBref'ten toplanan veri, performans analizi için oldukça kapsamlı ve kaliteli. xG, xAG gibi ileri metrikler de mevcut.

2. **Transfer Verisi İçin Alternatif:** Transfermarkt'tan doğrudan scraping yerine, Kaggle veya GitHub'dan hazır veri setlerinin kullanılması daha pratik ve güvenilir olacaktır.

3. **Veri Güncelliği:** Toplanan veriler 2021-2024 arası 3 sezonu kapsıyor. Bu, makine öğrenmesi modeli için yeterli veri miktarı sağlıyor.

4. **Lig Çeşitliliği:** 5 farklı Avrupa ligi, farklı oyun tarzları ve lig seviyeleri için çeşitlilik sağlıyor.

### Proje İçin Öneriler

1. **Veri Zenginleştirme:** Transfer verilerini ekledikten sonra, oyuncu piyasa değeri, maaş bilgileri gibi ekonomik verileri de dahil etmek projeyi güçlendirecektir.

2. **Temporal Analysis:** Transfer öncesi sadece son sezon değil, son 2-3 sezonun ortalamasını almak daha sağlıklı tahminler verebilir.

3. **Pozisyon Bazlı Modeller:** Farklı pozisyonlar için ayrı modeller geliştirmek (forvet, orta saha, defans) daha iyi sonuçlar verebilir.

4. **Lig Seviyesi Normalizasyonu:** Farklı liglerin seviyelerini dikkate alarak metrikleri normalize etmek önemli.

---

## 📚 Kaynaklar

### Veri Kaynakları
- [FBref](https://fbref.com/) - Advanced football statistics
- [soccerdata Python Library](https://github.com/probberechts/soccerdata) - Football data scraping library

### Önerilen Transfer Veri Setleri
- [Kaggle: Football Transfer Dataset](https://www.kaggle.com/datasets/mexwell/football-transfer-dataset)
- [Kaggle: Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcariboo/player-scores)
- [GitHub: transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)

### Akademik Referanslar
- Payyappalli, V. M., & Zhuang, J. (2019). A data-driven integer programming model for soccer clubs' decision making on player transfers. *Environment Systems and Decisions*, 39(4), 466-481.
- Wand, T. (2022). Analysis of the football transfer market network. *Journal of Statistical Physics*, 187(1), 1-18.

---

**Rapor Tarihi:** 23 Ekim 2025  
**Hazırlayan:** Data Collection Pipeline  
**Durum:** ✅ Başarıyla Tamamlandı

