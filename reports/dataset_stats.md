# Veri seti istatistikleri

Toplam örnek: **166** (dedupe öncesi 172, elenen 6)

## Kaynak dağılımı

| Kaynak | Örnek | Oran |
|---|---:|---:|
| wikipedia | 96 | 57.8% |
| synthetic | 60 | 36.1% |
| tvf | 10 | 6.0% |

## Uzunluklar (karakter)

| | En kısa | Ortalama | En uzun |
|---|---:|---:|---:|
| Soru | 21 | 80 | 166 |
| Cevap | 122 | 652 | 2125 |

## Benzersizlik

Sorular dedupe edildi, hepsi benzersiz. Benzersiz **cevap** sayısı: 125 / 166 (75.3%).

> Cevap tekrarı yüksek. Sebebi büyük ihtimalle `augment.py`'nin çevrimdışı modu: orada cevap tohumdan aynen kopyalanıyor. `ANTHROPIC_API_KEY` verip model moduyla yeniden üretirsen bu oran yükselir.
