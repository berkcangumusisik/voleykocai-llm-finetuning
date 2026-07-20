# BPE tokenizer raporu

## Korpus

- Veri seti satırı: 166
- Ham scrape paragrafı: 248
- Toplam: 214,015 karakter, 29,532 kelime

## Tokenizer

- Algoritma: byte-level BPE (`tokenizers` kütüphanesi)
- Hedef sözlük boyutu: 16000
- Gerçekleşen sözlük boyutu: 7532
- Özel tokenlar: <|endoftext|>, <|im_start|>, <|im_end|>, <pad>
- `<unk>` yok: byte fallback sayesinde her UTF-8 metin ayrıştırılabilir

## Encode / decode round-trip

| Cümle | Token | Aynı mı? |
|---|---:|:--:|
| Pasör çaprazı hücumda ne yapar? | 7 | ✅ |
| Manşet pasında dirsekler kilitli kalmalı. | 12 | ✅ |
| 5-1 rotasyonunda tek pasör vardır. | 9 | ✅ |
| Libero blok yapamaz ve servis atamaz. | 7 | ✅ |
| Setler 25 sayıya, tie-break 15 sayıya oynanır. | 13 | ✅ |
| Smaç yaklaşımı sol-sağ-sol-sağ ritmindedir. | 16 | ✅ |
| Türkiye Kadın Millî Voleybol Takımı'na 'Filenin Sult... | 13 | ✅ |
| Sıçrama yüksekliğini artırmak için pliometrik çalışın. | 9 | ✅ |
| Ağırlık ayak parmak uçlarında, dizler bükülü olmalı. 🏐 | 17 | ✅ |
| Eczacıbaşı ve VakıfBank Sultanlar Ligi'nin köklü kul... | 11 | ✅ |

**10/10** cümle kayıpsız geri döndü.

## Qwen3 tokenizer ile karşılaştırma

Aynı Türkçe voleybol cümlelerini kaç token'a bölüyoruz? Az token = bu alanda daha verimli.

| Cümle | VoleykoçAI | Qwen3 |
|---|---:|---:|
| Pasör çaprazı hücumda ne yapar? | 7 | 13 |
| Manşet pasında dirsekler kilitli kalmalı. | 12 | 16 |
| 5-1 rotasyonunda tek pasör vardır. | 9 | 12 |
| Libero blok yapamaz ve servis atamaz. | 7 | 14 |
| Setler 25 sayıya, tie-break 15 sayıya oyna... | 13 | 20 |
| Smaç yaklaşımı sol-sağ-sol-sağ ritmindedir. | 16 | 17 |
| Türkiye Kadın Millî Voleybol Takımı'na 'Fi... | 13 | 22 |
| Sıçrama yüksekliğini artırmak için pliomet... | 9 | 18 |
| Ağırlık ayak parmak uçlarında, dizler bükü... | 17 | 22 |
| Eczacıbaşı ve VakıfBank Sultanlar Ligi'nin... | 11 | 25 |
| **Toplam** | **114** | **179** |

Kendi tokenizer'ım aynı metni **%36.3 az** token'a bölüyor. Sözlüğü tamamen Türkçe voleybol metninden öğrendiği için 'pasör', 'manşet', 'rotasyon' gibi terimler tek parça kalıyor; Qwen'in çok dilli sözlüğü onları parçalara ayırıyor.

> Not: bu karşılaştırma tokenizer'ın **bu alandaki** verimliliğini gösterir, genel kaliteyi değil. Qwen'in sözlüğü yüzlerce dili kapsıyor, benimki tek alanı.
