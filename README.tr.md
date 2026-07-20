# voleykocai-llm-finetuning

🇬🇧 [English version](README.md)

**VoleykoçAI**: Türkçe konuşan bir voleybol antrenörlük asistanı. Bu depo bir yapay zekâ dersi ödevinin dört parçasını bir arada tutuyor: bir veri seti, bir BPE tokenizer, bir LoRA adaptörü ve bağımsız bir kimlik eğitimi.

Tema olarak voleybolu seçtim çünkü ilk üç ödev aynı alandan beslenince tokenizer da fine-tune da anlamlı oluyor: veri setinin dili tokenizer'ın korpusu, tokenizer'ın öğrendiği terimler de modelin konuştuğu dil.

---

## Teslim linkleri

| Ödev | İstenen | Hugging Face | Depodaki yer |
|---|---|---|---|
| 1 | Veri seti | [`voleykoc-antrenorluk-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-antrenorluk-tr) | [`01-dataset/`](01-dataset/) |
| 2 | BPE tokenizer | [`voleykoc-bpe-tokenizer`](https://huggingface.co/berkcangumusisik/voleykoc-bpe-tokenizer) | [`02-tokenizer/`](02-tokenizer/) |
| 3 | LoRA adaptörü | [`voleykoc-qwen3-4b-lora`](https://huggingface.co/berkcangumusisik/voleykoc-qwen3-4b-lora) | [`03-finetune/`](03-finetune/) |
| Ek | Kimlik eğitimi | [`voleykoc-identity-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-identity-tr) · [`voleykoc-identity-lora`](https://huggingface.co/berkcangumusisik/voleykoc-identity-lora) | [`04-identity/`](04-identity/) |

Yükleme adımlarının tamamı: [HUGGINGFACE.md](HUGGINGFACE.md)

---

## Kendin çalıştırmak istersen

```bash
# 0) Ortam kur (Python 3.11+)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Veriyi web'den topla                    [internet gerekir, ~2 dk]
python 01-dataset/scrape.py

# 2) Elle yazılmış 20 seedu çoğalt          [ANTHROPIC_API_KEY varsa model modu]
python 01-dataset/augment.py

# 3) Veri setini birleştir ve doğrula
python 01-dataset/build_dataset.py           # -> data/train.jsonl

# 4) BPE tokenizer'ı eğit
python 02-tokenizer/train_tokenizer.py       # -> 02-tokenizer/voleykoc-bpe-tokenizer/

# 5) Kimlik veri setini üret (ek ödev)
python 04-identity/build_identity.py         # -> data/identity/*.jsonl

# 6) Hugging Face'e yükle                    [token gerekir]
python 01-dataset/upload.py
python 02-tokenizer/upload.py
python 04-identity/upload.py

# 7) Fine-tune                               [Colab + T4 GPU gerekir]
#    03-finetune/finetune_voleykoc.ipynb ve
#    04-identity/finetune_identity.ipynb dosyalarını Colab'a yükleyip çalıştır
```

1–6 arası adımlar bu makinede çalışır. 7. adım çalışmaz: sebebini aşağıda "Bu makinede neyin çalışmadığı" bölümünde anlattım.

---

## Veri nereden geldi

Üç kaynak, `data/train.jsonl` içinde `source` alanıyla işaretli:

| Kaynak | Örnek | Nasıl |
|---|---:|---|
| `wikipedia` | 96 | Türkçe Wikipedia'nın 17 voleybol sayfası |
| `synthetic` | 60 | Elle yazdığım 20 seed örnekten çoğaltma |
| `tvf` | 10 | tvf.org.tr haber sayfaları |
| **Toplam** | **166** | dedupe sonrası |

**Wikipedia sayfaları:** Voleybol, Plaj voleybolu, Oturarak voleybol, FIVB, CEV, TVF, Türkiye kadın/erkek millî takımları, Sultanlar Ligi, Efeler Ligi, Kadınlar 1. Ligi, 1. Lig, Fenerbahçe, Eczacıbaşı, Galatasaray, Beşiktaş (kadın), Halkbank, Arkas.

Sayfa listesini MediaWiki API'siyle tek tek doğruladım. Umduğum bazı başlıklar (Smaç, Manşet, Servis) TR Wikipedia'da ayrı sayfa değil: teknik terimler ana Voleybol makalesinin içinde geçiyor. Ayrıca "Pasör" gibi başlıklar Voleybol'a yönlendirme yapıyor; `scrape.py` sayfanın gerçek başlığını okuyup aynı makaleyi iki kez indirmesini engelliyor.

**TVF:** Federasyonun oyun kuralları kitapçığı sitede yalnızca PDF olarak yayımlanmış, HTML sayfası yok. Bu yüzden haber akışını geziyorum: önce indeks sayfalarından `/icerik/...` linklerini topluyorum, sonra tek tek haberlere giriyorum.

**robots.txt:** İki kaynağın da izin verdiğini hem elle hem de scriptin içinden doğruladım: `tr.wikipedia.org` yalnızca `/w/` ve `/wiki/Special:` yollarını kapatıyor, `tvf.org.tr` ise `User-agent: * / Allow: /` diyor. `scrape.py` her çalıştığında bunu tekrar kontrol eder; kural değişmişse o kaynağı atlar. İstekler arasında 1 saniye bekliyorum.

**Lisans:** Wikipedia içeriği **CC BY-SA 4.0** altındadır; ondan türetilen veri seti satırları da bu lisansa tabidir. TVF içeriği tvf.org.tr'ye aittir, burada yalnızca eğitim amaçlı ve dönüştürülmüş biçimde kullanılmıştır.

---

## Ödev 1: Veri seti

**Şema.** Ödev, `alibayram/identity_finetune_magibu_q3` veri setindeki formata uymayı istiyor. O depo erişim onayı istediği için şemayı aynı ekibin açık deposundan (`magibu/turkish-multi-turn-dialog-dataset`) aldım:

```json
{"system": "...", "source": "wikipedia",
 "conversations": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}],
 "num_turns": 2}
```

**Scraping'den soru-cevaba.** Wikipedia'nın HTML'i düz: başlıklar ve paragraflar aynı seviyede sıralı duruyor. `scrape.py` sırayla yürüyüp bir başlık görünce yeni bölüm açıyor, paragraf görünce içinde bulunduğu bölüme ekliyor. `build_dataset.py` sonra her bölümü bir soruya çeviriyor: bölüm başlığından şablonla soru üretip paragrafları cevap yapıyor.

Kalıp seçimi rastgele değil, başlığın karakter toplamına göre. Böylece script iki kez çalıştığında aynı sonucu veriyor:

```python
def pick(templates: list[str], key: str) -> str:
    """Anahtara göre deterministik kalıp seçimi (rastgele değil)."""
    return templates[sum(ord(c) for c in key) % len(templates)]
```

Uzun bölümleri tek cevaba sıkıştırıp gerisini atmak yerine parçalara bölüyorum: hem veri kaybı olmuyor hem örnek sayısı artıyor. İkinci, üçüncü parçanın sorusuna `(devamı, 2. bölüm)` ekliyorum ki dedupe onları aynı soru sanıp elemesin.

**Sentetik taraf.** `seeds.jsonl`'da elle yazdığım 20 antrenörlük soru-cevabı var: 5-1 rotasyonu, manşet tekniği, smaç yaklaşımı ritmi, libero kuralları, U14 antrenman planı, sıçrama çalışması, sakatlık önleme, maç günü beslenmesi gibi. Hepsi gerçek antrenörlük bilgisi; şablon değil.

`augment.py` bunları çoğaltıyor. İki modu var: `ANTHROPIC_API_KEY` varsa modele veriyor ve gerçekten yeni soru-cevap çiftleri üretiyor; yoksa soruyu şablonlarla yeniden yazıp cevabı seed örnekten kopyalıyor.

> **Bu depodaki `train.jsonl` çevrimdışı modda üretildi.** Yani sentetik kısımdaki 60 örneğin cevapları 20 seed örnekten kopya. `reports/dataset_stats.md` bunu benzersiz cevap oranıyla raporluyor (166'da 125, %75). Bir API anahtarıyla `augment.py`'yi tekrar çalıştırırsan hem oran hem toplam örnek sayısı ciddi şekilde yükselir.

**Doğrulama.** `build_dataset.py` yazmadan önce her satırı kontrol ediyor: alan kümesi doğru mu, roller user→assistant sırasında mı, `num_turns` mesaj sayısıyla tutuyor mu, boş içerik var mı. Bir hata bulursa dosyayı yazmıyor ve çıkıyor.

---

## Ödev 2: BPE tokenizer

Depoda BPE'nin **iki** hâli var ve bunun bir sebebi var.

`bpe.py`, önceki ödevimde ([countries-bpe-tokenizer](https://github.com/berkcangumusisik/countries-bpe-tokenizer)) sıfırdan yazdığım implementasyon: saf standart kütüphane, `train`/`encode`/`decode`/`save`/`load`. Algoritmayı bildiğimin kanıtı olarak buraya taşıdım. Ama kendi JSON formatını yazıyor, Hugging Face onu okuyamıyor.

`train_tokenizer.py` ise yayımlanan tokenizer'ı `tokenizers` kütüphanesiyle eğitiyor: aynı algoritma (byte-level BPE), ama `AutoTokenizer.from_pretrained()` ile geri yüklenebilen dosyalar.

**Korpus.** Veri setinin tüm metni artı scrape edilen ham voleybol yazıları: 214.015 karakter / 29.532 kelime. Sistem mesajı her satırda aynı olduğu için korpusa yalnızca bir kez giriyor: yoksa tokenizer o cümleyi ezberliyor.

**Sonuç.** Hedef sözlük 16.000'di, gerçekleşen 7.532. Korpus küçük ve `min_frequency=2` uyguladığım için birleştirmeler erken tükendi. Sözlüğü yapay olarak şişirmek yerine gerçek sayıyı bıraktım.

**Alan verimliliği.** Aynı 10 Türkçe voleybol cümlesinde:

| | Toplam token |
|---|---:|
| VoleykoçAI | **114** |
| Qwen3-4B-Instruct-2507 | 179 |

Bu alanda %36 daha az. Sözlük tamamen Türkçe voleybol metninden öğrenildiği için `pasör`, `manşet`, `rotasyon` tek parça kalıyor; Qwen'in çok dilli sözlüğü onları parçalara ayırıyor. Karşılaştırmanın adil okunuşu şu: bu, tokenizer'ın **bu alandaki** verimliliğini gösterir, genel kalitesini değil. Qwen'in sözlüğü yüzlerce dili kapsıyor, benimki tek alanı.

**Byte fallback.** Taban sözlük 256 byte olduğu için `<unk>` yok: hiç görülmemiş bir karakter (emoji, başka alfabe) gelse bile ham byte'larıyla temsil ediliyor. Test cümlelerinin 10'u da encode→decode sonrası birebir aynı döndü; içlerinden biri bilerek emoji içeriyor.

Detaylı rapor: [`reports/tokenizer_report.md`](reports/tokenizer_report.md)

---

## Ödev 3: Fine-tune

`03-finetune/finetune_voleykoc.ipynb`, Colab T4 üzerinde:

| | |
|---|---|
| Temel model | `unsloth/Qwen3-4B-Instruct-2507` |
| Yöntem | 4-bit QLoRA |
| LoRA | r=16, alpha=16, dropout=0 |
| Hedef katmanlar | q, k, v, o, gate, up, down |
| Eğitim | 3 epoch, lr=2e-4, efektif batch 8 |
| Seed | 1337 |

4B parametrenin tamamını eğitmek yerine dikkat ve MLP katmanlarına küçük düşük-ranklı matrisler ekliyorum; eğitilen parametre oranı %1'in altında kalıyor. Colab'da bu yüzden mümkün.

Notebook eğitimden **önce** ve **sonra** aynı beş soruyu soruyor ve cevapları `karsilastirma.json`'a yazıyor. Fine-tune'un gerçekten bir şey değiştirdiğini böyle gösteriyorum: kayıp eğrisine bakmak tek başına yeterli değil.

Yalnızca LoRA adaptörü yayımlanıyor (~100 MB), birleştirilmiş model değil (~8 GB). Ödev zaten adaptörü istiyor.

### Neden tokenizer fine-tune'da kullanılmadı?

Bu, depoya bakan birinin soracağı ilk soru, o yüzden ayrı başlık açtım.

Ödev 2'deki tokenizer **bağımsız bir teslim**. Ödev 3'teki adaptör onu kullanmıyor; temel modelin kendi tokenizer'ıyla eğitildi.

Sebep: bir modelin sözlüğünü değiştirmek, embedding katmanını yeniden boyutlandırmayı ve modeli baştan eğitmeyi gerektirir. Qwen3-4B'nin embedding'i 151.000 token için öğrenilmiş; onu 7.532'lik yeni bir sözlüğe geçirmek, LoRA ile ince ayarın kapsamı dışında kalan bir iş. Ödev de bunu istemiyor: üç ayrı teslim istiyor.

Yani tokenizer, "bu alanda özel bir sözlük eğitebiliyorum" iddiasını kanıtlıyor; adaptör ise "bu alanda bir modeli ince ayarlayabiliyorum" iddiasını. İkisi ayrı sorular.

---

## Ek Ödev: Kimlik eğitimi

Bu, ilk üç ödevden bağımsız ayrı bir süreç. Amaç modele voleybol bilgisi öğretmek değil, **kim olduğunu** öğretmek.

`identity_seeds.jsonl`'da 26 kimlik soru-cevabı var (15 Türkçe, 11 İngilizce), yedi kategoride: adı, yaratıcısı, görevi, yetenekleri, **sınırları**, konuştuğu dil, nasıl eğitildiği. `build_identity.py` bunları soru kalıplarıyla 182 örneğe çıkarıyor ve hocanın veri setindeki gibi `turkish` / `english` diye ikiye bölüyor.

Kimlik verisinde aynı cevabın birden çok soru biçimine bağlanması **kasıtlı**: "adın ne", "kimsin", "kendini tanıt" hepsi aynı gerçeğe çıkar ve model kimliğini bu tekrardan öğrenir. Ana ödevdeki sentetik veride bu bir kusurdu; burada yöntemin kendisi.

`sinir` kategorisini bilerek koydum: bir kimlik yalnızca modelin ne olduğuyla değil, ne *olmadığıyla* da tanımlanır. O kategorideki örnekler modele sakatlık teşhisi koymamayı ve alan dışı sorularda kullanıcıyı doğru yere yönlendirmeyi öğretiyor.

Eğitim `max_steps=60` ile sınırlı. Kimlik verisi küçük ve tekrarlı; uzun eğitim modelin voleybol bilgisini ve genel akıcılığını bozar. Notebook eğitim sonunda kimlik dışı bir soru da soruyor (`5-1 rotasyon sistemi nasıl çalışır?`): model hâlâ işini yapabiliyor mu diye.

---

## Bu makinede neyin çalışmadığı

Ödev "Unsloth üzerinden" eğitmeyi istiyor. Unsloth, Triton'a dayanıyor ve Triton macOS'ta yok: bu depo bir Apple M5 Pro üzerinde geliştirildi, yani **eğitim burada çalışmıyor**.

Bu yüzden 3. ödev ve ek ödev birer Colab notebook'u olarak duruyor. Veri, tokenizer ve yükleme adımlarının hepsi lokalde çalışır; yalnızca GPU isteyen iki adım Colab'a taşındı.

Alternatif olarak Apple Silicon'da MLX ile LoRA eğitimi gerçekten mümkün, ama ödev açıkça Unsloth dediği için o yolu tercih etmedim.

---

## Depo yapısı

```
01-dataset/      scrape.py, seeds.jsonl, augment.py, build_dataset.py, upload.py
02-tokenizer/    bpe.py (sıfırdan), train_tokenizer.py (HF formatı), upload.py
03-finetune/     finetune_voleykoc.ipynb  [Colab]
04-identity/     identity_seeds.jsonl, build_identity.py, finetune_identity.ipynb, upload.py
data/            train.jsonl, identity/*.jsonl  (raw/ gitignore'da)
reports/         dataset_stats.md, tokenizer_report.md, identity_stats.md
hf_upload.py     üç upload.py'nin paylaştığı onay + yükleme mantığı
HUGGINGFACE.md   yükleme adımları, adım adım
```

Her aşama klasöründe bir `NOTLAR.md` var: o klasördeki scriptler hangi sırayla çalışır, ne üretir, ne gerektirir. Bu README'yi baştan okumadan tek bir aşamaya dönebilmek için.

---

## Atıflar

- Türkçe Wikipedia: voleybol sayfaları (CC BY-SA 4.0)
- [Türkiye Voleybol Federasyonu](https://tvf.org.tr)
- [Unsloth](https://github.com/unslothai/unsloth): hızlı LoRA fine-tune
- [`alibayram/identity_finetune_magibu_q3`](https://huggingface.co/datasets/alibayram/identity_finetune_magibu_q3): veri seti şeması referansı
- [`magibu/turkish-multi-turn-dialog-dataset`](https://huggingface.co/datasets/magibu/turkish-multi-turn-dialog-dataset): şemanın açık örneği
- [Qwen3](https://huggingface.co/Qwen): temel model
- Kendi önceki ödevim: [countries-bpe-tokenizer](https://github.com/berkcangumusisik/countries-bpe-tokenizer): `bpe.py` oradan geldi

## Lisans

[MIT](LICENSE). Veri kaynaklarının kendi lisansları için yukarıdaki "Veri nereden geldi" bölümüne bak.
