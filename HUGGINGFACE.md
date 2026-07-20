# Hugging Face'e yükleme: adım adım

Bu dosyayı ödevi teslim ettikten aylar sonra açtığımda adımları hatırlamak zorunda kalmayayım diye yazdım.

Beş repo yayımlanacak:

| # | Tür | Repo | Nereden yükleniyor |
|---|---|---|---|
| 1 | dataset | `voleykoc-antrenorluk-tr` | `01-dataset/upload.py` |
| 2 | model | `voleykoc-bpe-tokenizer` | `02-tokenizer/upload.py` |
| 3 | model | `voleykoc-qwen3-4b-lora` | Colab notebook'un son hücresi |
| + | dataset | `voleykoc-identity-tr` | `04-identity/upload.py` |
| + | model | `voleykoc-identity-lora` | Colab notebook'un son hücresi |

İlk üçü ödevin zorunlu teslimleri, son ikisi ek ödev.

---

## Adım 0: Token oluştur (tarayıcıda, bir kez)

1. https://huggingface.co/settings/tokens adresine git
2. **Create new token** → sekme: **Write**
3. İsim: `voleykocai-odev` · Rol: **Write**
   - Read yetkili token yüklemeye izin vermez, hata alırsın.
4. `hf_...` ile başlayan değeri kopyala. **Bu değer sadece bir kez gösterilir.**

---

## Adım 1: Giriş yap (terminalde, bir kez)

```bash
cd ~/Documents/GitHub/voleykocai-llm-finetuning
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

hf auth login        # token'ı yapıştır; "Add token as git credential?" → y
hf auth whoami       # çıktıda berkcangumusisik görmelisin
```

`hf: command not found` alırsan `huggingface_hub` sürümün eskidir; aynı işi `huggingface-cli login` / `huggingface-cli whoami` yapar.

Token'ı depoya, notebook'a veya herhangi bir dosyaya yazma. `huggingface_hub` onu kendi saklama alanında tutar, scriptler oradan okur.

---

## Adım 2: Dosyaları üret

Yükleyecek bir şey olması için önce üretim adımlarını çalıştır:

```bash
python 01-dataset/scrape.py           # ~2 dk, internet gerekir
python 01-dataset/augment.py          # ANTHROPIC_API_KEY varsa model modu
python 01-dataset/build_dataset.py    # -> data/train.jsonl
python 02-tokenizer/train_tokenizer.py  # -> 02-tokenizer/voleykoc-bpe-tokenizer/
python 04-identity/build_identity.py  # -> data/identity/*.jsonl
```

---

## Adım 3: Yükle

```bash
python 01-dataset/upload.py
python 02-tokenizer/upload.py
python 04-identity/upload.py
```

Her script yüklemeden **önce** ne yapacağını yazdırıp onay bekler:

```
Yüklenecek repo : berkcangumusisik/voleykoc-antrenorluk-tr (dataset, public)
Dosyalar        :
  - train.jsonl                  (185.4 KB, 166 satır)
  - README.md                    (4.1 KB)

Devam? [e/h]:
```

`h` yazarsan hiçbir şey yüklenmez. Repo yoksa otomatik oluşturulur (public), varsa üzerine yazılır.

Ayrıca giriş yaptığın hesap `berkcangumusisik` değilse script durur: yanlış hesaba yüklemeyi önlemek için.

---

## Adım 4: LoRA adaptörlerini Colab'dan yükle

Adaptörler GPU gerektirdiği için lokalde üretilmiyor; push işlemi notebook'un içinden yapılıyor.

1. `03-finetune/finetune_voleykoc.ipynb` dosyasını Colab'a yükle
2. `Runtime → Change runtime type → T4 GPU`
3. Soldaki 🔑 **Secrets** panelinden `HF_TOKEN` adıyla token'ı ekle, **Notebook access** anahtarını aç
4. Hücreleri sırayla çalıştır

Notebook token'ı şöyle okur: düz metin yazmaz:

```python
from google.colab import userdata
model.push_to_hub("berkcangumusisik/voleykoc-qwen3-4b-lora", token=userdata.get("HF_TOKEN"))
```

Ek ödev için aynısını `04-identity/finetune_identity.ipynb` ile tekrarla.

**Token'ı hücreye düz metin yazma**: notebook GitHub'a commit'lenecek, token da onunla birlikte sızar.

---

## Adım 5: Doğrula

Beş linki tarayıcıda aç:

- **Dataset sayfalarında** Data Studio / viewer satırları gösteriyor mu? Göstermiyorsa `train.jsonl` şeması bozuktur ya da README'deki `configs` bloğu eksiktir.
- **Tokenizer reposunda** `tokenizer.json` ve `tokenizer_config.json` var mı?
- **LoRA repolarında** `adapter_model.safetensors` ve `adapter_config.json` var mı? Model kartında `base_model` görünüyor mu?

Son kontrol: hiçbir şey lokalden okunmadan çalışmalı:

```bash
python -c "from datasets import load_dataset; print(load_dataset('berkcangumusisik/voleykoc-antrenorluk-tr'))"
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('berkcangumusisik/voleykoc-bpe-tokenizer'); print(t.tokenize('Pasör çaprazı hücumda ne yapar?'))"
```

---

## Takılırsan

| Belirti | Sebep / çözüm |
|---|---|
| `401 Unauthorized` | Token Read yetkili ya da süresi dolmuş → yeni Write token al, `hf auth login` tekrar |
| Script "giriş yapılmamış" diyor | `hf auth login` çalıştır; venv aktif mi kontrol et |
| Script "yanlış hesap" diyor | Farklı bir HF hesabıyla giriş yapılmış. Doğru hesapla gir, ya da `hf_upload.py` içindeki `HF_USER` değerini değiştir |
| Dataset viewer boş | `train.jsonl` satırlarından biri şemaya uymuyor → `build_dataset.py` tekrar çalıştır (doğrulama adımı hataları yakalar) |
| Repo private açıldı | Repo sayfası → Settings → Change visibility → Public |
| Colab `OutOfMemoryError` | `MAX_SEQ_LENGTH` 2048 → 1024, ya da temel modeli `unsloth/Qwen3-1.7B-Instruct` yap |
| Colab `HF_TOKEN` bulunamıyor | Secrets panelinde **Notebook access** anahtarı açık mı? |
