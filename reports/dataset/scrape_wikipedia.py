"""
Türkiye'deki tarihi yerler hakkında Vikipedi'den (tr.wikipedia.org) veri toplayan
web scraping script'i.

NOT: Bu script'i KENDİ bilgisayarında/ortamında çalıştırman gerekiyor; bu chat
ortamının ağ erişimi Wikipedia'ya kapalı olduğu için burada çalıştıramıyorum.
Kurulum:
    pip install requests beautifulsoup4 wikipedia-api

Çıktı: dataset/raw/<yer_adi>.txt  (her tarihi yer için ham metin)
"""

import os
import time
import requests
from bs4 import BeautifulSoup

# İstediğin kadar yer ekleyip çıkarabilirsin. Sayfa başlıkları tr.wikipedia.org
# ile birebir eşleşmeli (Türkçe karakterlere dikkat).
HISTORICAL_PLACES = [
    "Göbeklitepe",
    "Efes",
    "Ayasofya",
    "Topkapı Sarayı",
    "Pamukkale",
    "Truva",
    "Nemrut Dağı",
    "Sümela Manastırı",
    "Aspendos",
    "Perge",
    "Hattuşa",
    "Bergama",
    "Meryem Ana Evi",
    "Kapadokya",
    "Anıtkabir",
    "Selimiye Camii",
    "Mardin",
    "Safranbolu",
    "Divriği Ulu Camii ve Darüşşifası",
    "Zeugma",
    "Alanya Kalesi",
    "Rumeli Hisarı",
    "İshak Paşa Sarayı",
    "Sardes",
    "Milet",
    "Afrodisias",
    "Ksanthos",
    "Çatalhöyük",
    "Sagalassos",
    "Harran",
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "raw")
HEADERS = {"User-Agent": "TarihiYerlerDatasetBot/1.0 (egitim amacli)"}


def fetch_wikipedia_summary(title: str) -> str:
    """Vikipedi REST API'sinden sayfanın tam metnini (HTML->text) çeker."""
    url = f"https://tr.wikipedia.org/api/rest_v1/page/html/{requests.utils.quote(title)}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Kaynakça, kutu, referans gibi gürültü bölümlerini temizle
    for tag in soup.select("table, sup, .reference, .mw-editsection, style, script"):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if len(p) > 40)
    return text


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for place in HISTORICAL_PLACES:
        out_path = os.path.join(OUT_DIR, place.replace(" ", "_") + ".txt")
        if os.path.exists(out_path):
            continue
        try:
            text = fetch_wikipedia_summary(place)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[OK] {place} -> {len(text)} karakter")
        except Exception as e:
            print(f"[HATA] {place}: {e}")
        time.sleep(1)  # Wikipedia sunucularına nazik davran


if __name__ == "__main__":
    main()
