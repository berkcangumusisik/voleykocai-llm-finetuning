#!/usr/bin/env python3
"""VoleykoçAI -- ham voleybol metinlerini web'den toplar.

Ödevin "web scraping ek puan kazandırır" maddesi için yazdım. Türkçe
Wikipedia'nın voleybol sayfalarını ve Türkiye Voleybol Federasyonu'nun açık
sayfalarını geziyorum, her sayfayı başlık/paragraf bölümlerine ayırıp
data/raw/ altına JSON olarak yazıyorum. build_dataset.py bu JSON'ları
soru-cevap çiftlerine çeviriyor.

İki kaynağın robots.txt'ini de çalıştırmadan önce kontrol ettim:
  - tr.wikipedia.org : "User-agent: *" bloğu yalnızca /w/ ve /wiki/Special:
                       yollarını kapatıyor, normal makale sayfaları serbest.
  - tvf.org.tr       : "User-agent: * / Allow: /" -- tamamen serbest.
Yine de check_robots() ile her çalıştırmada tekrar doğruluyorum; kural
değişmişse o kaynağı atlıyorum. İstekler arasında REQUEST_DELAY saniye
bekliyorum, sunucuyu yormamak için.

Run:
    python 01-dataset/scrape.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.robotparser
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

# Kendimi tanıtıyorum: kim olduğu belli olmayan bir bot olmak istemiyorum.
USER_AGENT = (
    "VoleykocAI-Odev/1.0 (ders odevi veri toplama; "
    "https://github.com/berkcangumusisik/voleykocai-llm-finetuning)"
)
REQUEST_DELAY = 1.0  # saniye, istekler arası

# Türkçe Wikipedia sayfaları. Hepsinin var olduğunu MediaWiki API'siyle tek tek
# doğruladım (Smaç, Servis (voleybol), Manşet gibi umduğum başlıklar TR
# Wikipedia'da yok -- teknik terimler ana Voleybol makalesinin içinde geçiyor).
WIKIPEDIA_PAGES = [
    # oyunun kendisi: kurallar, pozisyonlar, teknik, taktik
    "Voleybol",
    "Plaj voleybolu",
    "Oturarak voleybol",
    # kurumlar
    "Uluslararası Voleybol Federasyonu",
    "Avrupa Voleybol Konfederasyonu",
    "Türkiye Voleybol Federasyonu",
    # millî takımlar ve ligler
    "Türkiye kadın millî voleybol takımı",
    "Türkiye erkek millî voleybol takımı",
    "Sultanlar Ligi",
    "Efeler Ligi",
    "Kadınlar 1. Ligi (voleybol)",
    "1. Lig (voleybol)",
    # kulüpler: taktik/kadro anlatımları buradan geliyor
    "Fenerbahçe (kadın voleybol takımı)",
    "Eczacıbaşı (kadın voleybol takımı)",
    "Galatasaray (kadın voleybol takımı)",
    "Beşiktaş (kadın voleybol takımı)",
    "Halkbank (erkek voleybol takımı)",
    "Arkas (erkek voleybol takımı)",
]

# TVF'nin oyun kuralları kitapçığı sitede yalnızca PDF olarak duruyor, HTML
# sayfası yok. Bu yüzden federasyonun haber akışını geziyorum: önce aşağıdaki
# indeks sayfalarından /icerik/... linklerini topluyorum, sonra tek tek haber
# sayfalarına giriyorum. Asıl Türkçe voleybol metni orada.
TVF_INDEX_PAGES = [
    "https://tvf.org.tr/",
    "https://tvf.org.tr/icerikler/haberler",
]
TVF_MAX_ARTICLES = 40  # nazik olmak için üst sınır (40 istek x 1 sn ≈ 40 sn)

# Wikipedia'da işimize yaramayan, sadece link listesi içeren bölümler.
SKIP_SECTIONS = {
    "kaynakça", "kaynaklar", "ayrıca bakınız", "dış bağlantılar",
    "notlar", "referanslar", "galeri", "kaynakca",
}

MIN_PARAGRAPH_CHARS = 80  # bundan kısa paragraflar cevap olamayacak kadar cılız


# ---- robots.txt ------------------------------------------------------------

_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _load_robots(scheme: str, host: str):
    """Bir host'un robots.txt'ini bir kez indir ve ayrıştır.

    RobotFileParser.read() kendi başına indirseydi urllib'in varsayılan
    User-Agent'ını kullanırdı; Wikimedia bunu 403 ile geri çeviriyor ve
    kütüphane 403'ü "her şey yasak" diye yorumlayıp tüm sayfaları eliyor.
    Bu yüzden robots.txt'i kendi User-Agent'ımla requests ile çekip
    parser'a parse() ile veriyorum.
    """
    key = f"{scheme}://{host}"
    if key in _robots_cache:
        return _robots_cache[key]

    parser = urllib.robotparser.RobotFileParser()
    try:
        resp = requests.get(
            f"{key}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=20
        )
        if resp.status_code >= 400:
            # robots.txt yoksa (404) kural yok demektir, çekmek serbest.
            parser.parse([])
        else:
            resp.encoding = resp.apparent_encoding or "utf-8"
            parser.parse(resp.text.splitlines())
    except requests.RequestException as exc:
        print(f"  ! robots.txt okunamadı ({key}): {exc}")
        parser = None

    _robots_cache[key] = parser
    time.sleep(REQUEST_DELAY)
    return parser


def check_robots(url: str) -> bool:
    """Bu URL'i USER_AGENT ile çekmeye izin var mı?"""
    parts = urlparse(url)
    parser = _load_robots(parts.scheme, parts.netloc)
    if parser is None:  # robots.txt'e ulaşamadıysam çekmiyorum
        return False
    return parser.can_fetch(USER_AGENT, url)


# ---- indirme ---------------------------------------------------------------

def fetch(url: str) -> str | None:
    """Sayfayı indir. Hata olursa None dön -- tek sayfa yüzünden durmuyoruz."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! indirilemedi: {exc}")
        return None
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


# ---- ayrıştırma ------------------------------------------------------------

def slugify(title: str) -> str:
    """'Kadınlar 1. Ligi (voleybol)' -> 'kadinlar-1-ligi-voleybol' (dosya adı)."""
    import re

    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")
    slug = title.lower().translate(tr)
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def clean(text: str) -> str:
    """Wikipedia'nın [1] [kaynak belirtilmeli] gibi işaretlerini temizle."""
    import re

    text = re.sub(r"\[\d+\]", "", text)               # dipnot numaraları
    text = re.sub(r"\[[^\]]{0,40}?\]", "", text)      # [kaynak belirtilmeli]
    text = re.sub(r"\s+", " ", text)                  # çoklu boşluk
    return text.strip()


def wikipedia_title(html: str) -> str:
    """Sayfanın gerçek başlığı.

    Wikipedia yönlendirmeleri sessizce takip ediyor: "Pasör" istediğimde bana
    "Voleybol" makalesini veriyor. Başlığı okuyup daha önce indirdiğim bir
    makaleye düştüysem atlıyorum, yoksa aynı metin veri setine iki kez girer.
    """
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.select_one("h1#firstHeading")
    return clean(h1.get_text(" ")) if h1 else ""


def parse_wikipedia(html: str) -> list[dict]:
    """Makaleyi {baslik, metin} bölümlerine ayır.

    Wikipedia'nın HTML'i düz: başlıklar ve paragraflar aynı seviyede sıralı
    duruyor. Ben de sırayla yürüyüp bir başlık görünce yeni bölüm açıyorum,
    paragraf görünce içinde bulunduğum bölüme ekliyorum.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.mw-parser-output")
    if body is None:
        return []

    # Tablo, kutu, dipnot gibi metin olmayan şeyleri baştan atıyorum.
    for junk in body.select("table, .reflist, .navbox, .infobox, style, sup.reference"):
        junk.decompose()

    sections: list[dict] = []
    current = {"baslik": "Giriş", "paragraflar": []}

    for el in body.find_all(["h2", "h3", "p"], recursive=True):
        if el.name in ("h2", "h3"):
            if current["paragraflar"]:
                sections.append(current)
            title = clean(el.get_text(" "))
            title = title.replace("[değiştir | kaynağı değiştir]", "").strip()
            current = {"baslik": title, "paragraflar": []}
        else:
            text = clean(el.get_text(" "))
            if len(text) >= MIN_PARAGRAPH_CHARS:
                current["paragraflar"].append(text)

    if current["paragraflar"]:
        sections.append(current)

    return [s for s in sections if s["baslik"].strip().lower() not in SKIP_SECTIONS]


def parse_generic(html: str) -> list[dict]:
    """TVF gibi Wikipedia olmayan sayfalar için basit paragraf toplayıcı."""
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select("script, style, nav, header, footer, form"):
        junk.decompose()

    # Sadece <p>: <li> alsaydım menü ve link listeleri de metin sanılırdı.
    paragraphs = []
    for p in soup.find_all("p"):
        text = clean(p.get_text(" "))
        if len(text) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(text)

    if not paragraphs:
        return []
    return [{"baslik": "Genel", "paragraflar": paragraphs}]


# ---- sürücü ----------------------------------------------------------------

def collect_tvf_articles() -> list[str]:
    """İndeks sayfalarını gezip /icerik/... haber linklerini topla.

    TVF'nin liste sayfalarında uzun paragraf yok, sadece link var; metin tek
    tek haber sayfalarının içinde. Bu yüzden iki aşamalı gidiyorum.
    """
    found: list[str] = []
    seen: set[str] = set()

    for index_url in TVF_INDEX_PAGES:
        if not check_robots(index_url):
            print(f"  ! robots.txt izin vermiyor: {index_url}")
            continue
        html = fetch(index_url)
        time.sleep(REQUEST_DELAY)
        if html is None:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not href.startswith("/icerik/"):
                continue
            full = "https://tvf.org.tr" + href
            if full not in seen:
                seen.add(full)
                found.append(full)

    return found


def scrape_one(name: str, url: str, parser, source: str,
               seen: set[str] | None = None) -> dict | None:
    print(f"- {name}\n  {url}")

    if not check_robots(url):
        print("  ! robots.txt izin vermiyor, atlıyorum")
        return None

    html = fetch(url)
    time.sleep(REQUEST_DELAY)
    if html is None:
        return None

    # Yönlendirme kontrolü: aynı makaleye ikinci kez düştüysem geç.
    if seen is not None:
        canonical = wikipedia_title(html)
        if canonical:
            if canonical in seen:
                print(f"  ! '{canonical}' zaten indirildi (yönlendirme), atlıyorum")
                return None
            seen.add(canonical)
            name = canonical

    sections = parser(html)
    if not sections:
        print("  ! kullanılabilir bölüm çıkmadı, atlıyorum")
        return None

    n_par = sum(len(s["paragraflar"]) for s in sections)
    n_chars = sum(len(p) for s in sections for p in s["paragraflar"])
    print(f"  {len(sections)} bölüm, {n_par} paragraf, {n_chars} karakter")

    return {
        "kaynak": source,
        "url": url,
        "baslik": name,
        "bolumler": sections,
    }


def main() -> None:
    os.makedirs(RAW_DIR, exist_ok=True)
    written = 0

    print("=== Türkçe Wikipedia ===")
    seen: set[str] = set()
    for page in WIKIPEDIA_PAGES:
        url = "https://tr.wikipedia.org/wiki/" + quote(page.replace(" ", "_"))
        doc = scrape_one(page, url, parse_wikipedia, "wikipedia", seen)
        if doc is None:
            continue
        path = os.path.join(RAW_DIR, f"wikipedia_{slugify(page)}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        written += 1

    print("\n=== Türkiye Voleybol Federasyonu ===")
    article_urls = collect_tvf_articles()
    print(f"{len(article_urls)} haber linki bulundu, ilk {TVF_MAX_ARTICLES} tanesi alınacak\n")
    for url in article_urls[:TVF_MAX_ARTICLES]:
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        doc = scrape_one(slug, url, parse_generic, "tvf")
        if doc is None:
            continue
        path = os.path.join(RAW_DIR, f"tvf_{slug[:80]}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        written += 1

    total_bytes = sum(
        os.path.getsize(os.path.join(RAW_DIR, f)) for f in os.listdir(RAW_DIR)
    )
    print(f"\n{written} dosya yazıldı -> {os.path.relpath(RAW_DIR, ROOT)} "
          f"({total_bytes / 1024:.1f} KB)")

    if written == 0:
        print("Hiçbir kaynak çekilemedi. İnternet bağlantısını kontrol et.")
        sys.exit(1)


if __name__ == "__main__":
    main()
