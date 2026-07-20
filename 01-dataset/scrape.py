#!/usr/bin/env python3
"""Scrapes raw Turkish volleyball text from Wikipedia and the TVF website.

Writes one JSON file per page into data/raw/. build_dataset.py turns those
into question/answer pairs.

Run:
    python 01-dataset/scrape.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.robotparser
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

USER_AGENT = (
    "VoleykocAI-Odev/1.0 (ders odevi veri toplama; "
    "https://github.com/berkcangumusisik/voleykocai-llm-finetuning)"
)
REQUEST_DELAY = 1.0

# Every title verified against the MediaWiki API. Smac, Manset and Servis are
# not separate pages on Turkish Wikipedia; those terms live inside Voleybol.
WIKIPEDIA_PAGES = [
    "Voleybol",
    "Plaj voleybolu",
    "Oturarak voleybol",
    "Uluslararası Voleybol Federasyonu",
    "Avrupa Voleybol Konfederasyonu",
    "Türkiye Voleybol Federasyonu",
    "Türkiye kadın millî voleybol takımı",
    "Türkiye erkek millî voleybol takımı",
    "Sultanlar Ligi",
    "Efeler Ligi",
    "Kadınlar 1. Ligi (voleybol)",
    "1. Lig (voleybol)",
    "Fenerbahçe (kadın voleybol takımı)",
    "Eczacıbaşı (kadın voleybol takımı)",
    "Galatasaray (kadın voleybol takımı)",
    "Beşiktaş (kadın voleybol takımı)",
    "Halkbank (erkek voleybol takımı)",
    "Arkas (erkek voleybol takımı)",
]

# TVF publishes its rulebook as PDF only, so the news feed is used instead.
TVF_INDEX_PAGES = [
    "https://tvf.org.tr/",
    "https://tvf.org.tr/icerikler/haberler",
]
TVF_MAX_ARTICLES = 40

SKIP_SECTIONS = {
    "kaynakça", "kaynaklar", "ayrıca bakınız", "dış bağlantılar",
    "notlar", "referanslar", "galeri", "kaynakca",
}

MIN_PARAGRAPH_CHARS = 80

_robots_cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _load_robots(scheme: str, host: str):
    """Fetch and parse a host's robots.txt once.

    RobotFileParser.read() would download with urllib's default User-Agent,
    which Wikimedia rejects with 403; the library reads that as "disallow
    everything". So the file is fetched with requests instead.
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
    parts = urlparse(url)
    parser = _load_robots(parts.scheme, parts.netloc)
    if parser is None:
        return False
    return parser.can_fetch(USER_AGENT, url)


def fetch(url: str) -> str | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  ! indirilemedi: {exc}")
        return None
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def slugify(title: str) -> str:
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosucgiosu")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower().translate(tr))
    return slug.strip("-")


def clean(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[[^\]]{0,40}?\]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def wikipedia_title(html: str) -> str:
    """Canonical page title, used to detect redirects to an already-seen page."""
    soup = BeautifulSoup(html, "html.parser")
    h1 = soup.select_one("h1#firstHeading")
    return clean(h1.get_text(" ")) if h1 else ""


def parse_wikipedia(html: str) -> list[dict]:
    """Split an article into {baslik, paragraflar} sections.

    Headings and paragraphs sit at the same level in document order, so this
    walks that order and appends each paragraph to the current section.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.mw-parser-output")
    if body is None:
        return []

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
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select("script, style, nav, header, footer, form"):
        junk.decompose()

    # <p> only; <li> would pull in menus and link lists.
    paragraphs = [
        clean(p.get_text(" ")) for p in soup.find_all("p")
        if len(clean(p.get_text(" "))) >= MIN_PARAGRAPH_CHARS
    ]
    if not paragraphs:
        return []
    return [{"baslik": "Genel", "paragraflar": paragraphs}]


def collect_tvf_articles() -> list[str]:
    """Two-stage crawl: index pages hold links, articles hold the text."""
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
            if not a["href"].startswith("/icerik/"):
                continue
            full = "https://tvf.org.tr" + a["href"]
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

    return {"kaynak": source, "url": url, "baslik": name, "bolumler": sections}


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
    print(f"{len(article_urls)} haber linki bulundu, "
          f"ilk {TVF_MAX_ARTICLES} tanesi alınacak\n")
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
