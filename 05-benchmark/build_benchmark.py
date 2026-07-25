#!/usr/bin/env python3
"""Builds the VoleykoçAI domain benchmark (Hafta 2.2 assignment).

A hand-written multiple-choice Turkish volleyball benchmark. Every question is
authored fresh for this test set and is NOT part of the training data
(01-dataset/seeds.jsonl or the scraped corpus), so it is a genuine held-out
evaluation of the model's coaching domain.

Format mirrors alibayram/yapay_zeka_turkce_mmlu so the same letter-matching
scoring (olcum.py) applies:

    {"soru": ..., "secenekler": [...], "cevap": <dogru sik indeksi>, "konu": ...}

Run:
    python 05-benchmark/build_benchmark.py
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "benchmark")
OUT_PATH = os.path.join(OUT_DIR, "voleykoc_benchmark.jsonl")
STATS_PATH = os.path.join(ROOT, "reports", "benchmark_stats.md")

SEED = 1337

# Her satir: (soru, [siklar], dogru_sik_indeksi, konu)
# Siklar A,B,C,D... sirasiyla; cevap 0-tabanli indeks.
SORULAR = [
    # ---- kurallar ----
    ("Bir voleybol seti kaç sayıya oynanır ve kazanmak için en az kaç sayı fark gerekir?",
     ["21 sayı, 1 fark", "25 sayı, 2 fark", "25 sayı, 1 fark", "15 sayı, 2 fark"], 1, "kural"),
    ("Beşinci set (tie-break) kaç sayıya oynanır?",
     ["25", "21", "15", "11"], 2, "kural"),
    ("Bir takım topu karşı sahaya göndermeden önce en fazla kaç kez oynayabilir (blok hariç)?",
     ["2", "3", "4", "Sınırsız"], 1, "kural"),
    ("Sahada bir takımdan aynı anda kaç oyuncu bulunur?",
     ["5", "6", "7", "4"], 1, "kural"),
    ("Rotasyon hangi yönde yapılır?",
     ["Saat yönünde", "Saat yönünün tersine", "Rastgele", "Antrenörün işaretine göre"], 0, "kural"),
    ("Servis atılırken top file üstüne değip karşı sahaya geçerse ne olur?",
     ["Servis geçersizdir, sayı rakibe gider", "Servis tekrar edilir", "Oyun devam eder, geçerlidir", "Servis atan oyuncu değişir"], 2, "kural"),
    ("Arka bölge oyuncusu hücum vuruşunu nereden yaparsa kurallara uygundur?",
     ["Fileye istediği kadar yakınlaşarak", "3 metre çizgisinin gerisinden sıçrayarak", "Sadece file önünden", "Arka bölge oyuncusu hücum yapamaz"], 1, "kural"),
    ("Blok, takımın üç vuruş hakkından sayılır mı?",
     ["Evet, birinci vuruş sayılır", "Hayır, bloktan sayılmaz", "Sadece sayı olursa sayılır", "Evet, üçüncü vuruş sayılır"], 1, "kural"),
    ("Erkeklerde file yüksekliği kaç metredir?",
     ["2.24 m", "2.35 m", "2.43 m", "2.50 m"], 2, "kural"),
    ("Voleybol sahasının ölçüleri nedir?",
     ["16 x 8 m", "18 x 9 m", "20 x 10 m", "18 x 12 m"], 1, "kural"),

    # ---- libero ----
    ("Libero aşağıdakilerden hangisini yapamaz?",
     ["Manşet pas", "File üstünden hücum vuruşu", "Servis karşılama", "Kurtarış"], 1, "kural"),
    ("Libero hangi bölgede oynar?",
     ["Sadece ön bölge", "Sadece arka bölge", "Tüm saha", "Sadece 1 numara"], 1, "kural"),
    ("Liberonun forması diğer oyunculara göre nasıldır?",
     ["Aynı renktedir", "Farklı (zıt) renktedir", "Numarasızdır", "Kaptan bandı taşır"], 1, "kural"),

    # ---- teknik ----
    ("Manşet pasında top ideal olarak nereyle karşılanır?",
     ["Avuç içleriyle", "Parmak uçlarıyla", "Ön kolların iç düz yüzeyiyle", "Bileklerle"], 2, "teknik"),
    ("Parmak pasta topa kaç parmakla ve nasıl dokunulur?",
     ["İki avuçla kavrayarak", "On parmak uçlarıyla, alnın üstünde", "Yumrukla", "Tek elle"], 1, "teknik"),
    ("Sağ elini kullanan bir smaçörün klasik dört adımlı yaklaşım ritmi nasıldır?",
     ["sağ-sol-sağ-sol, hepsi eşit", "sol-sağ-sol-sağ, son iki adım hızlı", "tek adım sıçrama", "yavaş dört adım"], 1, "teknik"),
    ("Blokta 'penetrasyon' ne demektir?",
     ["Elleri fileye paralel tutmak", "Bilekleri kırıp elleri karşı sahaya uzatmak", "Fileye dokunmak", "Sıçramadan blok yapmak"], 1, "teknik"),
    ("Manşet pasta itiş gücü ağırlıklı olarak nereden gelmelidir?",
     ["Kollardan", "Bileklerden", "Bacaklardan", "Omuzlardan"], 2, "teknik"),
    ("Float (titreşimli) servis ile jump servisin temel farkı nedir?",
     ["Float sıçrayarak atılır", "Float dönüşsüz ve sabit temaslı, jump sıçramalı ve güçlüdür", "İkisi aynıdır", "Jump servis dönüşsüzdür"], 1, "teknik"),
    ("Smaçta topa vuruş anında kol nasıl olmalıdır?",
     ["Dirsek bükülü", "Tam uzatılmış ve yüksekte", "Vücuda yakın", "Aşağıda"], 1, "teknik"),

    # ---- taktik / pozisyon ----
    ("5-1 rotasyon sisteminde takımda kaç pasör vardır?",
     ["1", "2", "3", "Pasör yoktur"], 0, "taktik"),
    ("4-2 sisteminin 5-1'e göre avantajı nedir?",
     ["Daha çok hücumcu", "Her rotasyonda ön bölgede pasör bulunması, sistemin basitliği", "Daha hızlı hücum", "Liberosuz oynanması"], 1, "taktik"),
    ("Pasör dış oyuncuya pası fileden ne kadar açık vermelidir?",
     ["Fileye yapışık", "Yaklaşık 30-50 cm açık", "2 metre açık", "Saha ortasına"], 1, "taktik"),
    ("Pasör çaprazı (opposite) hangi oyuncudur?",
     ["Liberonun yerine giren", "Pasörün karşısında dizilen, sağ kanattan hücum eden", "Orta oyuncu", "İkinci libero"], 1, "taktik"),
    ("Modern servis karşılamada genellikle kaç oyuncu görev alır?",
     ["6", "5", "3", "2"], 2, "taktik"),
    ("Orta oyuncunun (ortacı) temel görevleri nelerdir?",
     ["Sadece servis atmak", "Hızlı orta hücum ve blokun merkezinde olmak", "Sadece savunma", "Pas dağıtmak"], 1, "taktik"),
    ("Pasör arka bölgedeyken 5-1 sisteminde takımın hücum gücü nasıldır?",
     ["En düşük", "Değişmez", "En yüksek (üç hücumcu koşabilir)", "Sadece iki hücumcu vardır"], 2, "taktik"),

    # ---- kondisyon / sakatlık ----
    ("Sıçrama yüksekliğini artırmak için hangi antrenman türü en uygundur?",
     ["Sadece uzun mesafe koşusu", "Pliometrik ve kuvvet çalışması", "Sadece esneme", "Yüzme"], 1, "kondisyon"),
    ("Voleybolda en sık görülen sakatlıklardan biri hangisidir?",
     ["Ayak bileği burkulması", "Kaburga kırığı", "Beyin sarsıntısı", "Diz çıkığı"], 0, "sakatlik"),
    ("Sıçrayıcı dizi (patellar tendinopati) riskini azaltmak için ne önemlidir?",
     ["Antrenman hacmini ani artırmak", "Doğru iniş tekniği ve eksantrik kuvvet çalışması", "Sadece dinlenmek", "Daha çok sıçramak"], 1, "sakatlik"),
    ("Maç öncesi ısınmada hangisi tercih edilmelidir?",
     ["Uzun statik esneme", "Dinamik esneme ve hareketlilik", "Ağır kuvvet çalışması", "Hiç ısınmamak"], 1, "kondisyon"),
    ("Antrenman yükünü haftada en fazla ne kadar artırmak sakatlık riskini düşürür?",
     ["Yaklaşık %10", "%50", "%100", "Sınır yoktur"], 0, "sakatlik"),

    # ---- antrenman ----
    ("14 yaş altı grupta antrenmanın asıl amacı ne olmalıdır?",
     ["Maç kazanmak", "Beceri gelişimi ve oyun sevgisi", "Ağır kuvvet antrenmanı", "Erken uzmanlaşma"], 1, "antrenman"),
    ("Yeni başlayan bir grupta hangi teknikler önce öğretilmelidir?",
     ["Jump servis ve blok", "Manşet ve parmak pas", "Slide hücumu", "Çift blok"], 1, "antrenman"),
    ("Bir hücumcuya antrenman başına tam güç smaç sayısında sınır koymanın sebebi nedir?",
     ["Topları korumak", "Omuz ve sırt sakatlıklarını önlemek", "Süreyi kısaltmak", "Kural gereği"], 1, "antrenman"),
    ("Genç yaş grubunda erken uzmanlaşma (tek pozisyonda oynatma) neden sakıncalıdır?",
     ["Sıkıcı olduğu için", "Dengeli gelişimi engellediği ve sakatlık riskini artırdığı için", "Kural yasakladığı için", "Sakıncası yoktur"], 1, "antrenman"),
    ("Servis karşılamada oyuncular nasıl konumlanmalıdır?",
     ["Sabit noktalarda durarak", "Topun düşeceği tahmini noktaya göre hareket ederek", "Hepsi filede", "Rastgele"], 1, "antrenman"),

    # ---- Türk voleybolu / genel ----
    ("Türkiye Kadın Millî Voleybol Takımı'nın lakabı nedir?",
     ["Filenin Efeleri", "Filenin Sultanları", "Ay-Yıldızlılar", "Boğalar"], 1, "genel"),
    ("Türkiye Erkek Millî Voleybol Takımı'nın lakabı nedir?",
     ["Filenin Sultanları", "Filenin Efeleri", "Millî Filenin Aslanları", "Anadolu Kartalları"], 1, "genel"),
    ("Kadınlar en üst düzey voleybol ligi Türkiye'de hangi adla bilinir?",
     ["Efeler Ligi", "Sultanlar Ligi", "Süper Lig", "1. Lig"], 1, "genel"),
]


def make_rows() -> list[dict]:
    rows = []
    for soru, siklar, cevap, konu in SORULAR:
        assert 0 <= cevap < len(siklar), f"geçersiz cevap indeksi: {soru}"
        rows.append({
            "soru": soru,
            "secenekler": siklar,
            "cevap": cevap,
            "konu": konu,
        })
    return rows


def validate(rows: list[dict]) -> list[str]:
    problems = []
    sorular = set()
    for i, r in enumerate(rows):
        if set(r) != {"soru", "secenekler", "cevap", "konu"}:
            problems.append(f"satır {i}: alan kümesi yanlış")
        if len(r["secenekler"]) < 2:
            problems.append(f"satır {i}: yeterli şık yok")
        if not (0 <= r["cevap"] < len(r["secenekler"])):
            problems.append(f"satır {i}: cevap indeksi şık sayısını aşıyor")
        if r["soru"] in sorular:
            problems.append(f"satır {i}: yinelenen soru")
        sorular.add(r["soru"])
    return problems


def write_stats(rows: list[dict]) -> None:
    konular = Counter(r["konu"] for r in rows)
    L = ["# VoleykoçAI alan benchmark istatistikleri", ""]
    L.append(f"Toplam soru: **{len(rows)}** (çoktan seçmeli, tek doğru)")
    L.append("")
    L.append("## Konu dağılımı")
    L.append("")
    L.append("| Konu | Soru |")
    L.append("|---|---:|")
    for k, n in konular.most_common():
        L.append(f"| {k} | {n} |")
    L.append("")
    L.append("Tüm sorular bu test seti için elle yazıldı ve eğitim verisinde "
             "(`01-dataset/seeds.jsonl` ve scrape edilen korpus) yer almıyor; "
             "yani gerçek anlamda held-out bir değerlendirmedir.")
    L.append("")
    os.makedirs(os.path.dirname(STATS_PATH), exist_ok=True)
    with open(STATS_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main() -> None:
    rows = make_rows()
    problems = validate(rows)
    if problems:
        print(f"{len(problems)} hata:")
        for p in problems:
            print(f"  ! {p}")
        raise SystemExit(1)
    print("şema doğrulaması: tamam")

    random.Random(SEED).shuffle(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_stats(rows)
    print(f"{len(rows)} soru yazıldı -> {os.path.relpath(OUT_PATH, ROOT)}")
    print(f"Rapor -> {os.path.relpath(STATS_PATH, ROOT)}")


if __name__ == "__main__":
    main()
