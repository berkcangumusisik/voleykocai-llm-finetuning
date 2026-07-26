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

    # ---- ek kurallar ----
    ("Bir sette her takımın kaç mola hakkı vardır (FIVB)?",
     ["1", "2", "3", "Sınırsız"], 1, "kural"),
    ("Uluslararası kurallarda bir sette takım başına oyuncu değişikliği hakkı kaçtır?",
     ["3", "6", "12", "Sınırsız"], 1, "kural"),
    ("Servis, hakem düdüğünden sonra kaç saniye içinde atılmalıdır?",
     ["3", "5", "8", "10"], 2, "kural"),
    ("Arka bölge oyuncusu blok yapabilir mi?",
     ["Evet, serbesttir", "Hayır, arka bölge oyuncusu blok yapamaz", "Sadece libero yapabilir", "Sadece 6 numara yapabilir"], 1, "kural"),
    ("Top oyuncunun elinde bir an durursa (tutma/taşıma) ne olur?",
     ["Geçerlidir", "Faul, sayı rakibe geçer", "Servis tekrarlanır", "Uyarı verilir"], 1, "kural"),
    ("Topun geçerli sayılması için file üzerinde nereden geçmesi gerekir?",
     ["Antenlerin arasından", "Antenlerin dışından", "Direklerin üstünden", "Fark etmez"], 0, "kural"),
    ("Rotasyon (dizilim) hatası ne zaman değerlendirilir?",
     ["Servis vuruşu anındaki dizilime göre", "Ralinin ortasında", "Sadece sayı sonrası", "Antrenör molasında"], 0, "kural"),
    ("Oyun sırasında topu oynama hareketi yaparken fileye dokunmak faul müdür?",
     ["Hayır, serbesttir", "Evet, o eylemde fileye temas fauldür", "Sadece blokta faul", "Sadece serviste faul"], 1, "kural"),
    ("Saha üzerindeki 3 metre çizgisinin diğer adı nedir?",
     ["Servis çizgisi", "Hücum çizgisi", "Orta çizgi", "Dip çizgi"], 1, "kural"),
    ("Bir oyuncu topa üst üste iki kez dokunabilir mi (blok hariç, normal oyunda)?",
     ["Evet, serbest", "Hayır, ardışık iki temas fauldür", "Sadece serviste", "Sadece savunmada"], 1, "kural"),

    # ---- ek teknik ----
    ("Parmak pasta eller nerede hazırlanmalıdır?",
     ["Göğüs hizasında", "Alnın 15-20 cm üstünde", "Belde", "Başın arkasında"], 1, "teknik"),
    ("Manşet pasında dirsekler nasıl olmalıdır?",
     ["Bükülü", "Kilitli ve düz", "Serbest", "Yana açık"], 1, "teknik"),
    ("Kurtarışta (savunma) beklerken ağırlık nerede olmalıdır?",
     ["Topuklarda", "Ayak parmak uçlarında", "Tek ayakta", "Geride"], 1, "teknik"),
    ("Smaç yaklaşımında kollar sıçrama anında ne yapar?",
     ["Yanda durur", "Geriye savrulup yukarı fırlatılır", "Belde tutulur", "Öne uzatılır"], 1, "teknik"),
    ("Blokta dış eldeki parmaklar nereye çevrilmelidir?",
     ["Dışarı", "Sahanın içine", "Yukarı", "Fileye"], 1, "teknik"),
    ("Pankek (pancake) tekniği ne için kullanılır?",
     ["Servis atmak", "Yere düşen topu elin sırtıyla kurtarmak", "Blok yapmak", "Pas vermek"], 1, "teknik"),
    ("İyi bir servis karşılamada kollar topa ne zaman birleştirilmelidir?",
     ["En baştan birleşik tutulur", "Top gelirken birleştirilir", "Vuruştan sonra", "Hiç birleştirilmez"], 1, "teknik"),
    ("Jump float servisin amacı nedir?",
     ["Maksimum güç", "Öngörülemez, titreşimli ve dönüşsüz bir yörünge", "Yavaş top", "Yüksek top"], 1, "teknik"),

    # ---- ek taktik / pozisyon ----
    ("6-2 sisteminin temel özelliği nedir?",
     ["Tek pasör", "İki pasör arka bölgeden pas verir, hep üç ön hücumcu bulunur", "Pasörsüz oyun", "İki libero"], 1, "taktik"),
    ("Voleybolda 4 numara bölgesi sahanın neresidir?",
     ["Ön sağ", "Ön sol", "Arka orta", "Arka sağ"], 1, "pozisyon"),
    ("1 numara bölgesi neresidir (servis bölgesi)?",
     ["Arka sağ", "Ön sol", "Ön orta", "Arka sol"], 0, "pozisyon"),
    ("Birinci tempo (hızlı orta) hücum kime yapılır?",
     ["Dış oyuncuya", "Orta oyuncuya", "Liberoya", "Pasöre"], 1, "taktik"),
    ("Servis attıktan sonra oyuncuların pozisyonlarına geçmesine ne denir?",
     ["Rotasyon", "Switch (yer değiştirme)", "Blok", "Kapatma"], 1, "taktik"),
    ("Hücum kapatma (coverage) ne demektir?",
     ["Bloğu kapatmak", "Smaçörün bloktan sekecek topu için arkadaşlarının toplanması", "Servisi engellemek", "Fileyi kapatmak"], 1, "taktik"),
    ("W dizilimi kaç oyuncuyla yapılan bir servis karşılama düzenidir?",
     ["3", "4", "5", "6"], 2, "taktik"),
    ("Pasör sahada yoksa (out-of-system) genelde kim ikinci topu kaldırır?",
     ["Kaleci", "Libero ya da en yakın oyuncu", "Orta oyuncu zorunlu", "Hakem"], 1, "taktik"),

    # ---- ek kondisyon / sakatlık ----
    ("Ayak bileği burkulması voleybolda en çok nerede olur?",
     ["Servis atarken", "File altında rakip/arkadaş ayağına basınca", "Otururken", "Isınmada"], 1, "sakatlik"),
    ("Omuz sağlığı için hücumcularda hangi çalışma önemlidir?",
     ["Sadece smaç", "Rotator manşet ve skapula stabilizasyonu", "Sadece koşu", "Hiçbiri"], 1, "sakatlik"),
    ("Pliometrik çalışmada iniş tekniği neden önemlidir?",
     ["Daha yüksek sıçramak için", "Diz ve bilek sakatlığını önlemek için", "Hız için", "Estetik için"], 1, "kondisyon"),
    ("Maç günü ana öğün ne zaman yenmelidir?",
     ["Maçtan hemen önce", "Maçtan 3-4 saat önce", "Maç sırasında", "Yenmemeli"], 1, "kondisyon"),
    ("Antrenman periyotlamasının amacı nedir?",
     ["Sürekli maksimum yüklenmek", "Yükü planlı dağıtıp form ve toparlanmayı dengelemek", "Hiç dinlenmemek", "Sadece maç oynamak"], 1, "kondisyon"),

    # ---- ek antrenman ----
    ("Küçük alan oyunları (2'ye 2, 3'e 3) altyapıda neden faydalıdır?",
     ["Sadece eğlence", "Her oyuncunun top temasını ve karar sayısını artırır", "Kural gereği", "Faydası yok"], 1, "antrenman"),
    ("Antrenmanda geri bildirim nasıl verilmelidir?",
     ["Sadece hataları sayarak", "Somut ve uygulanabilir, tek bir odakla", "Hiç konuşmadan", "Bağırarak"], 1, "antrenman"),
    ("Soğuma (cool-down) ne işe yarar?",
     ["Isınmayı bozar", "Toparlanmayı destekler ve kalp hızını kademeli düşürür", "Performansı düşürür", "Gereksizdir"], 1, "antrenman"),
    ("Yeni bir beslenme/ekipmanı ilk kez ne zaman denememelisin?",
     ["Antrenmanda", "Önemli maç gününde", "Fark etmez", "Tatilde"], 1, "antrenman"),
    ("Isınma maç başlangıcından en fazla ne kadar önce bitmelidir?",
     ["1 saat", "30 dakika", "10 dakika", "Fark etmez"], 2, "antrenman"),

    # ---- ek genel / Türk voleybolu ----
    ("Uluslararası Voleybol Federasyonu (FIVB) hangi yıl kuruldu?",
     ["1928", "1947", "1964", "1980"], 1, "genel"),
    ("Voleybol hangi olimpiyatta ilk kez olimpik spor oldu?",
     ["1936 Berlin", "1964 Tokyo", "1992 Barselona", "2000 Sidney"], 1, "genel"),
    ("Bir voleybol topunun çevresi yaklaşık kaç cm'dir?",
     ["58-60 cm", "65-67 cm", "70-72 cm", "75-78 cm"], 1, "genel"),
    ("Erkek millî takımına 2020'lerde gelen büyük başarı hangisidir?",
     ["Dünya Kupası şampiyonluğu", "Milletler Ligi'nde (VNL) ilk kez üst sıralara çıkması", "Olimpiyat altını", "Hiçbiri"], 1, "genel"),
    ("Kadın millî takımı 2023'te hangi büyük turnuvada Avrupa şampiyonu oldu?",
     ["Dünya Şampiyonası", "Avrupa Şampiyonası (CEV)", "Olimpiyat", "Milletler Ligi"], 1, "genel"),
    ("VakıfBank ve Eczacıbaşı hangi ligin köklü kulüpleridir?",
     ["Efeler Ligi", "Sultanlar Ligi", "1. Lig", "Bölgesel Lig"], 1, "genel"),
    ("Avrupa kıtasının voleybol yönetim organı hangisidir?",
     ["FIVB", "CEV", "UEFA", "AVC"], 1, "genel"),
    ("Plaj voleybolunda bir takımda kaç oyuncu bulunur?",
     ["2", "3", "4", "6"], 0, "genel"),
    ("Plaj voleybolunda setler kaç sayıya oynanır (ilk iki set)?",
     ["25", "21", "15", "11"], 1, "genel"),
    ("Oturarak voleybolda hücum/servis anında hangi kural vardır?",
     ["Ayakta olunur", "Kalçanın yerle teması korunur", "Fark etmez", "Sadece diz yerde"], 1, "genel"),

    # ---- ek teknik/kural karışık ----
    ("Servisi karşılarken libero topu ön bölgeden parmak pasla çıkarırsa ne olur?",
     ["Her zaman serbest", "O topa file üstü seviyesinde hücum edilemez", "Sayı olur", "Servis tekrarlanır"], 1, "kural"),
    ("İyi bir smaç için pasın fileye uzaklığı ne olmalı?",
     ["Fileye yapışık", "30-50 cm açık", "2 metre açık", "Dip çizgide"], 1, "teknik"),
    ("Servis karşılamada float servise karşı oyuncu nasıl durmalı?",
     ["Dip çizgide", "Biraz öne, hazır konumda", "Filede", "Oturarak"], 1, "taktik"),
    ("Blokta zamanlama nasıl olmalı?",
     ["Smaçörden önce sıçramak", "Smaçörün vuruşuna göre, hafif sonra sıçramak", "Hiç sıçramamak", "Top yere değince"], 1, "teknik"),
    ("Manşet pasta topun dönmesi neyi gösterir?",
     ["İyi teknik", "İki kolun teması senkron değil", "Güçlü vuruş", "Doğru açı"], 1, "teknik"),
    ("Konsantrasyonu set ortasında düşen takıma antrenör molada ne vermeli?",
     ["Uzun taktik anlatımı", "Tek somut görev", "Sessizlik", "Azarlama"], 1, "antrenman"),
    ("Sıçrama yüksekliğini en çok hangi ikili birlikte belirler?",
     ["Boy ve kilo", "Kuvvet ve patlayıcılık", "Yaş ve cinsiyet", "Ayakkabı ve saha"], 1, "kondisyon"),
    ("Voleybolda 'ace' nedir?",
     ["Blok sayısı", "Doğrudan sayı getiren servis", "Smaç hatası", "Pas çeşidi"], 1, "genel"),
    ("Rakip analiz ederken en öncelikli bakılacak şey nedir?",
     ["Formaların rengi", "Hücum dağılımı ve zayıf karşılama bölgesi", "Seyirci sayısı", "Salon"], 1, "taktik"),
    ("Manşet pasta itiş ağırlıklı nereden gelir?",
     ["Kollardan", "Bacaklardan", "Bileklerden", "Boyundan"], 1, "teknik"),
    ("Orta oyuncu blokta hangi rolü üstlenir?",
     ["Sadece seyretmek", "Blokun merkezinde olup yanlara kaymak", "Servis atmak", "Pas vermek"], 1, "taktik"),
    ("Genç sporcuda ağır kuvvet antrenmanı yerine ne önerilir?",
     ["Hiç antrenman", "Kendi vücut ağırlığıyla çalışma", "Maksimum ağırlık", "Sadece koşu"], 1, "antrenman"),
    ("Bir takım servis hakkını kaybettiğinde ne olur (rally point)?",
     ["Hiçbir şey", "Rakip hem sayıyı hem servisi alır", "Sadece servis geçer", "Set biter"], 1, "kural"),
    ("Dört adımlı smaç yaklaşımında son iki adım nasıl olmalı?",
     ["Yavaş ve uzun", "Hızlı ve birleşik", "Durarak", "Geriye"], 1, "teknik"),
    ("Libero servis atabilir mi (FIVB genel kural)?",
     ["Her zaman", "Hayır (bazı ulusal liglerde tek rotasyonda izin var)", "Sadece son sette", "Sadece tie-break'te"], 1, "kural"),
    ("İyi bir antrenman planı yaş grubuna göre öncelikle neyi gözetmeli?",
     ["Sadece kazanmayı", "Gelişim aşaması ve güvenliği", "Seyirciyi", "Ekonomiyi"], 1, "antrenman"),
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
