"""README grafikleri: eğitim eğrileri, nesne boyutuna göre AP, örnek tespitler.

Çalıştırma:  python -m src.grafikler   (önce src.egit ve src.degerlendir)
"""
import shutil

import cv2
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from .cikarim import Tespit
from .dilimleme import tam_tahmin
from .veri import KOK, SINIFLAR, SINIFLAR_TR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAPOR = KOK / "reports"
GRAFIK = RAPOR / "figures"


def egitim_egrileri():
    fig, eksen = plt.subplots(figsize=(8, 4))
    for ad in ["yolo11s_640", "yolo11s_1024"]:
        d = pd.read_csv(KOK / "runs" / ad / "results.csv")
        d.columns = d.columns.str.strip()
        eksen.plot(d["epoch"], d["metrics/mAP50(B)"], label=f"{ad} mAP50")
        eksen.plot(d["epoch"], d["metrics/mAP50-95(B)"], "--", label=f"{ad} mAP50-95")
        shutil.copy(KOK / "runs" / ad / "results.csv", RAPOR / f"egitim_{ad}.csv")
    eksen.set_xlabel("Epoch")
    eksen.set_ylabel("Doğrulama mAP")
    eksen.grid(alpha=0.3)
    eksen.legend(fontsize=8)
    eksen.set_title("Eğitim eğrileri (doğrulama seti)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "egitim_egrileri.png", dpi=140)
    plt.close(fig)


def boyut_grafigi():
    d = pd.read_csv(RAPOR / "sonuclar_test.csv")
    fig, eksen = plt.subplots(figsize=(10, 4.5))
    x = np.arange(3)
    genislik = 0.8 / len(d)
    for i, satir in d.iterrows():
        eksen.bar(x + (i - (len(d) - 1) / 2) * genislik, satir[["AP_kucuk", "AP_orta", "AP_buyuk"]].values,
                  genislik, label=f"{satir['deneme']} ({satir['ms_goruntu']:.0f} ms)")
    eksen.set_xticks(x, ["Küçük (<32²)", "Orta", "Büyük (>96²)"])
    eksen.set_ylabel("AP@[.5:.95]")
    eksen.grid(axis="y", alpha=0.3)
    eksen.legend(fontsize=7, ncol=2)
    eksen.set_title("Nesne boyutuna göre başarı (test seti)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "boyuta_gore_ap.png", dpi=140)
    plt.close(fig)


def sinif_grafigi():
    d = pd.read_csv(RAPOR / "sonuclar_test.csv").set_index("deneme")
    secim = [d.index[0], d["AP"].idxmax()]
    fig, eksen = plt.subplots(figsize=(10, 4))
    x = np.arange(len(SINIFLAR))
    for i, ad in enumerate(secim):
        eksen.bar(x + (i - 0.5) * 0.4, [d.loc[ad, f"AP50_{s}"] for s in SINIFLAR], 0.4, label=ad)
    eksen.set_xticks(x, SINIFLAR_TR, rotation=30, ha="right")
    eksen.set_ylabel("AP50")
    eksen.grid(axis="y", alpha=0.3)
    eksen.legend(fontsize=8)
    eksen.set_title("Sınıf bazında AP50 (test seti)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "sinif_ap50.png", dpi=140)
    plt.close(fig)


def ornek_tespit(dosya="drone_2.jpg", esik=0.3):
    goruntu = cv2.imread(str(KOK / "ornekler" / dosya))
    m640 = YOLO(KOK / "models" / "yolo11s_640.pt")
    m1024 = YOLO(KOK / "models" / "yolo11s_1024.pt")
    paneller = []
    for baslik, (k, s, c) in [
        ("640 model, tam görüntü", tam_tahmin(m640, goruntu, imgsz=640)),
        ("1024 model, tam görüntü 1536", tam_tahmin(m1024, goruntu, imgsz=1536)),
    ]:
        tut = s >= esik
        paneller.append((f"{baslik}: {tut.sum()} nesne",
                         cv2.cvtColor(Tespit.ciz(goruntu, k[tut], c[tut]), cv2.COLOR_BGR2RGB)))
    fig, eksenler = plt.subplots(1, 2, figsize=(16, 5))
    for eksen, (baslik, img) in zip(eksenler, paneller):
        eksen.imshow(img)
        eksen.set_title(baslik)
        eksen.axis("off")
    plt.tight_layout()
    plt.savefig(GRAFIK / "ornek_tespit.jpg", dpi=110, pil_kwargs={"quality": 85})
    plt.close(fig)


if __name__ == "__main__":
    GRAFIK.mkdir(parents=True, exist_ok=True)
    egitim_egrileri()
    boyut_grafigi()
    sinif_grafigi()
    ornek_tespit()
