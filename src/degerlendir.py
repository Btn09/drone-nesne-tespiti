"""Farklı çıkarım ayarlarını COCO metrikleriyle karşılaştırır (küçük / orta / büyük nesne ayrımı dahil).

Çalıştırma:
    python -m src.degerlendir --bolum val     # ayarları seçmek için
    python -m src.degerlendir --bolum test    # son sonuçlar
"""
import argparse
import contextlib
import io
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from .dilimleme import dilimli_tahmin, tam_tahmin
from .veri import HEDEF, KOK, SINIFLAR

RAPOR = KOK / "reports"

# (ad, model dosyası, ayarlar)
DENEMELER = [
    ("640 model, tam görüntü 640", "yolo11s_640", dict(dilim=False, imgsz=640)),
    ("640 model, tam görüntü 1024", "yolo11s_640", dict(dilim=False, imgsz=1024)),
    ("1024 model, tam görüntü 1024", "yolo11s_1024", dict(dilim=False, imgsz=1024)),
    ("1024 model, tam görüntü 1536", "yolo11s_1024", dict(dilim=False, imgsz=1536)),
    ("640 model, parçalı 640", "yolo11s_640", dict(dilim=True, boyut=640, imgsz=640, tam_imgsz=640)),
    ("1024 model, parçalı 1024", "yolo11s_1024", dict(dilim=True, boyut=1024, imgsz=1024, tam_imgsz=1024)),
    ("1024 model, parçalı 640→1024", "yolo11s_1024", dict(dilim=True, boyut=640, imgsz=1024, tam_imgsz=1024)),
]


def tahmin_et(model, goruntu, ayar):
    if ayar["dilim"]:
        return dilimli_tahmin(model, goruntu, boyut=ayar["boyut"], imgsz=ayar["imgsz"],
                              tam_imgsz=ayar["tam_imgsz"])
    return tam_tahmin(model, goruntu, imgsz=ayar["imgsz"])


def coco_degerlendir(gt, tahminler):
    if not tahminler:
        return {}
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(tahminler)
        e = COCOeval(gt, dt, "bbox")
        # VisDrone'da görüntü başına yüzlerce nesne var; COCO'nun 100 sınırı yerine 500
        e.params.maxDets = [1, 100, 500]
        e.evaluate()
        e.accumulate()
        e.summarize()
    s = e.stats
    sonuc = {"AP": s[0], "AP50": s[1], "AP75": s[2], "AP_kucuk": s[3], "AP_orta": s[4], "AP_buyuk": s[5],
             "AR500": s[8]}
    # Sınıf bazında AP50
    kesinlik = e.eval["precision"][0, :, :, 0, 2]  # IoU=0.5, tüm alanlar, maxDets=500
    for i, ad in enumerate(SINIFLAR):
        p = kesinlik[:, i]
        sonuc[f"AP50_{ad}"] = float(p[p > -1].mean()) if (p > -1).any() else float("nan")
    return sonuc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bolum", default="test", choices=["val", "test"])
    ap.add_argument("--sinir", type=int, default=0, help="hızlı deneme için ilk N görüntü")
    ap.add_argument("--denemeler", nargs="*", help="sadece bu numaralı denemeler")
    args = ap.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(HEDEF / f"{args.bolum}_coco.json"))
    gorseller = gt.dataset["images"][: args.sinir or None]
    if args.sinir:
        gt.dataset["images"] = gorseller
        gt.dataset["annotations"] = [a for a in gt.dataset["annotations"] if a["image_id"] < args.sinir]
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()

    secili = [d for i, d in enumerate(DENEMELER) if not args.denemeler or str(i) in args.denemeler]
    modeller = {}
    satirlar = []
    for ad, model_adi, ayar in secili:
        if model_adi not in modeller:
            modeller[model_adi] = YOLO(KOK / "models" / f"{model_adi}.pt")
        model = modeller[model_adi]
        tam_tahmin(model, np.zeros((640, 640, 3), np.uint8))  # ısınma

        tahminler, sure = [], 0.0
        for g in gorseller:
            goruntu = cv2.imread(str(HEDEF / args.bolum / "images" / g["file_name"]))
            torch.cuda.synchronize()
            bas = time.perf_counter()
            kutular, skorlar, siniflar = tahmin_et(model, goruntu, ayar)
            torch.cuda.synchronize()
            sure += time.perf_counter() - bas
            for (x0, y0, x1, y1), s, c in zip(kutular, skorlar, siniflar):
                tahminler.append({"image_id": g["id"], "category_id": int(c),
                                  "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                                  "score": float(s)})
        satir = {"deneme": ad, **coco_degerlendir(gt, tahminler), "ms_goruntu": sure / len(gorseller) * 1000}
        satirlar.append(satir)
        print(f"{ad:32s} AP {satir['AP']:.3f} | AP50 {satir['AP50']:.3f} | küçük {satir['AP_kucuk']:.3f} | "
              f"orta {satir['AP_orta']:.3f} | büyük {satir['AP_buyuk']:.3f} | {satir['ms_goruntu']:.0f} ms", flush=True)

    RAPOR.mkdir(exist_ok=True)
    tablo = pd.DataFrame(satirlar)
    if not args.sinir:
        tablo.to_csv(RAPOR / f"sonuclar_{args.bolum}.csv", index=False, float_format="%.4f")
        with open(RAPOR / f"sonuclar_{args.bolum}.json", "w", encoding="utf-8") as f:
            json.dump(satirlar, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
