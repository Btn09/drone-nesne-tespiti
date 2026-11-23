"""Modelleri ONNX'e aktarır ve farklı çalıştırma biçimlerinde hızı ölçer.

Çalıştırma:  python -m src.hiz
Ölçülen süre: ön işleme + model + son işleme, tek görüntü (batch 1), 100 test görüntüsü ortalaması.
"""
import json
import time

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

from .veri import HEDEF, KOK

RAPOR = KOK / "reports"


def olc(model, goruntuler, imgsz, half=False, device=0, isinma=10):
    for g in goruntuler[:isinma]:
        model.predict(g, imgsz=imgsz, half=half, device=device, verbose=False)
    sureler = []
    for g in goruntuler:
        if device != "cpu":
            torch.cuda.synchronize()
        bas = time.perf_counter()
        model.predict(g, imgsz=imgsz, half=half, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        sureler.append(time.perf_counter() - bas)
    ms = sum(sureler) / len(sureler) * 1000
    return round(ms, 1), round(1000 / ms, 1)


def main():
    yollar = sorted((HEDEF / "test" / "images").glob("*.jpg"))[:100]
    goruntuler = [cv2.imread(str(p)) for p in yollar]
    satirlar = []
    for ad, imgsz in [("yolo11s_640", 640), ("yolo11s_1024", 1024)]:
        pt = KOK / "models" / f"{ad}.pt"
        model = YOLO(pt)
        onnx_yolu = model.export(format="onnx", imgsz=imgsz, simplify=True, verbose=False)
        onnx = YOLO(onnx_yolu, task="detect")

        # ONNX sadece işlemcide ölçülüyor: ekran kartı olmayan bir cihazda (ör. saha bilgisayarı) nasıl çalışır?
        for bicim, m, half, cihaz, adet in [
            ("PyTorch FP32 (GPU)", model, False, 0, 100),
            ("PyTorch FP16 (GPU)", model, True, 0, 100),
            ("ONNX Runtime (CPU)", onnx, False, "cpu", 30),
        ]:
            ms, fps = olc(m, goruntuler[:adet], imgsz, half=half, device=cihaz)
            satirlar.append({"model": ad, "imgsz": imgsz, "bicim": bicim, "ms": ms, "fps": fps})
            print(f"{ad:14s} {bicim:20s} {ms:7.1f} ms  {fps:6.1f} FPS", flush=True)

    RAPOR.mkdir(exist_ok=True)
    pd.DataFrame(satirlar).to_csv(RAPOR / "hiz.csv", index=False)
    with open(RAPOR / "hiz.json", "w", encoding="utf-8") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "olcumler": satirlar}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
