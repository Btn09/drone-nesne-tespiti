"""YOLO11 modelini VisDrone üzerinde ince ayarla eğitir.

Örnek:
    python -m src.egit --imgsz 640 --epochs 80 --ad yolo11s_640
    python -m src.egit --imgsz 1024 --epochs 80 --ad yolo11s_1024
"""
import argparse
import shutil

from ultralytics import YOLO

from .veri import HEDEF, KOK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt", help="COCO'da önceden eğitilmiş başlangıç ağırlıkları")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ad", required=True)
    ap.add_argument("--fraction", type=float, default=1.0, help="hız denemesi için verinin bir kısmı")
    ap.add_argument("--devam", action="store_true", help="yarıda kalan eğitime son kayıttan devam et")
    args = ap.parse_args()

    if args.devam:
        YOLO(KOK / "runs" / args.ad / "weights" / "last.pt").train(resume=True)
    else:
        YOLO(args.model).train(
            data=str(HEDEF / "visdrone.yaml"),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            fraction=args.fraction,
            project=str(KOK / "runs"),
            name=args.ad,
            exist_ok=True,
            seed=0,
            workers=8,
            close_mosaic=10,
            plots=True,
        )
    hedef = KOK / "models"
    hedef.mkdir(exist_ok=True)
    shutil.copy(KOK / "runs" / args.ad / "weights" / "best.pt", hedef / f"{args.ad}.pt")


if __name__ == "__main__":
    main()
