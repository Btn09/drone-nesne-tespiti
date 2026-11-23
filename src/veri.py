"""VisDrone2019-DET etiketlerini YOLO ve COCO formatına çevirir.

Çalıştırma:  python -m src.veri   (zip dosyaları data/indirme/ altında olmalı)

VisDrone satır formatı: x, y, genişlik, yükseklik, skor, sınıf, kesilme, örtülme
Sınıf 0 "yok sayılan bölge", 11 "diğer"; ikisini de atıyorum, kalan 10 sınıf 0-9 oluyor.
Skor 0 olan kutular da yok sayılan kutular.
"""
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image

KOK = Path(__file__).resolve().parent.parent
VERI = KOK / "data"
HEDEF = VERI / "visdrone"

SINIFLAR = ["pedestrian", "people", "bicycle", "car", "van",
            "truck", "tricycle", "awning-tricycle", "bus", "motor"]
SINIFLAR_TR = ["yaya", "insan", "bisiklet", "araba", "minibüs",
               "kamyon", "üç teker", "tenteli üç teker", "otobüs", "motosiklet"]

BOLUMLER = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val",
            "test": "VisDrone2019-DET-test-dev"}


def satirlari_oku(yol):
    kutular = []
    for satir in yol.read_text().strip().splitlines():
        d = [int(v) for v in satir.strip().rstrip(",").split(",")[:8]]
        x, y, w, h, skor, sinif = d[:6]
        if skor == 0 or sinif in (0, 11) or w <= 0 or h <= 0:
            continue
        kutular.append((x, y, w, h, sinif - 1))
    return kutular


def bolum_cevir(bolum, klasor_adi):
    zip_yolu = VERI / "indirme" / f"{klasor_adi}.zip"
    gecici = VERI / "acilmis"
    if not (gecici / klasor_adi).exists():
        with zipfile.ZipFile(zip_yolu) as z:
            z.extractall(gecici)
    kaynak = gecici / klasor_adi

    hedef_resim = HEDEF / bolum / "images"
    hedef_etiket = HEDEF / bolum / "labels"
    hedef_resim.mkdir(parents=True, exist_ok=True)
    hedef_etiket.mkdir(parents=True, exist_ok=True)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": ad} for i, ad in enumerate(SINIFLAR)]}
    for no, resim in enumerate(sorted((kaynak / "images").glob("*.jpg"))):
        g, y = Image.open(resim).size
        kutular = satirlari_oku(kaynak / "annotations" / f"{resim.stem}.txt")

        shutil.move(str(resim), hedef_resim / resim.name)
        with open(hedef_etiket / f"{resim.stem}.txt", "w") as f:
            for x, yy, w, h, s in kutular:
                f.write(f"{s} {(x + w / 2) / g:.6f} {(yy + h / 2) / y:.6f} {w / g:.6f} {h / y:.6f}\n")

        coco["images"].append({"id": no, "file_name": resim.name, "width": g, "height": y})
        for x, yy, w, h, s in kutular:
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": no, "category_id": s,
                                        "bbox": [x, yy, w, h], "area": w * h, "iscrowd": 0})
    with open(HEDEF / f"{bolum}_coco.json", "w") as f:
        json.dump(coco, f)
    print(f"{bolum}: {len(coco['images'])} görüntü, {len(coco['annotations'])} nesne")


def yaml_yaz():
    satirlar = [f"path: {HEDEF.as_posix()}", "train: train/images", "val: val/images", "test: test/images",
                "names:"] + [f"  {i}: {ad}" for i, ad in enumerate(SINIFLAR)]
    (HEDEF / "visdrone.yaml").write_text("\n".join(satirlar) + "\n")


if __name__ == "__main__":
    for bolum, ad in BOLUMLER.items():
        bolum_cevir(bolum, ad)
    yaml_yaz()
    shutil.rmtree(VERI / "acilmis", ignore_errors=True)
