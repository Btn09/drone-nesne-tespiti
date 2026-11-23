"""Tek görüntü için tespit ve çizim. API ve demo bunu kullanıyor."""
import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .dilimleme import dilimli_tahmin, tam_tahmin
from .veri import KOK, SINIFLAR_TR

MODEL_YOLU = Path(os.environ.get("MODEL_YOLU", KOK / "models" / "yolo11s_1024.pt"))
RENKLER = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
           (10, 249, 72), (23, 204, 146), (134, 219, 61), (211, 188, 0), (255, 115, 100)]


class Tespit:
    def __init__(self, model_yolu=MODEL_YOLU):
        self.model = YOLO(model_yolu)

    def tespit_et(self, goruntu, parcali=False, esik=0.3):
        """goruntu: BGR numpy dizisi.
        Varsayılan: tam görüntü 1536 çözünürlükte. Testte parçalı taramayla aynı AP'yi 5 kat hızlı veriyor."""
        if parcali:
            kutular, skorlar, siniflar = dilimli_tahmin(self.model, goruntu, boyut=640, imgsz=1024,
                                                        tam_imgsz=1024)
        else:
            kutular, skorlar, siniflar = tam_tahmin(self.model, goruntu, imgsz=1536)
        tut = skorlar >= esik
        kutular, skorlar, siniflar = kutular[tut], skorlar[tut], siniflar[tut]
        nesneler = [{"sinif": SINIFLAR_TR[c], "skor": round(float(s), 3),
                     "kutu": [round(float(v), 1) for v in k]}
                    for k, s, c in zip(kutular, skorlar, siniflar)]
        sayim = dict(Counter(SINIFLAR_TR[c] for c in siniflar).most_common())
        return nesneler, sayim, (kutular, skorlar, siniflar)

    @staticmethod
    def ciz(goruntu, kutular, siniflar, kalinlik=2):
        cikti = goruntu.copy()
        for (x0, y0, x1, y1), c in zip(kutular.astype(int), siniflar):
            cv2.rectangle(cikti, (x0, y0), (x1, y1), RENKLER[c], kalinlik)
        return cikti


def bayttan_goruntu(veri):
    goruntu = cv2.imdecode(np.frombuffer(veri, np.uint8), cv2.IMREAD_COLOR)
    if goruntu is None:
        raise ValueError("Dosya okunamadı")
    return goruntu
