"""Drone görüntülerinde nesne tespiti servisi.

Çalıştırma:  uvicorn api.app:app
Dokümantasyon: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from src.cikarim import Tespit, bayttan_goruntu

durum = {}


@asynccontextmanager
async def lifespan(app):
    durum["tespit"] = Tespit()
    yield
    durum.clear()


app = FastAPI(title="Drone Nesne Tespiti", lifespan=lifespan)


def oku(dosya):
    try:
        return bayttan_goruntu(dosya.file.read())
    except ValueError:
        raise HTTPException(400, "Dosya bir görüntü değil")


@app.post("/tespit")
def tespit(dosya: UploadFile = File(...),
           parcali: bool = Query(False, description="Görüntüyü parçalara bölerek tara (daha yavaş)"),
           esik: float = Query(0.3, ge=0, le=1)):
    nesneler, sayim, _ = durum["tespit"].tespit_et(oku(dosya), parcali, esik)
    return {"toplam": len(nesneler), "sayim": sayim, "nesneler": nesneler}


@app.post("/tespit/gorsel", response_class=Response)
def tespit_gorsel(dosya: UploadFile = File(...), parcali: bool = False, esik: float = Query(0.3, ge=0, le=1)):
    goruntu = oku(dosya)
    _, _, (kutular, _, siniflar) = durum["tespit"].tespit_et(goruntu, parcali, esik)
    _, png = cv2.imencode(".jpg", Tespit.ciz(goruntu, kutular, siniflar), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(png.tobytes(), media_type="image/jpeg")
