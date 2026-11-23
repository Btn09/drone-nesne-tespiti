from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app

ORNEK = Path(__file__).resolve().parent.parent / "ornekler" / "drone_1.jpg"


def dosya():
    return {"dosya": ("drone.jpg", ORNEK.read_bytes(), "image/jpeg")}


@pytest.fixture(scope="module")
def istemci():
    with TestClient(app) as c:
        yield c


def test_tespit(istemci):
    sonuc = istemci.post("/tespit", files=dosya()).json()
    assert sonuc["toplam"] > 20
    assert sonuc["toplam"] == sum(sonuc["sayim"].values())
    assert all(0.3 <= n["skor"] <= 1 for n in sonuc["nesneler"])


def test_parcali_ve_tam_yakin_sonuc(istemci):
    # Test setinde iki yöntemin AP'si aynı; tek görüntüde de bulunan nesne sayıları yakın olmalı
    parcali = istemci.post("/tespit?parcali=true", files=dosya()).json()["toplam"]
    tam = istemci.post("/tespit?parcali=false", files=dosya()).json()["toplam"]
    assert parcali > 20 and tam > 20
    assert abs(parcali - tam) / max(parcali, tam) < 0.5


def test_esik_yukselince_azaliyor(istemci):
    dusuk = istemci.post("/tespit?esik=0.2", files=dosya()).json()["toplam"]
    yuksek = istemci.post("/tespit?esik=0.6", files=dosya()).json()["toplam"]
    assert yuksek <= dusuk


def test_gorsel(istemci):
    cevap = istemci.post("/tespit/gorsel", files=dosya())
    assert cevap.status_code == 200
    assert cevap.content[:2] == b"\xff\xd8"


def test_gecersiz_dosya(istemci):
    cevap = istemci.post("/tespit", files={"dosya": ("a.txt", b"merhaba", "text/plain")})
    assert cevap.status_code == 400


def test_gecersiz_esik(istemci):
    assert istemci.post("/tespit?esik=2", files=dosya()).status_code == 422
