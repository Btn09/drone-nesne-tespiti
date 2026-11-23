"""Parçalı (slicing) çıkarım.

Drone görüntüleri büyük (çoğu ~2000x1500) ve nesneler çok küçük. Görüntüyü olduğu gibi 640'a küçültünce
10-15 piksellik bir yaya 4-5 piksele iniyor ve model göremiyor. Bunun yerine görüntüyü birbiriyle örtüşen
parçalara bölüp her parçayı ayrı ayrı modele veriyorum, sonra kutuları tek görüntüde birleştiriyorum.
Büyük nesneler parçalara bölünebileceği için tam görüntü tahminini de ekliyorum.
"""
import numpy as np
import torch
from torchvision.ops import batched_nms


def parca_koordinatlari(genislik, yukseklik, boyut=640, ortusme=0.2):
    adim = int(boyut * (1 - ortusme))

    def baslangiclar(uzunluk):
        if uzunluk <= boyut:
            return [0]
        bas = list(range(0, uzunluk - boyut, adim))
        bas.append(uzunluk - boyut)  # son parça kenara yaslansın
        return bas

    return [(x, y, min(x + boyut, genislik), min(y + boyut, yukseklik))
            for y in baslangiclar(yukseklik) for x in baslangiclar(genislik)]


def sonuctan_al(sonuc, dx=0, dy=0):
    k = sonuc.boxes
    kutular = k.xyxy.clone()
    kutular[:, [0, 2]] += dx
    kutular[:, [1, 3]] += dy
    return kutular, k.conf, k.cls


def dilimli_tahmin(model, goruntu, boyut=640, ortusme=0.2, imgsz=640, conf=0.001, iou=0.6,
                   tam_goruntu=True, tam_imgsz=None, grup=16, max_det=1000):
    """goruntu: BGR numpy dizisi. Dönüş: xyxy kutular, skorlar, sınıflar (numpy)."""
    y, g = goruntu.shape[:2]
    koordinatlar = parca_koordinatlari(g, y, boyut, ortusme)
    parcalar = [goruntu[y0:y1, x0:x1] for x0, y0, x1, y1 in koordinatlar]

    kutular, skorlar, siniflar = [], [], []
    for i in range(0, len(parcalar), grup):
        sonuclar = model.predict(parcalar[i:i + grup], imgsz=imgsz, conf=conf, iou=iou,
                                 max_det=max_det, verbose=False)
        for sonuc, (x0, y0, _, _) in zip(sonuclar, koordinatlar[i:i + grup]):
            b, s, c = sonuctan_al(sonuc, x0, y0)
            kutular.append(b)
            skorlar.append(s)
            siniflar.append(c)

    if tam_goruntu:
        sonuc = model.predict(goruntu, imgsz=tam_imgsz or imgsz, conf=conf, iou=iou,
                              max_det=max_det, verbose=False)[0]
        b, s, c = sonuctan_al(sonuc)
        kutular.append(b)
        skorlar.append(s)
        siniflar.append(c)

    kutular, skorlar, siniflar = torch.cat(kutular), torch.cat(skorlar), torch.cat(siniflar)
    # Parçaların örtüştüğü yerlerde aynı nesne birden fazla kez bulunuyor; sınıf bazında NMS ile tekilleştir
    tut = batched_nms(kutular.float(), skorlar.float(), siniflar.long(), iou)[:max_det]
    return kutular[tut].cpu().numpy(), skorlar[tut].cpu().numpy(), siniflar[tut].cpu().numpy().astype(int)


def tam_tahmin(model, goruntu, imgsz=640, conf=0.001, iou=0.6, max_det=1000):
    sonuc = model.predict(goruntu, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)[0]
    b, s, c = sonuctan_al(sonuc)
    return b.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy().astype(int)


def parca_sayisi(genislik, yukseklik, boyut=640, ortusme=0.2):
    return len(parca_koordinatlari(genislik, yukseklik, boyut, ortusme))


if __name__ == "__main__":
    for g, y in [(1360, 765), (1920, 1080), (2000, 1500)]:
        print(g, y, parca_sayisi(g, y), np.array(parca_koordinatlari(g, y))[:3].tolist())
