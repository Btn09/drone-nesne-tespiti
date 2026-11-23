"""Tarayıcıda drone görüntüsü yükleyip tespitleri görmek için arayüz.

Çalıştırma:  python demo.py   ->  http://127.0.0.1:7860
"""
from pathlib import Path

import cv2
import gradio as gr

from src.cikarim import Tespit

tespit = Tespit()
ORNEKLER = sorted(str(p) for p in (Path(__file__).parent / "ornekler").glob("*.jpg"))


def calistir(goruntu, parcali, esik):
    if goruntu is None:
        return None, "Önce bir görüntü yükleyin."
    bgr = cv2.cvtColor(goruntu, cv2.COLOR_RGB2BGR)
    nesneler, sayim, (kutular, _, siniflar) = tespit.tespit_et(bgr, parcali, esik)
    cizili = cv2.cvtColor(Tespit.ciz(bgr, kutular, siniflar), cv2.COLOR_BGR2RGB)
    satirlar = "\n".join(f"| {s} | {n} |" for s, n in sayim.items())
    return cizili, f"**Toplam {len(nesneler)} nesne**\n\n| Sınıf | Adet |\n|---|---|\n{satirlar}"


with gr.Blocks(title="Drone Nesne Tespiti") as arayuz:
    gr.Markdown("## Drone Görüntülerinde Nesne Tespiti\nYaya, araç, motosiklet, otobüs gibi 10 sınıf. "
                "Varsayılan olarak tam görüntü yüksek çözünürlükte taranıyor; parçalı tarama da seçilebilir.")
    with gr.Row():
        with gr.Column():
            giris = gr.Image(label="Drone görüntüsü")
            parcali = gr.Checkbox(False, label="Parçalı tarama (daha yavaş)")
            esik = gr.Slider(0.05, 0.9, 0.3, step=0.05, label="Güven eşiği")
            dugme = gr.Button("Tespit et", variant="primary")
        with gr.Column():
            cikis = gr.Image(label="Sonuç")
            ozet = gr.Markdown()
    dugme.click(calistir, [giris, parcali, esik], [cikis, ozet])
    if ORNEKLER:
        gr.Examples(ORNEKLER, [giris])

if __name__ == "__main__":
    arayuz.launch()
