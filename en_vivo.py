"""Versión local con la webcam (opcional: en Colab usa la celda de cámara del notebook).

Uso:
    python en_vivo.py [--modelos /ruta/modelos] [--camara 0]
Teclas: ESC = salir, R = reiniciar conteo.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

import config
import pose


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", default=config.DIR_MODELOS, type=Path)
    ap.add_argument("--camara", default=config.CAMARA, type=int)
    args = ap.parse_args()

    analizador = pose.Analizador(args.modelos)
    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {args.camara}")
    t0 = time.monotonic()
    with pose.DetectorPose() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            t = time.monotonic() - t0
            lm = det.detectar(frame, t * 1000)
            analizador.paso_pose(lm, frame.shape[1], frame.shape[0], t)
            pose.dibujar(frame, lm)
            analizador.dibujar_hud(frame, t)
            cv2.imshow("Deteccion de ejercicios", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            if k in (ord("r"), ord("R")):
                analizador.reiniciar()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
