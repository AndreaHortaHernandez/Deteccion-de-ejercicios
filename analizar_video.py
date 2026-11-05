"""Analiza un video con los modelos entrenados: reconoce el ejercicio, cuenta repeticiones
y dice si cada una está bien o qué error tiene. Genera un video con el resultado dibujado.

Uso:
    python analizar_video.py --video prueba.mp4 --salida prueba_analizado.mp4 [--modelos /ruta/modelos]
"""
import argparse
import sys
from pathlib import Path

import cv2

import config
import pose


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--salida", type=Path, help="video de salida (por defecto <video>_analizado.mp4)")
    ap.add_argument("--modelos", default=config.DIR_MODELOS, type=Path)
    args = ap.parse_args()
    salida = args.salida or args.video.with_name(args.video.stem + "_analizado.mp4")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"No pude abrir {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analizador = pose.Analizador(args.modelos)
    out, i = None, 0
    with pose.DetectorPose() as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] > 960:
                esc = 960 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * esc)), interpolation=cv2.INTER_AREA)
            if out is None:
                out = cv2.VideoWriter(str(salida), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                      (frame.shape[1], frame.shape[0]))
            t = i / fps
            lm = det.detectar(frame, t * 1000)
            v = analizador.paso(pose.angulos(lm, frame.shape[1], frame.shape[0]), t)
            if v:
                estado = "BIEN" if v["ok"] else "MAL" if v["ok"] is False else "?"
                print(f"  {t:6.1f}s  {config.EJERCICIOS[v['ejercicio']]['nombre']:15s} rep {v['rep']:2d}: "
                      f"{estado:4s} {v['texto']} ({v['prob']:.0%})")
            pose.dibujar(frame, lm)
            analizador.dibujar_hud(frame, t)
            out.write(frame)
            i += 1
    cap.release()
    if out:
        out.release()
    print("\nResumen:")
    for ej, c in analizador.conteo.items():
        if c["reps"]:
            print(f"  {config.EJERCICIOS[ej]['nombre']:15s} reps: {c['reps']:2d}   bien: {c['bien']:2d}   mal: {c['mal']:2d}")
    if not analizador.historial:
        print("  No se detectaron repeticiones.")
    print(f"\nVideo anotado: {salida}")


if __name__ == "__main__":
    main()
