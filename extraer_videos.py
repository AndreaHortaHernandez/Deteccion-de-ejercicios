"""Extrae la pose (33 puntos por frame) de los videos grabados y la guarda en CSV.

Estructura esperada de los videos:
    <videos>/<apellido_nombre>/<ejercicio>__<etiqueta>__<n>.mp4
    ej. videos/horta_andrea/sentadilla__mal__2.mp4

Salida: <datos>/<apellido_nombre>/<mismo_nombre>.csv
Los videos ya procesados se saltan (si el CSV existe y es más nuevo que el video).

Uso:
    python extraer_videos.py --videos /ruta/videos --datos /ruta/datos
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

import config
import pose

EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}


def validar_nombre(ruta):
    """Devuelve (ejercicio, etiqueta) o lanza ValueError explicando qué está mal."""
    partes = ruta.stem.split("__")
    if len(partes) < 2:
        raise ValueError("el nombre debe ser ejercicio__etiqueta__n (con doble guion bajo)")
    ej, et = partes[0].lower(), partes[1].lower()
    if ej not in config.EJERCICIOS:
        raise ValueError(f"ejercicio '{ej}' no válido; usa: {', '.join(config.EJERCICIOS)}")
    if config.clasificar_etiqueta(ej, et) is None:
        validas = list(config.EJERCICIOS[ej]["etiquetas"]) + list(config.EJERCICIOS[ej].get("errores", {}))
        raise ValueError(f"etiqueta '{et}' no válida para {ej}; usa: {', '.join(validas)}")
    return ej, et


def actualizar_conteos(dir_datos, ids):
    """Agrega a <datos>/conteos_reales.csv una fila vacía por cada video que todavía no esté, para
    anotar a mano cuántas repeticiones se hicieron (PROTOCOLO_GRABACION.md). Nunca borra lo anotado."""
    ruta = dir_datos / "conteos_reales.csv"
    filas = []
    if ruta.exists():
        with open(ruta, encoding="utf-8") as fh:
            filas = list(csv.DictReader(fh))
    ya = {f["id"] for f in filas}
    nuevos = [i for i in ids if i not in ya]
    if not nuevos:
        return 0
    for i in nuevos:
        ej, et = i.split("/")[-1].split("__")[:2]
        filas.append({"id": i, "ejercicio": ej, "etiqueta": et, "reps_reales": ""})
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "ejercicio", "etiqueta", "reps_reales"], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(filas, key=lambda f: f["id"]))
    return len(nuevos)


def procesar(video, salida, detector):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("OpenCV no pudo abrir el video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    filas, i, w, h = [], 0, None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # se reduce el video: la pose no necesita más resolución y es mucho más rápido
        if frame.shape[1] > config.ANCHO_POSE:
            esc = config.ANCHO_POSE / frame.shape[1]
            frame = cv2.resize(frame, (config.ANCHO_POSE, int(frame.shape[0] * esc)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        t = i / fps
        lm = detector.detectar(frame, t * 1000)
        if lm is not None:
            filas.append((t, lm))
        i += 1
    cap.release()
    if not filas:
        raise ValueError(f"no se detectó a ninguna persona en {i} frames")
    pose.guardar_sesion(salida, filas, w, h)
    return i, len(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--datos", default=config.DIR_DATOS, type=Path)
    ap.add_argument("--forzar", action="store_true", help="reprocesar aunque el CSV ya exista")
    args = ap.parse_args()

    videos = sorted(p for p in args.videos.rglob("*") if p.suffix.lower() in EXT)
    if not videos:
        sys.exit(f"No encontré videos en {args.videos}")
    print(f"{len(videos)} videos encontrados en {args.videos}")
    errores, nuevos, validos = [], 0, []
    for v in videos:
        rel = v.relative_to(args.videos)
        try:
            validar_nombre(v)
        except ValueError as e:
            errores.append(f"{rel}: {e}")
            continue
        persona = v.parent.name if v.parent != args.videos else "sin_nombre"
        salida = args.datos / persona / (v.stem + ".csv")
        validos.append(f"{persona}/{v.stem}")
        if salida.exists() and salida.stat().st_mtime >= v.stat().st_mtime and not args.forzar:
            continue
        # un detector nuevo por video: el modo video de MediaPipe sigue a la persona entre frames
        try:
            with pose.DetectorPose() as det:
                total, conpose = procesar(v, salida, det)
            nuevos += 1
            print(f"  [ok] {rel}: {conpose}/{total} frames con pose")
            if conpose < 0.7 * total:
                print(f"    [!] en {total - conpose} frames no se vio a la persona; revisa el encuadre")
        except Exception as e:  # noqa: BLE001 - se reporta y se sigue con el resto
            errores.append(f"{rel}: {e}")
    print(f"\nListo: {nuevos} videos procesados. CSV en {args.datos}")
    agregados = actualizar_conteos(args.datos, validos)
    if agregados:
        print(f"Se agregaron {agregados} filas a {args.datos / 'conteos_reales.csv'}: anoten las reps reales "
              "en la columna reps_reales (las que tengan).")
    if errores:
        print(f"\n{len(errores)} video(s) con problemas (no se usaron):")
        for e in errores:
            print("  [x]", e)


if __name__ == "__main__":
    main()
