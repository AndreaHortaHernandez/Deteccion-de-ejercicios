"""Graba los 9 videos de una persona con la webcam de la laptop, guiando cada paso en pantalla.

La laptop va a la altura de la cadera, a ~3 m, con la pantalla hacia la persona. Para cada video:
    ESPACIO -> "Preparate" 3 s -> empieza a grabar -> quieto 2 s -> "AHORA" (8-10 reps)
    -> ESPACIO al terminar -> quieto 2 s -> deja de grabar -> escriben las reps contadas + ENTER
Los videos se guardan ya con su nombre (<videos>/<persona>/<ejercicio>__<etiqueta>__1.mp4) y las reps en
<videos>/<persona>/conteos.csv, así que no hace falta renombrar nada. Si se cierra, al volver a correrlo sigue
donde se quedó.

Uso (en la laptop, no en Colab; solo necesita opencv-python y numpy):
    python grabar_webcam.py --persona perez_ana --num 1
    python grabar_webcam.py --persona perez_ana --num 1 --camara 1     # si hay otra cámara
Teclas: ESPACIO = siguiente paso, R = repetir la toma, ESC = salir (lo grabado se queda).
Al terminar, suban la carpeta videos/<persona>/ completa (videos + conteos.csv) a Drive.
extraer_videos.py pasa esos conteos a datos/conteos_reales.csv.
"""
import argparse
import os
import sys
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

import config
from guia_grabacion import INSTRUCCIONES, PREPARACION, guardar_conteos, leer_conteos, plan, validar_persona

VENTANA = "Grabacion - Deteccion de ejercicios"
PREPARA_S, QUIETO_S = 3, 2
MAX_REPS_S = 90          # si nadie presiona ESPACIO, la fase de reps termina sola
FPS_MIN = 20

VERDE, AMARILLO, ROJO, BLANCO, GRIS = (90, 200, 90), (40, 200, 240), (60, 60, 230), (255, 255, 255), (170, 170, 170)


def ascii(s):
    """Las fuentes de OpenCV no tienen acentos."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def pitido(largo=False):
    def _():
        try:
            import winsound
            winsound.Beep(1200 if largo else 880, 450 if largo else 150)
        except Exception:  # noqa: BLE001 - sin sonido no pasa nada
            print("\a", end="", flush=True)
    threading.Thread(target=_, daemon=True).start()


def texto(img, s, y, escala=1.0, color=BLANCO, grosor=2, centro=True, x=30):
    s = ascii(s)
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, escala, grosor)
    if centro:
        x = max(10, (img.shape[1] - w) // 2)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_DUPLEX, escala, (0, 0, 0), grosor + 6, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_DUPLEX, escala, color, grosor, cv2.LINE_AA)
    return h


def envolver(s, n=70):
    palabras, lineas, actual = s.split(), [], ""
    for p in palabras:
        if len(actual) + len(p) + 1 > n:
            lineas.append(actual)
            actual = p
        else:
            actual = f"{actual} {p}".strip()
    return lineas + [actual] if actual else lineas


def franja(img, y0, y1, alfa=0.55):
    sub = img[y0:y1]
    img[y0:y1] = (sub * (1 - alfa)).astype(np.uint8)


def limpiar(s):
    """minúsculas, sin acentos ni ñ, espacios -> '_' (ej. 'Pérez López' -> 'perez_lopez')."""
    s = ascii(s)
    s = "".join(c if c.isalnum() or c in " _-" else "" for c in s)
    return "_".join(s.lower().replace("-", " ").split())


def pedir_nombre():
    """Pregunta apellido y nombre y devuelve 'apellido_nombre' (como en la práctica 2)."""
    while True:
        apellido, nombre = limpiar(input("Apellido(s): ")), limpiar(input("Nombre(s): "))
        if not (apellido and nombre):
            print("Apellido y nombre son obligatorios.")
            continue
        carpeta = f"{apellido}_{nombre}"
        if input(f"Tu carpeta será '{carpeta}'. ¿Correcto? (s/n): ").strip().lower() == "s":
            return carpeta


def abrir_camara(indice):
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(indice)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir la cámara {indice}. Prueben con --camara 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def corregir_fps(ruta, fps_real, fps_usado):
    """El VideoWriter necesita los fps al abrirse; si la cámara fue más lenta o más rápida de lo medido,
    se reescribe el archivo con los fps reales para que los tiempos (duración de cada rep) cuadren."""
    if abs(fps_real - fps_usado) / fps_usado < 0.05:
        return
    tmp = ruta.with_suffix(".tmp.mp4")
    cap = cv2.VideoCapture(str(ruta))
    ok, frame = cap.read()
    out = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps_real, (frame.shape[1], frame.shape[0]))
    while ok:
        out.write(frame)
        ok, frame = cap.read()
    cap.release()
    out.release()
    os.replace(tmp, ruta)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--persona", help="apellido_nombre (si no se da, se pregunta)")
    ap.add_argument("--num", type=int, help="número de la persona del 1 al 5 (si no se da, se pregunta)")
    ap.add_argument("--videos", default=config.RAIZ / "videos", type=Path)
    ap.add_argument("--conteos", type=Path, help="CSV de reps contadas (por defecto, conteos.csv dentro de la carpeta de la persona)")
    ap.add_argument("--camara", default=config.CAMARA, type=int)
    ap.add_argument("--omitir", default="", help="ejercicios que esta persona no va a grabar, separados por coma")
    ap.add_argument("--ventana", action="store_true", help="en ventana en lugar de pantalla completa")
    args = ap.parse_args()
    if not args.persona:
        args.persona = pedir_nombre()
    if not args.num:
        while not args.num:
            r = input("Número de persona (1 a 5, el que acordaron en el equipo): ").strip()
            args.num = int(r) if r.isdigit() and int(r) >= 1 else None
    validar_persona(args.persona)
    omitir = set(filter(None, args.omitir.split(",")))
    if omitir - set(config.EJERCICIOS):
        sys.exit(f"--omitir: usa {', '.join(config.EJERCICIOS)}")

    videos = plan(args.num, omitir)
    carpeta = args.videos / args.persona
    carpeta.mkdir(parents=True, exist_ok=True)
    args.conteos = args.conteos or carpeta / "conteos.csv"
    filas = leer_conteos(args.conteos)

    def hecho(archivo):
        vid = f"{args.persona}/{Path(archivo).stem}"
        return (carpeta / archivo).exists() and bool(filas.get(vid, {}).get("reps_reales"))

    cap = abrir_camara(args.camara)
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    if not args.ventana:
        cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # medir los fps reales de la cámara (con poca luz bajan)
    t0, n, frame = time.monotonic(), 0, None
    while time.monotonic() - t0 < 2.5:
        ok, f = cap.read()
        if ok:
            frame, n = f, n + 1
            vista = f.copy()
            texto(vista, "Midiendo la camara...", 60, 1.2)
            cv2.imshow(VENTANA, vista)
            cv2.waitKey(1)
    if frame is None:
        sys.exit("La cámara no entrega imagen.")
    fps_med = n / (time.monotonic() - t0)
    h_cam, w_cam = frame.shape[:2]
    print(f"Cámara: {w_cam}x{h_cam} a ~{fps_med:.0f} fps")

    i = next((k for k, (_, _, a) in enumerate(videos) if not hecho(a)), len(videos))
    estado, t_estado, escrito = ("inicio" if i < len(videos) else "fin"), time.monotonic(), ""
    writer, ruta_tmp, n_frames, t_grab = None, None, 0, 0.0

    def cambiar(nuevo):
        nonlocal estado, t_estado
        estado, t_estado = nuevo, time.monotonic()

    def detener_grabacion():
        nonlocal writer
        if writer is not None:
            writer.release()
            writer = None

    while True:
        ok, frame = cap.read()
        if not ok:
            if cv2.waitKey(10) & 0xFF == 27:
                break
            continue
        ahora = time.monotonic()
        dt = ahora - t_estado
        if writer is not None:
            writer.write(frame)
            n_frames += 1

        # ---- transiciones por tiempo
        if estado == "prepara" and dt >= PREPARA_S:
            ruta_tmp = carpeta / ("_grabando" + ".mp4")
            writer = cv2.VideoWriter(str(ruta_tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps_med, (w_cam, h_cam))
            n_frames, t_grab = 0, ahora
            cambiar("quieto1")
        elif estado == "quieto1" and dt >= QUIETO_S:
            pitido(largo=True)
            cambiar("reps")
        elif estado == "reps" and dt >= MAX_REPS_S:
            cambiar("quieto2")
        elif estado == "quieto2" and dt >= QUIETO_S:
            duracion = ahora - t_grab
            detener_grabacion()
            pitido()
            fps_real = n_frames / duracion if duracion > 0 else fps_med
            corregir_fps(ruta_tmp, fps_real, fps_med)
            escrito = ""
            cambiar("cuantas")

        # ---- dibujar (sobre una copia: el video se guarda limpio)
        vista = frame.copy()
        H, W = vista.shape[:2]
        k = W / 1280
        if i < len(videos):
            ej, et, archivo = videos[i]
            cab = f"Video {i + 1}/{len(videos)}  -  {config.EJERCICIOS[ej]['nombre']}  -  {et.replace('_', ' ').upper()}"
            franja(vista, 0, int(60 * k))
            texto(vista, cab, int(42 * k), 1.0 * k, BLANCO if et == "bien" else AMARILLO)
        if writer is not None:
            cv2.circle(vista, (W - int(40 * k), int(95 * k)), int(14 * k), ROJO, -1)
            texto(vista, f"REC {ahora - t_grab:4.0f} s", int(105 * k), 0.8 * k, ROJO, centro=False, x=W - int(220 * k))

        if estado == "inicio":
            franja(vista, int(H * 0.55), H)
            y = int(H * 0.62)
            texto(vista, f"Persona {args.num}: {args.persona}", y, 1.1 * k)
            y += int(55 * k)
            for linea in ["Laptop a la altura de la cadera, a ~3 m, persona DE PERFIL (lado izquierdo).",
                          "Revisen que se vea TODO el cuerpo durante todo el movimiento.",
                          f"Camara: {w_cam}x{h_cam} a {fps_med:.0f} fps" +
                          ("   [!] poca luz: prendan mas luces" if fps_med < FPS_MIN else "")]:
                texto(vista, linea, y, 0.75 * k, ROJO if "[!]" in linea else BLANCO)
                y += int(40 * k)
            texto(vista, "ESPACIO para empezar", y + int(20 * k), 1.1 * k, VERDE)
        elif estado == "listo":
            ej, et, archivo = videos[i]
            franja(vista, int(H * 0.58), H)
            y = int(H * 0.65)
            texto(vista, PREPARACION[ej], y, 0.65 * k, GRIS)
            y += int(45 * k)
            for linea in envolver(INSTRUCCIONES[(ej, et)], 60):
                texto(vista, linea, y, 0.95 * k)
                y += int(45 * k)
            texto(vista, "Ponte en posicion.  ESPACIO = empezar", y + int(15 * k), 1.0 * k, VERDE)
        elif estado == "prepara":
            texto(vista, "Preparate", int(H * 0.40), 1.6 * k, AMARILLO, 3)
            texto(vista, str(PREPARA_S - int(dt)), int(H * 0.65), 5.0 * k, AMARILLO, 10)
        elif estado in ("quieto1", "quieto2"):
            texto(vista, "QUIETO", int(H * 0.40), 2.2 * k, AMARILLO, 4)
            texto(vista, str(QUIETO_S - int(dt)), int(H * 0.65), 5.0 * k, AMARILLO, 10)
        elif estado == "reps":
            texto(vista, "AHORA: 8-10 repeticiones", int(H * 0.22), 1.6 * k, VERDE, 3)
            franja(vista, int(H * 0.86), H)
            texto(vista, "ESPACIO al terminar la ultima repeticion", int(H * 0.94), 0.9 * k, BLANCO)
        elif estado == "cuantas":
            franja(vista, int(H * 0.35), int(H * 0.80))
            texto(vista, "Cuantas repeticiones contaron?", int(H * 0.47), 1.3 * k, BLANCO, 3)
            texto(vista, (escrito or "_") + "  ", int(H * 0.63), 2.5 * k, VERDE, 5)
            texto(vista, "numero + ENTER     R = repetir la toma", int(H * 0.74), 0.85 * k, GRIS)
        elif estado == "fin":
            franja(vista, int(H * 0.30), int(H * 0.80))
            texto(vista, f"Listo, {args.persona}: {len(videos)} videos", int(H * 0.45), 1.4 * k, VERDE, 3)
            texto(vista, f"Guardados en videos/{args.persona}/", int(H * 0.57), 0.9 * k)
            texto(vista, "ESC para salir", int(H * 0.70), 0.9 * k, GRIS)

        cv2.imshow(VENTANA, vista)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:
            break
        if estado == "inicio" and tecla == 32:
            cambiar("listo")
        elif estado == "listo" and tecla == 32:
            cambiar("prepara")
        elif estado == "reps" and tecla == 32:
            cambiar("quieto2")
        elif estado == "cuantas":
            if tecla in (ord("r"), ord("R")):
                ruta_tmp.unlink(missing_ok=True)
                cambiar("listo")
            elif ord("0") <= tecla <= ord("9") and len(escrito) < 2:
                escrito += chr(tecla)
            elif tecla == 8:
                escrito = escrito[:-1]
            elif tecla == 13 and escrito:
                ej, et, archivo = videos[i]
                os.replace(ruta_tmp, carpeta / archivo)
                vid = f"{args.persona}/{Path(archivo).stem}"
                filas[vid] = {"id": vid, "ejercicio": ej, "etiqueta": et, "reps_reales": escrito}
                guardar_conteos(args.conteos, filas)
                print(f"  [ok] {archivo}: {escrito} reps")
                i += 1
                while i < len(videos) and hecho(videos[i][2]):
                    i += 1
                cambiar("listo" if i < len(videos) else "fin")

    detener_grabacion()
    if ruta_tmp is not None and ruta_tmp.exists():
        ruta_tmp.unlink()      # toma sin terminar
    cap.release()
    cv2.destroyAllWindows()
    hechos = sum(hecho(a) for _, _, a in videos)
    print(f"\n{hechos}/{len(videos)} videos de {args.persona} en {carpeta}")
    print(f"Reps contadas en {args.conteos}")
    if hechos == len(videos):
        print("Suban la carpeta a Drive: deteccion_ejercicios/videos/" + args.persona)


if __name__ == "__main__":
    main()
