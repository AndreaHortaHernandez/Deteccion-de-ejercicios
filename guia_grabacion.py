"""Guía para grabar los videos propios (ver PROTOCOLO_GRABACION.md).

Cuatro comandos:

  grabar     Guía EN VIVO mientras graban (en una laptop o en Colab, junto al celular que graba): dice
             qué video sigue, cuándo empezar a grabar, quedarse quieto, hacer las reps y detener;
             pregunta cuántas reps contaron y las guarda en conteos_reales.csv. Si se corta, al
             volver a correrlo sigue donde se quedó.
                 python guia_grabacion.py grabar --persona perez_ana --num 1

  guion      Hoja de grabación de cada persona: sus 9 videos en el orden que le toca, qué hacer en
             cada uno y el nombre del archivo. Se imprime y, con --salida, se guarda en Markdown.
                 python guia_grabacion.py guion --personas perez_ana,lopez_luis,ruiz_eva,diaz_jo,sosa_mar
                 python guia_grabacion.py guion --personas ... --salida guion_grabacion.md

  renombrar  Los videos del celular salen como VID_20261122_153012.mp4. Si se grabaron EN EL ORDEN
             DEL GUION, este comando les pone el nombre correcto y los mueve a <videos>/<persona>/.
             Sin --aplicar solo muestra qué haría.
                 python guia_grabacion.py renombrar --origen celular/ana --persona perez_ana --num 1 --videos videos
                 python guia_grabacion.py renombrar ... --aplicar

  revisar    Antes de extraer la pose: ¿están los 9 videos de cada persona, con nombre válido,
             horizontales, sin cámara lenta y con una duración razonable?
                 python guia_grabacion.py revisar --videos videos

La persona número N (1, 2, 3...) define el orden dentro de cada ejercicio, para que el cansancio no se
confunda con la técnica: 1 y 4 -> bien, error 1, error 2; 2 y 5 -> error 1, error 2, bien;
3 y 6 -> error 2, bien, error 1.
"""
import argparse
import re
import shutil
import sys
import time
from pathlib import Path

import config

EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".3gp"}

PREPARACION = {
    "sentadilla": "De pie, de perfil (lado izquierdo hacia la cámara), pies al ancho de los hombros.",
    "lagartija": "De perfil, en plancha. Revisen que se vea de la cabeza a los talones.",
    "curl": "De pie, de perfil, una mancuerna ligera (2-5 kg, o una botella con agua) en cada mano. "
            "Los DOS brazos al mismo tiempo, no alternado.",
}

INSTRUCCIONES = {
    ("sentadilla", "bien"): "Bajar hasta muslos paralelos al piso o más (rodilla ~90°), espalda neutra, talones en el piso.",
    ("sentadilla", "poco_profunda"): "Igual que bien, pero bajar solo hasta la mitad (rodilla ~120-130°).",
    ("sentadilla", "torso_inclinado"): "Profundidad normal, pero llevar el pecho hacia adelante (torso casi horizontal).",
    ("lagartija", "bien"): "Cuerpo recto de la cabeza a los talones, manos bajo los hombros, bajar hasta codo ~90°.",
    ("lagartija", "cadera_caida"): "Profundidad normal, pero con el abdomen flojo: la cadera se cae (arco en la espalda baja).",
    ("lagartija", "poco_profunda"): "Cuerpo recto, pero bajar solo un tercio (codo ~130°).",
    ("curl", "bien"): "Codos pegados al torso, torso quieto; extender casi del todo abajo y subir completo.",
    ("curl", "balanceo"): "Echar el torso hacia atrás (~15-20°) para subir con impulso. Moderado: no forzar la espalda.",
    ("curl", "recorrido_parcial"): "Solo la mitad del movimiento: no extender abajo ni subir del todo.",
}

COMUN = ("2 s quieto -> 8-10 repeticiones a ritmo normal (~2 s cada una) -> 2 s quieto -> detener la "
         "grabación. Todas las reps con la misma técnica.")


def orden_etiquetas(ejercicio, num):
    """Etiquetas de un ejercicio en el orden que le toca a la persona número num (1, 2, 3...)."""
    base = ["bien"] + list(config.EJERCICIOS[ejercicio]["errores"])
    k = (num - 1) % len(base)
    return base[k:] + base[:k]


def plan(num, omitir=()):
    """Lista de (ejercicio, etiqueta, nombre_de_archivo) en el orden de grabación."""
    return [(ej, et, f"{ej}__{et}__1.mp4")
            for ej in config.EJERCICIOS if ej not in omitir
            for et in orden_etiquetas(ej, num)]


def validar_persona(nombre):
    if not re.fullmatch(r"[a-z0-9_]+", nombre):
        sys.exit(f"Nombre de persona '{nombre}' no válido: usa minúsculas, sin acentos ni espacios (ej. perez_ana).")


def info_video(ruta):
    """(ancho, alto, fps, duración en s) leyendo el primer cuadro (ya rotado como se ve), o None."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    return w, h, fps, (n / fps if fps else 0.0)


def problemas_video(info):
    if info is None:
        return ["OpenCV no pudo abrirlo"]
    w, h, fps, dur = info
    p = []
    if h > w:
        p.append(f"vertical ({w}x{h}); debe ser horizontal")
    if min(w, h) < 480:
        p.append(f"resolución baja ({w}x{h})")
    if fps > 65:
        p.append(f"{fps:.0f} fps: parece cámara lenta")
    elif 0 < fps < 20:
        p.append(f"solo {fps:.0f} fps")
    if dur < 15:
        p.append(f"dura {dur:.0f} s (¿están las 8-10 reps?)")
    elif dur > 120:
        p.append(f"dura {dur:.0f} s (¿son dos videos en uno?)")
    return p


# --------------------------------------------------------------------------------- guion
def cmd_guion(args):
    personas = [p.strip() for p in args.personas.split(",") if p.strip()]
    omitir = set(args.omitir.split(",")) if args.omitir else set()
    lineas = ["# Guion de grabación", "",
              "Antes de empezar (ver PROTOCOLO_GRABACION.md):", "",
              "- Celular **horizontal**, en tripié a la altura de la cadera, a 2.5-4 m, **de perfil**.",
              "- 720p o 1080p a 30 fps. Sin cámara lenta, sin filtros, sin zoom.",
              "- Fondo liso, sin contraluz, sin otras personas en cuadro. Ropa ajustada.",
              "- Graben 5 s de prueba y revisen que se vea todo el cuerpo durante todo el movimiento.",
              "- **Un video por renglón, en este orden.** Si una toma sale mal, bórrenla en el celular y",
              "  repítanla; así `guia_grabacion.py renombrar` les pone el nombre correcto.",
              "- Mientras uno hace el ejercicio, otro cuenta las repeticiones y las anota.",
              f"- En cada video: {COMUN}", ""]
    for num, persona in enumerate(personas, 1):
        validar_persona(persona)
        lineas += ["", f"## Persona {num}: {persona}", "",
                   f"Carpeta en Drive: `videos/{persona}/`", ""]
        ej_actual = None
        for i, (ej, et, archivo) in enumerate(plan(num, omitir), 1):
            if ej != ej_actual:
                ej_actual = ej
                lineas += [f"### {config.EJERCICIOS[ej]['nombre']}", "", f"_{PREPARACION[ej]}_", ""]
            lineas += [f"{i}. **{et}** -> `{archivo}`",
                       f"   - {INSTRUCCIONES[(ej, et)]}",
                       "   - Reps contadas: ____   ¿Se vio todo el cuerpo? ____", ""]
    texto = "\n".join(lineas)
    print(texto)
    if args.salida:
        Path(args.salida).write_text(texto + "\n", encoding="utf-8")
        print(f"\nGuardado en {args.salida}")


# -------------------------------------------------------------------------------- grabar
CAMPOS_CONTEO = ["id", "ejercicio", "etiqueta", "reps_reales"]


def leer_conteos(ruta):
    import csv
    if not ruta.exists():
        return {}
    with open(ruta, encoding="utf-8") as fh:
        return {f["id"]: f for f in csv.DictReader(fh)}


def guardar_conteos(ruta, filas):
    import csv
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CAMPOS_CONTEO, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(filas.values(), key=lambda f: f["id"]))


def esperar(texto):
    input(f"\n   >> {texto} [Enter] ")


def cuenta(segundos, texto, rapido):
    print(f"\n   {texto}", end="", flush=True)
    for s in range(segundos, 0, -1):
        print(f" {s}...", end="", flush=True)
        if not rapido:
            time.sleep(1)
    print()


def cmd_grabar(args):
    """Guía en vivo: dice qué video sigue, cuándo grabar, quedarse quieto, hacer las reps y detener;
    pregunta las reps contadas y las guarda en conteos_reales.csv (si se corta, retoma donde iba)."""
    validar_persona(args.persona)
    omitir = set(args.omitir.split(",")) if args.omitir else set()
    videos = plan(args.num, omitir)
    filas = leer_conteos(args.conteos)
    linea = "=" * 64

    print(f"{linea}\nGRABACIÓN · Persona {args.num}: {args.persona} · {len(videos)} videos\n{linea}")
    print("Antes de empezar:\n"
          "  - Celular HORIZONTAL, en tripié a la altura de la cadera, a 2.5-4 m.\n"
          "  - De PERFIL, con el lado izquierdo hacia la cámara.\n"
          "  - Sin cámara lenta ni zoom. Fondo liso, sin contraluz, nadie más en cuadro.\n"
          "  - Roles: uno hace el ejercicio, otro cuenta las reps, otro maneja la cámara.\n"
          "  - Graben 5 s de prueba, revisen que se vea todo el cuerpo y BORREN esa prueba.")
    esperar("Cuando todo esté listo")

    ej_actual = None
    for i, (ej, et, archivo) in enumerate(videos, 1):
        vid = f"{args.persona}/{Path(archivo).stem}"
        if filas.get(vid, {}).get("reps_reales"):
            print(f"\n[{i}/{len(videos)}] {archivo}: ya grabado ({filas[vid]['reps_reales']} reps), sigue el próximo.")
            ej_actual = ej
            continue
        if ej != ej_actual:
            if ej_actual is not None:
                print(f"\n{linea}\nCAMBIO DE EJERCICIO. Descansen 1 minuto.")
            else:
                print(f"\n{linea}")
            print(f"{config.EJERCICIOS[ej]['nombre'].upper()}\n  {PREPARACION[ej]}\n{linea}")
            ej_actual = ej
        while True:
            print(f"\n[{i}/{len(videos)}] {config.EJERCICIOS[ej]['nombre']} · {et.upper().replace('_', ' ')}")
            print(f"   Qué hacer: {INSTRUCCIONES[(ej, et)]}")
            print(f"   Archivo:   {archivo}")
            esperar("Posición inicial. Cámara: EMPIECEN A GRABAR y luego")
            cuenta(2, "Quieto...", args.rapido)
            print("   ¡AHORA! 8-10 repeticiones, ~2 s cada una. Quien cuenta, cuente.")
            esperar("Al terminar la última repetición")
            cuenta(2, "Quieto otra vez...", args.rapido)
            print("   >> Cámara: DETENGAN LA GRABACIÓN.")
            r = input("\n   ¿Cuántas reps contaron? (número; R = repetir la toma): ").strip().lower()
            while not (r.isdigit() or r == "r"):
                r = input("   Escribe un número o R: ").strip().lower()
            if r == "r":
                print("   Borren esa toma del celular y vamos de nuevo.")
                continue
            filas[vid] = {"id": vid, "ejercicio": ej, "etiqueta": et, "reps_reales": r}
            guardar_conteos(args.conteos, filas)
            break

    print(f"\n{linea}\n¡Listo, {args.persona}! {len(videos)} videos grabados. Conteo guardado en {args.conteos}")
    print("Ahora:\n"
          f"  1. Revisen que en el celular haya exactamente {len(videos)} videos de esta persona (sin pruebas ni repetidos).\n"
          f"  2. Súbanlos SIN cambiarles el nombre a Drive: deteccion_ejercicios/celular/{args.persona}/\n"
          "  3. En el notebook, paso 3b: renombrar y revisar.")


# ----------------------------------------------------------------------------- renombrar
def cmd_renombrar(args):
    validar_persona(args.persona)
    omitir = set(args.omitir.split(",")) if args.omitir else set()
    esperados = plan(args.num, omitir)
    clave = (lambda p: p.stat().st_mtime) if args.orden == "fecha" else (lambda p: p.name.lower())
    videos = sorted((p for p in args.origen.iterdir() if p.is_file() and p.suffix.lower() in EXT), key=clave)
    if len(videos) != len(esperados):
        print(f"En {args.origen} hay {len(videos)} videos y se esperaban {len(esperados)}:")
        for v in videos:
            print("   ", v.name)
        sys.exit("Borren las tomas repetidas o de prueba (o usen --omitir si alguien se saltó un ejercicio).")

    destino = args.videos / args.persona
    print(f"Persona {args.num} ({args.persona}), orden por {args.orden}:\n")
    print(f"  {'video del celular':<32} {'dur.':>6}   nuevo nombre")
    conflictos = []
    for v, (_, _, nuevo) in zip(videos, esperados):
        info = info_video(v)
        dur = f"{info[3]:.0f} s" if info else "?"
        aviso = problemas_video(info) if info else []
        ext = v.suffix.lower()
        final = destino / (Path(nuevo).stem + ext)
        if final.exists():
            conflictos.append(final)
        print(f"  {v.name:<32} {dur:>6}   {final.name}" + (f"   [!] {'; '.join(aviso)}" if aviso else ""))
    print("\nRevisen que cada video corresponda a su nombre (abran uno o dos si dudan).")
    if conflictos:
        sys.exit(f"Ya existen {len(conflictos)} archivo(s) en {destino}; no se sobrescribe nada.")
    if not args.aplicar:
        print("Nada se movió. Si todo está bien, repitan con --aplicar.")
        return
    destino.mkdir(parents=True, exist_ok=True)
    for v, (_, _, nuevo) in zip(videos, esperados):
        shutil.move(str(v), str(destino / (Path(nuevo).stem + v.suffix.lower())))
    print(f"Listo: {len(videos)} videos movidos a {destino}")


# ------------------------------------------------------------------------------- revisar
def cmd_revisar(args):
    if not args.videos.is_dir():
        sys.exit(f"No existe la carpeta {args.videos}")
    carpetas = sorted(p for p in args.videos.iterdir() if p.is_dir() and not p.name.startswith("mendeley_"))
    if not carpetas:
        sys.exit(f"No hay carpetas de personas en {args.videos} (las de Mendeley no se revisan).")
    total_mal = 0
    for carpeta in carpetas:
        videos = sorted(p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXT)
        print(f"\n{carpeta.name}: {len(videos)} videos")
        vistos = set()
        for v in videos:
            partes = v.stem.split("__")
            ej, et = (partes + ["", ""])[:2]
            if len(partes) < 3 or config.clasificar_etiqueta(ej, et) is None:
                print(f"  [x] {v.name}: nombre no válido (ejercicio__etiqueta__n.mp4)")
                total_mal += 1
                continue
            vistos.add((ej, et))
            p = problemas_video(info_video(v))
            if p:
                print(f"  [!] {v.name}: {'; '.join(p)}")
                total_mal += 1
        faltan = [f"{ej}__{et}" for ej in config.EJERCICIOS
                  for et in ["bien"] + list(config.EJERCICIOS[ej]["errores"]) if (ej, et) not in vistos]
        if faltan:
            print(f"  Faltan: {', '.join(faltan)}")
    print(f"\n{len(carpetas)} persona(s) revisada(s); {total_mal} video(s) con avisos.")
    print("Si un ejercicio falta a propósito (alguien no pudo hacerlo), anótenlo para el reporte.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("guion", help="hoja de grabación por persona")
    g.add_argument("--personas", required=True, help="apellido_nombre separados por coma, en orden (1, 2, 3...)")
    g.add_argument("--omitir", default="", help="ejercicios que nadie va a grabar, separados por coma")
    g.add_argument("--salida", help="guardar el guion en este archivo Markdown")
    g.set_defaults(func=cmd_guion)

    gr = sub.add_parser("grabar", help="guía paso a paso mientras graban")
    gr.add_argument("--persona", required=True, help="apellido_nombre")
    gr.add_argument("--num", required=True, type=int, help="número de la persona (1, 2, 3...)")
    gr.add_argument("--omitir", default="", help="ejercicios que esta persona no va a grabar, separados por coma")
    gr.add_argument("--conteos", default=config.DIR_DATOS / "conteos_reales.csv", type=Path,
                    help="dónde guardar las reps contadas (en Colab: $DATOS/conteos_reales.csv)")
    gr.add_argument("--rapido", action="store_true", help=argparse.SUPPRESS)
    gr.set_defaults(func=cmd_grabar)

    r = sub.add_parser("renombrar", help="poner nombre a los videos del celular de una persona")
    r.add_argument("--origen", required=True, type=Path, help="carpeta con SOLO los videos de esa persona")
    r.add_argument("--persona", required=True, help="apellido_nombre")
    r.add_argument("--num", required=True, type=int, help="número de la persona en el guion (1, 2, 3...)")
    r.add_argument("--videos", default=Path("videos"), type=Path, help="carpeta videos/ del proyecto")
    r.add_argument("--omitir", default="", help="ejercicios que esta persona no grabó, separados por coma")
    r.add_argument("--orden", choices=("nombre", "fecha"), default="nombre",
                   help="cómo ordenar los videos del celular (el nombre suele llevar la hora)")
    r.add_argument("--aplicar", action="store_true", help="mover de verdad (sin esto solo muestra)")
    r.set_defaults(func=cmd_renombrar)

    v = sub.add_parser("revisar", help="revisar los videos propios antes de extraer la pose")
    v.add_argument("--videos", default=Path("videos"), type=Path)
    v.set_defaults(func=cmd_revisar)

    args = ap.parse_args()
    if getattr(args, "omitir", ""):
        malos = set(args.omitir.split(",")) - set(config.EJERCICIOS)
        if malos:
            sys.exit(f"--omitir: ejercicio(s) no válido(s): {', '.join(malos)}")
    args.func(args)


if __name__ == "__main__":
    main()
