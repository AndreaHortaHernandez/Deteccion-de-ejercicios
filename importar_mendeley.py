"""Descarga el dataset público de Mendeley y lo acomoda con los nombres que usa este proyecto.

Dataset: "A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels"
Duddela Sai Prashanth et al., Mendeley Data, v2, 2026. DOI: 10.17632/kgbb3yn47p.2 (licencia CC BY 4.0)

Solo baja los 3 ejercicios del proyecto y, por defecto, solo la vista de perfil (side):
~148 videos, ~512 MB. Queda así:
    <videos>/mendeley_subject_001/sentadilla__bien__side.mp4
    <videos>/mendeley_subject_001/sentadilla__mal__side.mp4
    ...
Los que ya están descargados se saltan, así que se puede volver a correr si se corta.

Uso:
    python importar_mendeley.py --videos /ruta/videos [--vistas side,diagonal] [--max-personas 5]
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

import config

API = "https://data.mendeley.com/public-api/datasets/kgbb3yn47p"
VERSION = 2
# la API rechaza el cliente por defecto de Python, así que se manda un User-Agent genérico
CABECERAS = {"User-Agent": "Mozilla/5.0 (deteccion-ejercicios)", "Accept": "application/json"}
CALIDAD = {"good": "bien", "bad": "mal"}


def pedir(url, timeout=120):
    return urllib.request.urlopen(urllib.request.Request(url, headers=CABECERAS), timeout=timeout)


def rutas_carpetas():
    """Devuelve {ruta legible: id} de todas las carpetas del dataset."""
    carpetas = json.load(pedir(f"{API}/folders/{VERSION}"))
    por_id = {c["id"]: c for c in carpetas}

    def ruta(c):
        partes = [c["name"]]
        while c.get("parent_id") in por_id:
            c = por_id[c["parent_id"]]
            partes.append(c["name"])
        return "/".join(reversed(partes))

    return {ruta(c): c["id"] for c in carpetas}


def descargar(url, destino):
    tmp = destino.with_suffix(".part")
    with pedir(url, timeout=300) as r, open(tmp, "wb") as f:
        while bloque := r.read(1 << 20):
            f.write(bloque)
    tmp.replace(destino)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--vistas", default="side", help="vistas separadas por coma: side, diagonal, front")
    ap.add_argument("--max-personas", type=int, default=0, help="limitar a las primeras N personas (para probar)")
    args = ap.parse_args()
    vistas = [v.strip() for v in args.vistas.split(",") if v.strip()]

    print("Leyendo el índice del dataset...")
    ids = rutas_carpetas()
    tareas = []
    for ej, cfg in config.EJERCICIOS.items():
        for calidad, etiqueta in CALIDAD.items():
            for vista in vistas:
                clave = f"Dataset Exercise-wise/{cfg['mendeley']}/{calidad}/{vista}"
                if clave not in ids:
                    sys.exit(f"No encontré la carpeta '{clave}' en el dataset")
                for f in json.load(pedir(f"{API}/files?folder_id={ids[clave]}&version={VERSION}")):
                    m = re.search(r"subject_(\d+)", f["filename"])
                    if not m:
                        continue
                    persona = f"mendeley_subject_{m.group(1)}"
                    destino = args.videos / persona / f"{ej}__{etiqueta}__{vista}.mp4"
                    tareas.append((persona, destino, f["size"], f["content_details"]["download_url"]))

    if args.max_personas:
        personas = sorted({t[0] for t in tareas})[:args.max_personas]
        tareas = [t for t in tareas if t[0] in personas]
    total_mb = sum(t[2] for t in tareas) / 1e6
    print(f"{len(tareas)} videos ({total_mb:.0f} MB) de {len({t[0] for t in tareas})} personas")

    bajados = saltados = 0
    fallas = []
    for i, (persona, destino, tam, url) in enumerate(tareas, 1):
        if destino.exists() and destino.stat().st_size == tam:
            saltados += 1
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        try:
            descargar(url, destino)
            bajados += 1
            print(f"  [{i}/{len(tareas)}] {persona}/{destino.name}")
        except Exception as e:  # noqa: BLE001 - se reporta y se sigue con el resto
            fallas.append(f"{persona}/{destino.name}: {e}")
    print(f"\nListo: {bajados} descargados, {saltados} ya estaban. Carpeta: {args.videos}")
    if fallas:
        print(f"{len(fallas)} fallaron (vuelve a correr el script para reintentar):")
        for f in fallas:
            print("  [x]", f)


if __name__ == "__main__":
    main()
