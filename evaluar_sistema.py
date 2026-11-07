"""Evalúa el sistema COMPLETO, tal como corre en vivo, con las personas de prueba final.

entrenar.py mide cada modelo por separado. Aquí cada video de prueba pasa cuadro por cuadro por
el Analizador real (reconocimiento con retraso y voto, conteo de reps, técnica con el umbral):
- Ejercicio reconocido por video y cuánto tarda en reconocerlo.
- Conteo de repeticiones: sistema completo vs. contador con el ejercicio conocido y, si existe
  <datos>/conteos_reales.csv (columnas id, reps_reales), contra el conteo real (MAE, % exacto).
- Técnica por video: se promedia p(mal) de sus reps (igual que el veredicto de serie en vivo) y se
  compara con la etiqueta del video; "pareado" = en qué fracción de personas su video mal sale con
  más p(mal) que su video bien.
- Sensibilidad del conteo a DELTA_INICIO y RECUPERACION (solo si hay conteo real).
- Línea de tiempo del ángulo principal con cada rep marcada bien/mal.

Usa los modelos de <modelos>/evaluacion, entrenados SIN las personas de prueba.
El conteo real se anota a mano en <datos>/conteos_reales.csv (extraer_videos.py lo crea con una fila
vacía por video). También escribe <salidas>/conteos_para_llenar.csv con lo que contó el sistema.

Uso:
    python evaluar_sistema.py --datos /ruta/datos --modelos /ruta/modelos --salidas /ruta/salidas
                              [--conteos /ruta/conteos_reales.csv]
"""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import evaluacion as ev
import pose
from entrenar import cargar_sesiones

DELTAS = [15, 20, 25, 30, 35]
RECUPS = [0.5, 0.6, 0.7, 0.8, 0.9]


def simular(an, s):
    """Pasa una sesión por el Analizador (ya reiniciado). Devuelve el segundo en que reconoció un ejercicio."""
    t_det = None
    for a, t in zip(s["A"], s["t"]):
        an.paso(a, t)
        if t_det is None and an.ejercicio is not None:
            t_det = float(t - s["t"][0])
    return t_det


def leer_conteos(ruta):
    if not ruta or not Path(ruta).exists():
        return {}
    with open(ruta, encoding="utf-8") as fh:
        return {r["id"]: int(r["reps_reales"]) for r in csv.DictReader(fh) if r.get("reps_reales", "").strip()}


def linea_de_tiempo(casos, ruta):
    fig, axs = plt.subplots(len(casos), 1, figsize=(11, 2.3 * len(casos)), squeeze=False)
    for ax, (s, hist) in zip(axs[:, 0], casos):
        k = pose.SENALES.index(config.EJERCICIOS[s["ejercicio"]]["angulo"])
        ax.plot(s["t"], s["A"][:, k], c="k", lw=1)
        for v in hist:
            color = "tab:green" if v["ok"] else "tab:red" if v["ok"] is False else "gray"
            ax.axvspan(v["t_inicio"], v["t"], color=color, alpha=0.25)
        n_bien = sum(1 for v in hist if v["ok"])
        ax.set_title(f"{s['id']}  (etiqueta: {s['etiqueta']})  ·  reps: {len(hist)}  bien: {n_bien}  "
                     f"mal: {len(hist) - n_bien}", fontsize=9)
        ax.set_ylabel(f"{pose.SENALES[k]} (°)")
    axs[-1, 0].set_xlabel("tiempo (s)   ·   verde = rep bien, rojo = rep mal")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def sensibilidad(sesiones, reales, ruta, reporte):
    """MAE del conteo para cada combinación de umbrales (contador con el ejercicio conocido)."""
    con = [s for s in sesiones if s["id"] in reales]
    mae = np.zeros((len(DELTAS), len(RECUPS)))
    for i, d in enumerate(DELTAS):
        for j, r in enumerate(RECUPS):
            mae[i, j] = np.mean([abs(len(pose.contar_reps(s["t"], s["A"], s["ejercicio"], delta_inicio=d,
                                                          recuperacion=r)) - reales[s["id"]]) for s in con])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(mae, cmap="viridis_r")
    for i in range(len(DELTAS)):
        for j in range(len(RECUPS)):
            actual = DELTAS[i] == config.DELTA_INICIO and RECUPS[j] == config.RECUPERACION
            ax.text(j, i, f"{mae[i, j]:.2f}" + ("\n(actual)" if actual else ""), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold" if actual else None)
    ax.set_xticks(range(len(RECUPS)), RECUPS)
    ax.set_yticks(range(len(DELTAS)), DELTAS)
    ax.set(xlabel="RECUPERACION", ylabel="DELTA_INICIO (°)", title=f"MAE del conteo ({len(con)} videos)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    i, j = np.unravel_index(np.argmin(mae), mae.shape)
    reporte.append(f"\nSensibilidad del conteo ({len(con)} videos con conteo real): MAE actual = "
                   f"{mae[DELTAS.index(config.DELTA_INICIO), RECUPS.index(config.RECUPERACION)]:.2f}; "
                   f"mínimo {mae[i, j]:.2f} con DELTA_INICIO={DELTAS[i]}, RECUPERACION={RECUPS[j]}.\n"
                   "  Ojo: si eligen los umbrales con este mapa, ese MAE ya no es una estimación honesta\n"
                   "  (es el mismo problema de elegir el modelo con el conjunto de prueba).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=config.DIR_DATOS, type=Path)
    ap.add_argument("--modelos", default=config.DIR_MODELOS, type=Path)
    ap.add_argument("--salidas", default=config.DIR_SALIDAS, type=Path)
    ap.add_argument("--conteos", type=Path, help="CSV con id,reps_reales (por defecto <datos>/conteos_reales.csv)")
    args = ap.parse_args()
    args.salidas.mkdir(parents=True, exist_ok=True)
    dir_eval = args.modelos / "evaluacion"
    ruta_p = dir_eval / "personas_prueba.json"
    prueba = json.loads(ruta_p.read_text(encoding="utf-8")) if ruta_p.exists() else []
    if not prueba:
        sys.exit("No hay personas de prueba final: corre entrenar.py sin --sin-prueba (se necesitan 10+ personas).")
    reales = leer_conteos(args.conteos or args.datos / "conteos_reales.csv")

    print(f"Leyendo sesiones de {args.datos} ...")
    sesiones = cargar_sesiones(args.datos)
    de_prueba = [s for s in sesiones if s["persona"] in prueba]
    an = pose.Analizador(dir_eval)
    filas, casos, elegidos = [], [], set()
    for s in de_prueba:
        an.reiniciar()
        t_det = simular(an, s)
        ej = s["ejercicio"]
        ej_sis = max(an.conteo, key=lambda e: an.conteo[e]["reps"]) if any(
            c["reps"] for c in an.conteo.values()) else an.ejercicio
        hist = [v for v in an.historial if v["ejercicio"] == ej]
        evaluadas = [v for v in hist if v["ok"] is not None]
        p_mal = [v["p_mal"] for v in evaluadas if v["p_mal"] is not None]
        m_tec = an.m_tec.get(ej, {})
        umbral = m_tec.get("umbral_mal") or 0.5
        filas.append({
            "id": s["id"], "persona": s["persona"], "ejercicio": ej, "etiqueta": s["etiqueta"],
            "ejercicio_sistema": ej_sis, "segundos_para_reconocer": t_det,
            "reps_contador": len(pose.contar_reps(s["t"], s["A"], ej)),
            "reps_sistema": an.conteo[ej]["reps"],
            "reps_otro_ejercicio": sum(c["reps"] for e, c in an.conteo.items() if e != ej),
            "reps_reales": reales.get(s["id"], ""),
            "frac_mal": float(np.mean([not v["ok"] for v in evaluadas])) if evaluadas else np.nan,
            "p_mal_video": float(np.mean(p_mal)) if p_mal else np.nan,
            "tecnica_video": ("mal" if np.mean(p_mal) >= umbral else "bien") if p_mal else "",
        })
        if (ej, s["etiqueta"]) not in elegidos:
            elegidos.add((ej, s["etiqueta"]))
            casos.append((s, hist))

    # ----- reporte
    rep = [f"Personas de prueba final: {len(prueba)} ({', '.join(prueba)})   Videos: {len(filas)}",
           "Modelos: " + ", ".join(f"{k}: {v['algoritmo']}" for k, v in
                                   [("ejercicio", an.m_ej)] + list(an.m_tec.items()))]
    ok_ej = [f["ejercicio_sistema"] == f["ejercicio"] for f in filas]
    lat = [f["segundos_para_reconocer"] for f in filas if f["segundos_para_reconocer"] is not None]
    rep.append(f"\nEJERCICIO por video: {np.mean(ok_ej):.1%} correctos ({sum(ok_ej)}/{len(ok_ej)}). "
               f"Tiempo para reconocerlo: mediana {np.median(lat):.1f} s" if lat else "\nNo reconoció ningún ejercicio.")
    for f in filas:
        if f["ejercicio_sistema"] != f["ejercicio"]:
            rep.append(f"  [x] {f['id']}: dijo {f['ejercicio_sistema']}")
    ev.grafico_confusion(np.array([f["ejercicio"] for f in filas]),
                         np.array([str(f["ejercicio_sistema"]) for f in filas]),
                         sorted(config.EJERCICIOS), "Sistema completo: ejercicio por video (prueba)",
                         args.salidas / "sistema_confusion_ejercicio.png")

    d = np.array([f["reps_sistema"] - f["reps_contador"] for f in filas])
    rep.append(f"\nCONTEO: sistema completo vs. contador con el ejercicio conocido: diferencia media "
               f"{d.mean():+.2f}, MAE {np.abs(d).mean():.2f} (lo que se pierde por el retraso del reconocimiento)")
    con_real = [f for f in filas if f["reps_reales"] != ""]
    if con_real:
        for col in ("reps_contador", "reps_sistema"):
            e = np.array([f[col] - f["reps_reales"] for f in con_real])
            rep.append(f"  {col:14s} vs. real ({len(con_real)} videos): MAE {np.abs(e).mean():.2f}   "
                       f"exacto {np.mean(e == 0):.1%}   ±1 {np.mean(np.abs(e) <= 1):.1%}   sesgo {e.mean():+.2f}")
    else:
        rep.append(f"  Sin conteo real: anoten las reps en la columna reps_reales de {args.datos / 'conteos_reales.csv'}\n"
                   "  (lo crea extraer_videos.py; ver PROTOCOLO_GRABACION.md) para medir el error de conteo.")

    rep.append("\nTÉCNICA por video (promedio de p(mal) de sus reps vs. etiqueta del video):")
    for ej in config.EJERCICIOS:
        sub = [f for f in filas if f["ejercicio"] == ej and f["tecnica_video"]]
        if len({f["etiqueta"] for f in sub}) < 2:
            continue
        y = np.array([f["etiqueta"] for f in sub])
        pred = np.array([f["tecnica_video"] for f in sub])
        p = np.array([[1 - f["p_mal_video"], f["p_mal_video"]] for f in sub])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = ev.metricas(y, pred, p, ["bien", "mal"], "mal")
        pares = []
        for persona in {f["persona"] for f in sub}:
            b = [f["p_mal_video"] for f in sub if f["persona"] == persona and f["etiqueta"] == "bien"]
            mm = [f["p_mal_video"] for f in sub if f["persona"] == persona and f["etiqueta"] == "mal"]
            if b and mm:
                pares.append(1.0 if mm[0] > b[0] else 0.5 if mm[0] == b[0] else 0.0)
        rep.append(f"  {ej:11s} {len(sub):2d} videos   acierto bal. = {m['acierto_bal']:.3f}   "
                   f"recall mal = {m['recall_mal']:.3f}   ROC-AUC = {m['roc_auc']:.3f}   "
                   f"pareado = {np.mean(pares) if pares else np.nan:.3f} ({len(pares)} personas)")
        ev.grafico_confusion(y, pred, ["bien", "mal"], f"Sistema completo: técnica por video · {ej}",
                             args.salidas / f"sistema_confusion_tecnica_{ej}.png")

    if reales:
        sensibilidad(sesiones, reales, args.salidas / "sensibilidad_conteo.png", rep)
    linea_de_tiempo(casos, args.salidas / "linea_de_tiempo.png")

    campos = list(filas[0])
    with open(args.salidas / "sistema_por_video.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)
    with open(args.salidas / "conteos_para_llenar.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "ejercicio", "etiqueta", "reps_contador", "reps_reales"])
        for s in sesiones:
            w.writerow([s["id"], s["ejercicio"], s["etiqueta"],
                        len(pose.contar_reps(s["t"], s["A"], s["ejercicio"])), reales.get(s["id"], "")])
    texto = "\n".join(rep)
    (args.salidas / "reporte_sistema.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nReporte, sistema_por_video.csv, conteos_para_llenar.csv y gráficos en {args.salidas}")


if __name__ == "__main__":
    main()
