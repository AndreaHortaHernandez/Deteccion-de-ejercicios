"""Experimentos para el reporte. Solo usan las personas de ENTRENAMIENTO: la prueba final no se toca
(salvo al final de la comparación y en el punto 3, que miden en las personas de prueba modelos
entrenados sin ellas).

0. comparacion: versión 1 contra versión 2 con los mismos modelos y las mismas particiones por
   persona (pareado, 5 x 5 folds), con Wilcoxon por persona y la prueba final.
1. ablaciones: cuánto aporta cada decisión de preprocesamiento (suavizado, lado fijo, p5/p95,
   largo de ventana), cada familia de rasgos de técnica y la secuencia completa (ROCKET), frente
   a la versión 1 del proyecto.
2. curva: curva de aprendizaje por número de PERSONAS de entrenamiento (sesgo vs. varianza:
   ¿faltan datos o falta modelo?).
3. vista: robustez al punto de vista. Modelos entrenados con la vista de perfil, probados con
   la vista diagonal de las personas de prueba. Requiere bajar esa vista:
       python importar_mendeley.py --videos <videos> --vistas side,diagonal
       python extraer_videos.py --videos <videos> --datos <datos>

Las ablaciones y la curva usan modelos fijos (LogReg y Random Forest, sin búsqueda de
hiperparámetros ni umbral): aquí interesa cuánto informa cada variante, no exprimir un modelo.

Uso:
    python experimentos.py --datos /ruta/datos --modelos /ruta/modelos --salidas /ruta/salidas
                           [--solo comparacion,ablaciones,curva,vista]
"""
import argparse
import contextlib
import csv
import json
import time
import warnings
from pathlib import Path


import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import wilcoxon

import config
import evaluacion as ev
import pose
from entrenar import cargar_sesiones, dataset_ejercicio, dataset_tecnica, dividir_personas
from modelos import Rocket


@contextlib.contextmanager
def con_config(**cambios):
    """Cambia parámetros de config.py mientras dura el bloque."""
    viejos = {k: getattr(config, k) for k in cambios}
    for k, v in cambios.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in viejos.items():
            setattr(config, k, v)


def modelos_fijos():
    return {
        "LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")),
        "RandomForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0)),
    }


def modelo_rocket():
    return make_pipeline(Rocket(n_kernels=1000), StandardScaler(),
                         RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight="balanced"))


def cv_rapida(modelo, X, y, g, n_rep=2):
    """F1 macro (media y DE entre folds) con StratifiedGroupKFold por persona, repetida."""
    vals = []
    for r in range(n_rep):
        cv = StratifiedGroupKFold(min(5, len(set(g))), shuffle=True, random_state=config.SEMILLA + r)
        for tr, te in cv.split(X, y, g):
            m = clone(modelo).fit(X[tr], y[tr])
            vals.append(f1_score(y[te], m.predict(X[te]), average="macro"))
    return float(np.mean(vals)), float(np.std(vals))


def reprocesar(sesiones):
    """Recalcula las señales con la config actual a partir de los landmarks crudos."""
    res = []
    for s in sesiones:
        t, A = pose.senales_sesion(*s["crudo"])
        res.append({**s, "t": t, "A": A})
    return res


# ----------------------------------------------------------------- rasgos de la versión 1
def rasgos_ventana_v1(A):
    A = np.asarray(A, dtype=np.float64)[:, :len(pose.ANGULOS)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.concatenate([np.nanmean(A, 0), np.nanstd(A, 0), np.nanmin(A, 0), np.nanmax(A, 0)])


def rasgos_rep_v1(A, t):
    A = np.asarray(A, dtype=np.float64)[:, :len(pose.ANGULOS)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mn, mx = np.nanmin(A, 0), np.nanmax(A, 0)
        return np.concatenate([mn, mx, np.nanmean(A, 0), np.nanstd(A, 0), mx - mn, [t[-1] - t[0]]])


def datasets_v1(sesiones):
    """Rasgos de la versión 1. Devuelve (X, y, persona, id_video) para ejercicio y para cada técnica."""
    X, y, g, ids = [], [], [], []
    for s in sesiones:
        for W in pose.ventanas(s["t"], s["A"], paso=config.PASO_ENTRENAR_S):
            if pose.hay_movimiento(W):
                X.append(rasgos_ventana_v1(W))
                y.append(s["ejercicio"])
                g.append(s["persona"])
                ids.append(s["id"])
    ej = (np.array(X), np.array(y), np.array(g), np.array(ids))
    tec = {}
    for e in config.EJERCICIOS:
        Xt, yt, gt, it = [], [], [], []
        for s in sesiones:
            if s["ejercicio"] == e:
                for A, t in pose.contar_reps(s["t"], s["A"], e):
                    Xt.append(rasgos_rep_v1(A, t))
                    yt.append(s["etiqueta"])
                    gt.append(s["persona"])
                    it.append(s["id"])
        tec[e] = (np.array(Xt), np.array(yt), np.array(gt), np.array(it))
    return ej, tec


# ----------------------------------------------------------------- 1. ablaciones
PREPRO = [
    ("v2 (por defecto: prepro v2 + rasgos basicos)", {}),
    ("sin suavizado (mediana)", {"MEDIANA_FRAMES": 1}),
    ("lado elegido en cada cuadro", {"LADO_FIJO": False}),
    ("min/max en lugar de p5/p95", {"ESTADISTICOS": "minmax"}),
]
VENTANAS = [("ventana de 1 s", {"VENTANA_S": 1.0, "MIN_FRAMES_VENTANA": 4}),
            ("ventana de 3 s", {"VENTANA_S": 3.0})]
V1 = {"MEDIANA_FRAMES": 1, "LADO_FIJO": False, "ESTADISTICOS": "minmax"}
TODAS = ("basicos", "estad", "fase", "flex")


def _filas_tarea(filas, tarea, variante, X, y, g, modelos):
    for nom, mod in modelos.items():
        if len(set(y)) < 2 or len(set(g)) < 2:
            continue
        m, d = cv_rapida(mod, X, y, g)
        filas.append({"tarea": tarea, "variante": variante, "modelo": nom, "f1_media": m, "f1_de": d})
        print(f"  {tarea:22s} {variante:42s} {nom:13s} F1 = {m:.3f} ± {d:.3f}", flush=True)


def ablaciones(base, dir_salidas, reporte):
    filas = []
    fijos = modelos_fijos()
    # preprocesamiento (con los rasgos por defecto): afecta a los dos modelos
    for variante, cambios in PREPRO + VENTANAS:
        with con_config(**cambios):
            ses = reprocesar(base)
            Xs, y, g, _, _ = dataset_ejercicio(ses, True)
            _filas_tarea(filas, "Ejercicio", variante, Xs["tabular"], y, g, fijos)
            if cambios.get("VENTANA_S"):
                continue  # el largo de ventana no cambia la técnica
            for ej in config.EJERCICIOS:
                Xs, y, g, _, _, _ = dataset_tecnica(ses, ej, True)
                _filas_tarea(filas, f"Tecnica {ej}", variante, Xs["tabular"], y, g, fijos)
    with con_config(**V1):
        (Xe, ye, ge, _), tec = datasets_v1(reprocesar(base))
        _filas_tarea(filas, "Ejercicio", "≈ v1 (sin suavizado, lado por cuadro, rasgos v1)", Xe, ye, ge, fijos)
        for ej, (X, y, g, _) in tec.items():
            _filas_tarea(filas, f"Tecnica {ej}", "≈ v1 (sin suavizado, lado por cuadro, rasgos v1)", X, y, g, fijos)

    # familias de rasgos de técnica (con el preprocesamiento por defecto)
    with con_config(RASGOS_REP=TODAS):
        nombres = np.array(pose.nombres_rasgos_rep())
        grupos = ev.grupos_rasgos(nombres)
        fam = np.array(["fase" if n.startswith("fase_") else "flex" if n.startswith("flex_")
                        else "basicos" if n.startswith("bas_") else "estad" for n in nombres])
        for ej in config.EJERCICIOS:
            Xs, y, g, _, _, _ = dataset_tecnica(base, ej, True)
            X = Xs["tabular"]
            principal = config.EJERCICIOS[ej]["angulo"]
            conjuntos = [
                ("rasgos: basicos (v1, por defecto)", np.isin(fam, ["basicos"])),
                ("rasgos: estad + fase + flex (v2, 47)", np.isin(fam, ["estad", "fase", "flex"])),
                ("rasgos: solo estad (genéricos)", np.isin(fam, ["estad"])),
                ("rasgos: solo fase", np.isin(fam, ["fase"])),
                ("rasgos: solo flex", np.isin(fam, ["flex"])),
                ("rasgos: estad + fase", np.isin(fam, ["estad", "fase"])),
                ("rasgos: v2 sin las 2 medidas nuevas", (fam != "basicos") & ~np.isin(grupos, pose.MEDIDAS)),
                (f"rasgos: v2 solo el ángulo principal ({principal})",
                 (fam != "basicos") & np.isin(grupos, [principal, "fase (tiempos/velocidad)"])),
            ]
            for variante, cols in conjuntos:
                _filas_tarea(filas, f"Tecnica {ej}", variante, X[:, cols], y, g, fijos)
            _filas_tarea(filas, f"Tecnica {ej}", "secuencia completa (ROCKET + Ridge)", Xs["secuencia"], y, g,
                         {"ROCKET": modelo_rocket()})

    with open(dir_salidas / "ablaciones.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    reporte.append(f"\n{'=' * 78}\n1. ABLACIONES (F1 macro, StratifiedGroupKFold por persona x 2)\n{'=' * 78}")
    for tarea in dict.fromkeys(f["tarea"] for f in filas):
        reporte.append(f"\n{tarea}")
        sub = [f for f in filas if f["tarea"] == tarea]
        for f in sub:
            reporte.append(f"  {f['variante']:48s} {f['modelo']:13s} {f['f1_media']:.3f} ± {f['f1_de']:.3f}")
        _grafico_ablacion(sub, tarea, dir_salidas / f"ablaciones_{tarea.lower().replace(' ', '_')}.png")


def _grafico_ablacion(filas, tarea, ruta):
    variantes = list(dict.fromkeys(f["variante"] for f in filas))
    modelos = list(dict.fromkeys(f["modelo"] for f in filas))
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(variantes) * max(1, len(modelos) / 1.5) + 1.5))
    alto = 0.8 / len(modelos)
    con_leyenda = set()
    for j, m in enumerate(modelos):
        for i, v in enumerate(variantes):
            f = next((f for f in filas if f["variante"] == v and f["modelo"] == m), None)
            if f:
                ax.barh(i + j * alto, f["f1_media"], alto, xerr=f["f1_de"], color=f"C{j}",
                        label=None if m in con_leyenda else m)
                con_leyenda.add(m)
    ax.set_yticks(np.arange(len(variantes)) + 0.4 - alto / 2, variantes, fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel="F1 macro (media ± DE entre folds)", title=f"Ablaciones · {tarea}", xlim=(0, 1.05))
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------- 0. v1 vs. v2
def _datos_versiones(sesiones):
    """{tarea: {"v1": (X, y, persona, id), "v2": (...)}} con el preprocesamiento de cada versión."""
    with con_config(**V1):
        ej1, tec1 = datasets_v1(reprocesar(sesiones))
    Xs, y, _, ids, per = dataset_ejercicio(sesiones, True)
    datos = {"Ejercicio": {"v1": ej1, "v2": (Xs["tabular"], y, per, ids)}}
    for ej in config.EJERCICIOS:
        Xs, y, _, ids, per, _ = dataset_tecnica(sesiones, ej, True)
        datos[f"Tecnica {ej}"] = {"v1": tec1[ej], "v2": (Xs["tabular"], y, per, ids)}
    return datos


def _video(y, pred, ids):
    """Decisión por video: la etiqueta mayoritaria de sus reps/ventanas."""
    vid = sorted(set(ids))
    yv = np.array([y[ids == v][0] for v in vid])
    pv = np.array([max(set(pred[ids == v]), key=list(pred[ids == v]).count) for v in vid])
    return yv, pv


def comparacion(entr, prueba, dir_salidas, reporte, n_rep=5):
    """v1 contra v2 con los MISMOS modelos y las MISMAS particiones por persona.

    Para cada repetición se reparten las personas en 5 grupos al azar; las dos versiones usan
    esos mismos grupos, así que la comparación es pareada. Se mide F1 macro por rep/ventana, acierto
    balanceado por video y acierto balanceado por persona (Wilcoxon pareado v2 vs. v1). Al final,
    cada versión se entrena con las personas de entrenamiento y se mide en las de prueba final.
    """
    modelos = {**modelos_fijos(),
               # el SVM de la versión 1 (C=10)
               "SVM_RBF": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))}
    datos = _datos_versiones(entr)
    datos_p = _datos_versiones(prueba) if prueba else None
    personas = sorted({s["persona"] for s in entr})
    filas = []
    reporte.append(f"\n{'=' * 78}\n0. VERSIÓN 1 vs. VERSIÓN 2 (mismos modelos, mismas particiones; "
                   f"{n_rep} x 5 folds por persona)\n{'=' * 78}")
    reporte.append("  F1 = F1 macro por rep/ventana (media ± DE entre folds); video = acierto bal. por video;\n"
                   "  p = Wilcoxon pareado del acierto bal. por persona (v2 vs. v1); prueba = F1 en las personas de prueba final")
    for tarea, dv in datos.items():
        reporte.append(f"\n{tarea}")
        for nom, mod in modelos.items():
            res = {}
            for ver, (X, y, per, ids) in dv.items():
                f1s, vids, pp = [], [], np.zeros((n_rep, len(personas)))
                for r in range(n_rep):
                    orden = np.random.default_rng(config.SEMILLA + r).permutation(personas)
                    pred = np.empty(len(y), dtype=object)
                    for f in range(5):
                        te = np.isin(per, orden[f::5])
                        m = clone(mod).fit(X[~te], y[~te])
                        pred[te] = m.predict(X[te])
                        f1s.append(f1_score(y[te], pred[te].astype(str), average="macro"))
                    pred = pred.astype(str)
                    yv, pv = _video(y, pred, ids)
                    vids.append(balanced_accuracy_score(yv, pv))
                    pp[r] = [balanced_accuracy_score(y[per == p], pred[per == p]) if (per == p).any() else np.nan
                             for p in personas]
                prueba_f1 = np.nan
                if datos_p:
                    Xp, yp, _, _ = datos_p[tarea][ver]
                    if len(yp):
                        prueba_f1 = f1_score(yp, clone(mod).fit(X, y).predict(Xp), average="macro")
                res[ver] = (np.mean(f1s), np.std(f1s), np.mean(vids), np.nanmean(pp, axis=0), prueba_f1)
            a, b = res["v1"][3], res["v2"][3]
            ok = ~np.isnan(a) & ~np.isnan(b)
            p = 1.0 if np.allclose(a[ok], b[ok]) else float(wilcoxon(b[ok], a[ok], zero_method="zsplit").pvalue)
            for ver in ("v1", "v2"):
                m, d, v, _, tf = res[ver]
                filas.append({"tarea": tarea, "modelo": nom, "version": ver, "f1_media": m, "f1_de": d,
                              "video_acierto_bal": v, "prueba_f1": tf, "wilcoxon_p": p})
            (m1, d1, v1, _, t1), (m2, d2, v2, _, t2) = res["v1"], res["v2"]
            reporte.append(f"  {nom:13s} v1 F1 {m1:.3f}±{d1:.3f} video {v1:.3f} prueba {t1:.3f} | "
                           f"v2 F1 {m2:.3f}±{d2:.3f} video {v2:.3f} prueba {t2:.3f} | dif F1 {m2 - m1:+.3f}  p = {p:.3f}")
            print(f"  {tarea:22s} {nom:13s} v1 {m1:.3f}  v2 {m2:.3f}  p = {p:.3f}", flush=True)
    with open(dir_salidas / "comparacion_v1_v2.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)


# ----------------------------------------------------------------- 2. curva de aprendizaje
def curva_aprendizaje(base, dir_salidas, reporte, n_val=5, n_rep=10):
    personas = sorted({s["persona"] for s in base})
    if len(personas) < n_val + 4:
        reporte.append("\n2. CURVA DE APRENDIZAJE: se necesitan al menos 9 personas de entrenamiento.")
        return
    tamanos = np.unique(np.linspace(3, len(personas) - n_val, 6).astype(int))
    reporte.append(f"\n{'=' * 78}\n2. CURVA DE APRENDIZAJE (acierto bal.; {n_val} personas de validación, "
                   f"{n_rep} sorteos por tamaño)\n{'=' * 78}")
    tareas = [("Ejercicio", "ejercicio", dataset_ejercicio(base, True))]
    tareas += [(f"Tecnica {ej}", f"tecnica_{ej}", dataset_tecnica(base, ej, True)) for ej in config.EJERCICIOS]
    for nombre, slug, datos in tareas:
        Xs, y, _, _, per = datos[:5]
        X = Xs["tabular"]
        if len(set(y)) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        reporte.append(f"\n{nombre}")
        for j, (nom, mod) in enumerate(modelos_fijos().items()):
            ent, val = np.full((len(tamanos), n_rep), np.nan), np.full((len(tamanos), n_rep), np.nan)
            for i, k in enumerate(tamanos):
                for r in range(n_rep):
                    rng = np.random.default_rng(config.SEMILLA + 100 * i + r)
                    pv = rng.choice(personas, n_val, replace=False)
                    resto = [p for p in personas if p not in pv]
                    pt = rng.choice(resto, k, replace=False)
                    tr, te = np.isin(per, pt), np.isin(per, pv)
                    if len(set(y[tr])) < 2:
                        continue
                    m = clone(mod).fit(X[tr], y[tr])
                    ent[i, r] = balanced_accuracy_score(y[tr], m.predict(X[tr]))
                    val[i, r] = balanced_accuracy_score(y[te], m.predict(X[te]))
            for serie, est, nombre_serie in ((ent, "--", "entrenamiento"), (val, "-", "validación")):
                mu, de = np.nanmean(serie, 1), np.nanstd(serie, 1)
                ax.plot(tamanos, mu, est, color=f"C{j}", marker="o", ms=3, label=f"{nom} · {nombre_serie}")
                ax.fill_between(tamanos, mu - de, mu + de, color=f"C{j}", alpha=0.12)
            reporte.append(f"  {nom:13s} " + "   ".join(
                f"{k} pers.: val {np.nanmean(val[i]):.3f} (ent {np.nanmean(ent[i]):.3f})" for i, k in enumerate(tamanos)))
        ax.set(xlabel="personas de entrenamiento", ylabel="acierto balanceado", title=f"Curva de aprendizaje · {nombre}",
               ylim=(0.3, 1.02))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(dir_salidas / f"curva_aprendizaje_{slug}.png", dpi=120)
        plt.close(fig)
    reporte.append("  Lectura: si la validación sigue subiendo con más personas, faltan datos (varianza);\n"
                   "  si se estanca lejos de entrenamiento, el modelo sobreajusta; si ambas están bajas, faltan rasgos (sesgo).")


# ----------------------------------------------------------------- 3. vista diagonal
def robustez_vista(dir_datos, dir_modelos, reporte):
    reporte.append(f"\n{'=' * 78}\n3. ROBUSTEZ AL PUNTO DE VISTA (modelos entrenados con perfil)\n{'=' * 78}")
    dir_eval = dir_modelos / "evaluacion"
    ruta_p = dir_eval / "personas_prueba.json"
    prueba = json.loads(ruta_p.read_text(encoding="utf-8")) if ruta_p.exists() else []
    if not prueba or not (dir_eval / "ejercicio.joblib").exists():
        reporte.append("  Falta correr entrenar.py con personas de prueba (se necesitan los modelos de evaluacion/).")
        return
    diag = [s for s in cargar_sesiones(dir_datos, vistas=("diagonal",)) if s["persona"] in prueba]
    if not diag:
        reporte.append("  No hay CSV de vista diagonal de las personas de prueba. Bájalos con:\n"
                       "    python importar_mendeley.py --videos <videos> --vistas side,diagonal\n"
                       "    python extraer_videos.py --videos <videos> --datos <datos>")
        return
    lado = [s for s in cargar_sesiones(dir_datos, vistas=("side",)) if s["persona"] in prueba]
    for vista, ses in (("perfil (side)", lado), ("diagonal", diag)):
        reporte.append(f"\n  Vista {vista}: {len(ses)} videos")
        m = joblib.load(dir_eval / "ejercicio.joblib")
        Xs, y, _, _, _ = dataset_ejercicio(ses, True)
        if len(y):
            pred = m["modelo"].predict(Xs["tabular"])
            reporte.append(f"    Ejercicio            F1 macro = {f1_score(y, pred, average='macro'):.3f}   "
                           f"acierto bal. = {balanced_accuracy_score(y, pred):.3f}   ({len(y)} ventanas)")
        for ej in config.EJERCICIOS:
            ruta = dir_eval / f"tecnica_{ej}.joblib"
            if not ruta.exists():
                continue
            m = joblib.load(ruta)
            Xs, y, _, _, _, reps = dataset_tecnica(ses, ej, True)
            if len(y) == 0:
                reporte.append(f"    Tecnica {ej:12s} no se detectaron repeticiones")
                continue
            pred = m["modelo"].predict(Xs[m.get("entrada", "tabular")])
            sin = sum(1 for n in reps.values() if n == 0)
            reporte.append(f"    Tecnica {ej:12s} F1 macro = {f1_score(y, pred, average='macro'):.3f}   "
                           f"acierto bal. = {balanced_accuracy_score(y, pred):.3f}   ({len(y)} reps; "
                           f"{sin} videos sin reps detectadas)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=config.DIR_DATOS, type=Path)
    ap.add_argument("--modelos", default=config.DIR_MODELOS, type=Path)
    ap.add_argument("--salidas", default=config.DIR_SALIDAS, type=Path)
    ap.add_argument("--solo", default="comparacion,ablaciones,curva,vista")
    args = ap.parse_args()
    args.salidas.mkdir(parents=True, exist_ok=True)
    que = {q.strip() for q in args.solo.split(",")}
    reporte = []
    t0 = time.time()
    if que & {"comparacion", "ablaciones", "curva"}:
        print(f"Leyendo sesiones de {args.datos} ...")
        todas = cargar_sesiones(args.datos, crudo=True)
        ruta_p = args.modelos / "evaluacion" / "personas_prueba.json"
        prueba = (json.loads(ruta_p.read_text(encoding="utf-8")) if ruta_p.exists()
                  else dividir_personas({s["persona"] for s in todas}))
        base = [s for s in todas if s["persona"] not in prueba]
        reporte.append(f"Personas de entrenamiento: {len({s['persona'] for s in base})} "
                       f"(se excluyen las {len(prueba)} de prueba final)")
        if "comparacion" in que:
            comparacion(base, [s for s in todas if s["persona"] in prueba], args.salidas, reporte)
        if "ablaciones" in que:
            ablaciones(base, args.salidas, reporte)
        if "curva" in que:
            curva_aprendizaje(base, args.salidas, reporte)
    if "vista" in que:
        robustez_vista(args.datos, args.modelos, reporte)
    texto = "\n".join(reporte)
    # una corrida parcial (--solo) no pisa el reporte completo
    sufijo = "" if que >= {"comparacion", "ablaciones", "curva", "vista"} else "_" + "_".join(sorted(que))
    (args.salidas / f"reporte_experimentos{sufijo}.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nListo en {time.time() - t0:.0f} s. Reporte, ablaciones.csv y gráficos en {args.salidas}")


if __name__ == "__main__":
    main()
