"""Entrena y evalúa los modelos a partir de los CSV de pose.

1. Modelo de ejercicio: clasifica ventanas de ~1 s (sentadilla / lagartija / curl).
2. Modelo de técnica (uno por ejercicio): clasifica cada repetición (bien / tipo de error).

Para cada problema compara Random Forest, SVM (RBF) y MLP con validación cruzada agrupada
(las ventanas o reps de un mismo video, o de una misma persona si hay 3 o más, nunca quedan
a la vez en entrenamiento y prueba) y guarda el mejor según F1 macro.

Uso:
    python entrenar.py --datos /ruta/datos --modelos /ruta/modelos --salidas /ruta/salidas
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
import pose

CANDIDATOS = {
    "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1),
    "SVM_RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                   class_weight="balanced", probability=True, random_state=0)),
    # early_stopping=False: en scikit-learn 1.8 falla con etiquetas de texto
    "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                         max_iter=2000, random_state=0)),
}


def cargar_sesiones(dir_datos):
    sesiones = []
    for csv in sorted(dir_datos.rglob("*.csv")):
        partes = csv.stem.split("__")
        if len(partes) < 2 or partes[0] not in config.EJERCICIOS \
                or partes[1] not in config.EJERCICIOS[partes[0]]["etiquetas"]:
            print(f"  (se ignora {csv.name}: nombre no válido)")
            continue
        t, A = pose.leer_sesion(csv)
        if len(A) < config.VENTANA:
            print(f"  (se ignora {csv.name}: muy pocos frames con pose)")
            continue
        sesiones.append({"persona": csv.parent.name, "id": f"{csv.parent.name}/{csv.stem}",
                         "ejercicio": partes[0], "etiqueta": partes[1], "t": t, "A": A})
    return sesiones


def dataset_ejercicio(sesiones, por_persona):
    X, y, g = [], [], []
    for s in sesiones:
        A = s["A"]
        for i in range(0, len(A) - config.VENTANA + 1, config.PASO_VENTANA):
            W = A[i:i + config.VENTANA]
            if not pose.hay_movimiento(W):
                continue
            X.append(pose.rasgos_ventana(W))
            y.append(s["ejercicio"])
            g.append(s["persona"] if por_persona else s["id"])
    return np.array(X), np.array(y), np.array(g)


def dataset_tecnica(sesiones, ej, por_persona):
    X, y, g, reps_por_sesion = [], [], [], {}
    for s in sesiones:
        if s["ejercicio"] != ej:
            continue
        cont = pose.ContadorReps(config.EJERCICIOS[ej]["angulo"])
        n = 0
        for a, t in zip(s["A"], s["t"]):
            rep = cont.actualizar(a, t)
            if rep is not None:
                X.append(pose.rasgos_rep(*rep))
                y.append(s["etiqueta"])
                g.append(s["persona"] if por_persona else s["id"])
                n += 1
        reps_por_sesion[s["id"]] = n
    return np.array(X), np.array(y), np.array(g), reps_por_sesion


def evaluar(nombre, X, y, g, nombres_rasgos, dir_salidas, reporte):
    """Compara los candidatos con CV agrupada, guarda la matriz de confusión del mejor y lo devuelve entrenado."""
    clases = sorted(set(y))
    reporte.append(f"\n{'=' * 70}\n{nombre}\n{'=' * 70}")
    reporte.append(f"Muestras por clase: {dict(sorted(Counter(map(str, y)).items()))}   Grupos: {len(set(g))}")
    if len(clases) < 2:
        reporte.append("Solo hay una clase: no se puede entrenar un clasificador.")
        return None
    n_grupos = len(set(g))
    mejor, mejor_f1, mejor_pred = "RandomForest", -1.0, None
    if n_grupos >= 2:
        cv = GroupKFold(n_splits=min(5, n_grupos))
        reporte.append(f"Validación cruzada agrupada con {cv.get_n_splits()} folds:")
        for nom, mod in CANDIDATOS.items():
            try:
                pred = cross_val_predict(clone(mod), X, y, groups=g, cv=cv)
            except Exception as e:  # noqa: BLE001 - un candidato que falla no detiene a los demás
                reporte.append(f"  {nom:13s} no se pudo evaluar ({type(e).__name__}: {e})")
                continue
            f1 = f1_score(y, pred, average="macro")
            reporte.append(f"  {nom:13s} accuracy = {accuracy_score(y, pred):.3f}   F1 macro = {f1:.3f}")
            if f1 > mejor_f1:
                mejor, mejor_f1, mejor_pred = nom, f1, pred
    else:
        reporte.append("Solo hay 1 video/persona: no se puede validar. Graba más videos para tener métricas.")
    if mejor_pred is not None:
        reporte.append(f"\nMejor modelo: {mejor}\n" + classification_report(y, mejor_pred, zero_division=0))
        cm = confusion_matrix(y, mejor_pred, labels=clases)
        fig, ax = plt.subplots(figsize=(1.6 * len(clases) + 2.5, 1.4 * len(clases) + 2))
        ConfusionMatrixDisplay(cm, display_labels=clases).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{nombre}\n{mejor} · F1 macro = {mejor_f1:.3f}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(dir_salidas / f"confusion_{nombre.split(':')[0].strip().lower().replace(' ', '_')}.png", dpi=120)
        plt.close(fig)
    modelo = clone(CANDIDATOS[mejor]).fit(X, y)
    if mejor == "RandomForest":
        imp = sorted(zip(modelo.feature_importances_, nombres_rasgos), reverse=True)[:8]
        reporte.append("Rasgos más importantes (Random Forest):")
        reporte += [f"  {n:22s} {v:.3f}" for v, n in imp]
    return {"modelo": modelo, "clases": list(modelo.classes_), "rasgos": nombres_rasgos, "algoritmo": mejor,
            "f1_cv": mejor_f1}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=config.DIR_DATOS, type=Path)
    ap.add_argument("--modelos", default=config.DIR_MODELOS, type=Path)
    ap.add_argument("--salidas", default=config.DIR_SALIDAS, type=Path)
    args = ap.parse_args()
    args.modelos.mkdir(parents=True, exist_ok=True)
    args.salidas.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo sesiones de {args.datos} ...")
    sesiones = cargar_sesiones(args.datos)
    if not sesiones:
        sys.exit("No hay CSV válidos. Corre primero extraer_videos.py.")
    personas = sorted({s["persona"] for s in sesiones})
    por_persona = len(personas) >= 3
    reporte = [f"Sesiones (videos): {len(sesiones)}   Personas: {len(personas)} ({', '.join(personas)})",
               "Grupos de validación: " + ("persona (mide qué tan bien generaliza a gente nueva)"
                                           if por_persona else "video (con 3+ personas se agrupa por persona)")]
    tabla = Counter((s["ejercicio"], s["etiqueta"]) for s in sesiones)
    reporte.append("Videos por ejercicio y etiqueta:")
    reporte += [f"  {e:11s} {et:18s} {n}" for (e, et), n in sorted(tabla.items())]

    X, y, g = dataset_ejercicio(sesiones, por_persona)
    m = evaluar("Ejercicio", X, y, g, pose.NOMBRES_RASGOS_VENTANA, args.salidas, reporte)
    if m is None:
        sys.exit("\n".join(reporte) + "\n\nSe necesitan videos de al menos 2 ejercicios.")
    joblib.dump(m, args.modelos / "ejercicio.joblib")

    for ej in config.EJERCICIOS:
        X, y, g, reps = dataset_tecnica(sesiones, ej, por_persona)
        sin_reps = [k for k, n in reps.items() if n == 0]
        if sin_reps:
            reporte.append(f"\n[!] {ej}: no se detectaron repeticiones en {', '.join(sin_reps)}")
        if len(X) == 0:
            reporte.append(f"\n{ej}: sin repeticiones, no se entrena modelo de técnica.")
            continue
        mt = evaluar(f"Tecnica {ej}", X, y, g, pose.NOMBRES_RASGOS_REP, args.salidas, reporte)
        if mt is not None:
            joblib.dump(mt, args.modelos / f"tecnica_{ej}.joblib")

    texto = "\n".join(reporte)
    (args.salidas / "reporte_entrenamiento.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nModelos en {args.modelos}\nReporte y matrices de confusión en {args.salidas}")


if __name__ == "__main__":
    main()
