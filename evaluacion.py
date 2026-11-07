"""Validación cruzada anidada y agrupada, métricas, intervalos de confianza, pruebas y gráficos."""
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, fbeta_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold

import config
import pose
from modelos import ModeloCalibrado

# avisos esperados con pocas personas por fold (alguna con una sola clase); no indican un error
warnings.filterwarnings("ignore", message="A single label was found")
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ETIQUETAS_METRICAS = {
    "f1_macro": "F1 macro", "acierto_bal": "Acierto bal.", "mcc": "MCC", "recall_mal": "Recall mal",
    "precision_mal": "Precisión mal", "f2_mal": "F2 mal", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC",
    "brier": "Brier", "accuracy": "Accuracy",
}


# ----------------------------------------------------------------- métricas
def metricas(y, pred, proba, clases, positiva=None):
    """Métricas de una predicción. Con `positiva` ("mal") agrega las de la clase crítica."""
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    m = {"f1_macro": f1_score(y, pred, labels=clases, average="macro", zero_division=0),
         "acierto_bal": balanced_accuracy_score(y, pred),
         "mcc": matthews_corrcoef(y, pred),
         "accuracy": accuracy_score(y, pred)}
    dos_clases = len(set(y)) > 1
    if positiva is not None:
        yb, pb = y == positiva, pred == positiva
        p = proba[:, clases.index(positiva)]
        m["recall_mal"] = recall_score(yb, pb, zero_division=0)
        m["precision_mal"] = precision_score(yb, pb, zero_division=0)
        m["f2_mal"] = fbeta_score(yb, pb, beta=2, zero_division=0)
        m["pr_auc"] = average_precision_score(yb, p) if dos_clases else np.nan
        m["roc_auc"] = roc_auc_score(yb, p) if dos_clases else np.nan
        m["brier"] = brier_score_loss(yb, p)
    else:
        Y = (y[:, None] == np.array(clases)[None, :]).astype(float)
        m["brier"] = float(np.mean(np.sum((Y - proba) ** 2, axis=1)))
        if len(set(y)) == len(clases) == 2:  # p. ej. dos tipos de error
            m["roc_auc"] = roc_auc_score(y == clases[1], proba[:, 1])
        elif len(set(y)) == len(clases):
            m["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr", labels=clases)
    return m


def positiva_de(clases):
    """La clase crítica es "mal" cuando la técnica es bien/mal (con errores específicos no hay una sola)."""
    return "mal" if set(clases) == {"bien", "mal"} else None


def decidir(proba, clases, umbral, positiva):
    """Misma regla que ModeloCalibrado.predict, aplicada a probabilidades ya promediadas."""
    if umbral is not None and positiva is not None and len(clases) == 2:
        ip = clases.index(positiva)
        return np.where(proba[:, ip] >= umbral, positiva, clases[1 - ip])
    return np.array(clases)[np.argmax(proba, axis=1)]


def agregar_por_video(ids, y, proba, clases, umbral=None, positiva=None):
    """La etiqueta del dataset es por video: se promedia la probabilidad de sus reps/ventanas."""
    ids, y = np.asarray(ids), np.asarray(y)
    vid = sorted(set(ids))
    pv = np.array([proba[ids == v].mean(axis=0) for v in vid])
    yv = np.array([Counter(y[ids == v]).most_common(1)[0][0] for v in vid])
    return np.array(vid), yv, pv, decidir(pv, clases, umbral, positiva)


def acierto_pareado(ids, per, y, proba, clases, positiva):
    """Para cada persona con un video bien y uno mal: ¿su video mal tiene mayor p(mal) promedio?
    Mide si el modelo distingue la técnica DENTRO de una persona, aunque no acierte el nivel
    absoluto (las personas difieren mucho entre sí). 0.5 = azar. Devuelve (acierto, n personas)."""
    if positiva is None:
        return np.nan, 0
    ids, per, y = np.asarray(ids), np.asarray(per), np.asarray(y)
    p = proba[:, clases.index(positiva)]
    aciertos = []
    for persona in sorted(set(per)):
        m = per == persona
        pv = {v: p[m & (ids == v)].mean() for v in set(ids[m])}
        etq = {v: y[m & (ids == v)][0] for v in pv}
        malos = [pv[v] for v in pv if etq[v] == positiva]
        buenos = [pv[v] for v in pv if etq[v] != positiva]
        if malos and buenos:
            aciertos.append(np.mean([1.0 if a > b else 0.5 if a == b else 0.0 for a in malos for b in buenos]))
    return (float(np.mean(aciertos)), len(aciertos)) if aciertos else (np.nan, 0)


def por_persona(y, pred, g):
    """Acierto balanceado de cada persona (vector ordenado por persona)."""
    personas = sorted(set(g))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # personas con una sola clase en sus muestras
        return personas, np.array([balanced_accuracy_score(y[g == p], pred[g == p]) for p in personas])


def ic_bootstrap(y, pred, g, clases, n=1000, semilla=config.SEMILLA):
    """IC 95 % del F1 macro remuestreando PERSONAS con reemplazo (las muestras de una persona
    no son independientes, así que remuestrear muestras daría un intervalo demasiado angosto)."""
    rng = np.random.default_rng(semilla)
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    personas = np.array(sorted(set(g)))
    idx = {p: np.where(g == p)[0] for p in personas}
    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n):
            sel = np.concatenate([idx[p] for p in rng.choice(personas, len(personas), replace=True)])
            if len(set(y[sel])) < 2:
                continue
            vals.append(f1_score(y[sel], pred[sel], labels=clases, average="macro", zero_division=0))
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))) if vals else (np.nan, np.nan)


def wilcoxon_holm(puntajes, mejor):
    """Wilcoxon pareado (por persona) del mejor contra cada otro modelo, con corrección de Holm."""
    otros = [n for n in puntajes if n != mejor]
    crudos = []
    for n in otros:
        d = puntajes[mejor] - puntajes[n]
        if np.allclose(d, 0):
            crudos.append(1.0)
        else:
            crudos.append(float(wilcoxon(puntajes[mejor], puntajes[n], zero_method="zsplit").pvalue))
    orden = np.argsort(crudos)
    ajust, acumulado = [0.0] * len(otros), 0.0
    for r, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, crudos[i] * (len(otros) - r)))
        ajust[i] = acumulado
    return {n: (crudos[i], ajust[i]) for i, n in enumerate(otros)}


# ----------------------------------------------------------------- validación cruzada anidada
def validar(cand, X, y, g, positiva, n_rep=config.REPETICIONES_CV, n_ext=config.FOLDS_EXTERNOS,
            n_int=config.FOLDS_INTERNOS, criterio=config.CRITERIO_UMBRAL, semilla=config.SEMILLA,
            guardar_modelos=False):
    """CV externa agrupada y estratificada, repetida `n_rep` veces con distinta partición.

    En cada fold externo, ModeloCalibrado hace su propia CV interna (hiperparámetros, calibración
    y umbral) solo con las personas de entrenamiento: las del fold externo nunca influyen en nada.
    Devuelve las predicciones fuera de fold y las métricas de cada fold.
    """
    clases = sorted(set(y))
    k = min(n_ext, len(set(g)))
    reps = []
    for r in range(n_rep):
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=semilla + r)
        proba = np.zeros((len(y), len(clases)))
        pred = np.empty(len(y), dtype=object)
        folds, params, umbrales, modelos, tes = [], [], [], [], []
        for tr, te in cv.split(X, y, g):
            tes.append(te)
            mod = ModeloCalibrado(clone(cand["base"]), cand["grid"], positiva, criterio, n_int, semilla)
            mod.fit(X[tr], y[tr], groups=g[tr])
            pp = mod.predict_proba(X[te])
            for j, c in enumerate(mod.classes_):
                proba[te, clases.index(c)] = pp[:, j]
            pred[te] = mod.predict(X[te])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                folds.append(metricas(y[te], pred[te], proba[te], clases, positiva))
            params.append(mod.mejores_params_)
            umbrales.append(mod.umbral_)
            if guardar_modelos and r == 0:
                modelos.append((mod, te))
        reps.append({"proba": proba, "pred": pred.astype(str), "folds": folds, "params": params,
                     "umbrales": umbrales, "modelos": modelos, "te": tes})
    return {"clases": clases, "reps": reps}


def resumir(res, y, g, ids, positiva):
    """Media ± DE sobre todos los folds (de todas las repeticiones), métricas agrupadas por video,
    acierto por persona (promedio entre repeticiones) e IC por bootstrap de personas."""
    clases = res["clases"]
    folds = [f for r in res["reps"] for f in r["folds"]]
    claves = list(folds[0])
    s = {"media": {c: float(np.nanmean([f[c] for f in folds])) for c in claves},
         "de": {c: float(np.nanstd([f[c] for f in folds])) for c in claves}}
    r0 = res["reps"][0]
    s["ic_f1"] = ic_bootstrap(y, r0["pred"], g, clases)
    umbrales = [u for r in res["reps"] for u in r["umbrales"] if u is not None]
    s["umbral"] = float(np.median(umbrales)) if umbrales else None
    vid, yv, pv, predv = agregar_por_video(ids, y, r0["proba"], clases, s["umbral"], positiva)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s["video"] = metricas(yv, predv, pv, clases, positiva)
    s["pareado"] = float(np.nanmean([acierto_pareado(ids, g, y, r["proba"], clases, positiva)[0]
                                     for r in res["reps"]])) if positiva else np.nan
    personas, pp = zip(*[por_persona(y, r["pred"], g) for r in res["reps"]])
    s["personas"], s["por_persona"] = personas[0], np.mean(pp, axis=0)
    s["params"] = Counter(str(p) for r in res["reps"] for p in r["params"]).most_common(1)[0][0]
    return s


def cobertura(proba, y, clases, prob_min=config.PROB_MIN_EJERCICIO):
    """Para el modelo de ejercicio en vivo: con qué fracción de ventanas se anima a decidir
    (prob. máxima >= prob_min) y qué tan bien acierta en esas."""
    conf = proba.max(axis=1)
    ok = conf >= prob_min
    pred = np.array(clases)[proba.argmax(axis=1)]
    return float(ok.mean()), float((pred[ok] == np.asarray(y)[ok]).mean()) if ok.any() else np.nan


# ----------------------------------------------------------------- importancia
def grupos_rasgos(nombres):
    """Asigna cada columna a la señal de la que sale (o a 'fase'), para permutar por grupo."""
    por_largo = sorted(pose.SENALES, key=len, reverse=True)  # "rodilla_pie" antes que "rodilla"
    grupos = []
    for n in nombres:
        if n.startswith("fase_") or n == "bas_duracion_s":
            grupos.append("fase (tiempos/velocidad)")
            continue
        base = n[4:] if n.startswith("bas_") else n[5:] if n.startswith("flex_") else n
        grupos.append(next((s for s in por_largo if base == s or base.startswith(s + "_")), "otro"))
    return np.array(grupos)


def importancia_permutacion(modelos, X, y, nombres, n_repeticiones=3, semilla=config.SEMILLA):
    """Caída del acierto balanceado al permutar cada grupo de rasgos (y cada rasgo) en los folds
    de prueba de la CV externa. A diferencia de la importancia MDI del Random Forest, se mide en
    datos no vistos y no se reparte entre rasgos redundantes."""
    rng = np.random.default_rng(semilla)
    grupos = grupos_rasgos(nombres)
    unidades = {f"[{gr}]": np.where(grupos == gr)[0] for gr in sorted(set(grupos))}
    if X.shape[1] <= 80:  # rasgos individuales (no para la secuencia de ROCKET)
        unidades.update({n: np.array([i]) for i, n in enumerate(nombres)})
    caidas = {u: [] for u in unidades}
    for mod, te in modelos:
        Xte, yte = X[te], y[te]
        base = balanced_accuracy_score(yte, mod.predict(Xte))
        for u, cols in unidades.items():
            for _ in range(n_repeticiones):
                Xp = Xte.copy()
                perm = rng.permutation(len(Xte))
                Xp[:, cols] = Xte[perm][:, cols]
                caidas[u].append(base - balanced_accuracy_score(yte, mod.predict(Xp)))
    return {u: (float(np.mean(v)), float(np.std(v))) for u, v in caidas.items()}


# ----------------------------------------------------------------- gráficos
def _guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def grafico_confusion(y, pred, clases, titulo, ruta):
    """Matriz normalizada por fila (recall de cada clase) con el conteo entre paréntesis."""
    cm = confusion_matrix(y, pred, labels=clases)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(clases) + 2.5, 1.4 * len(clases) + 2))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(clases)), clases, rotation=30, ha="right")
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    _guardar(fig, ruta)


def grafico_pr_roc(curvas, positiva, titulo, ruta):
    """curvas: {etiqueta: (y, p_positiva)}. Curva Precisión-Recall (la importante con la clase
    crítica) y ROC, con la regla de decisión marcada si se pasa (y, p, umbral)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for et, val in curvas.items():
        y, p = val[0], val[1]
        yb = np.asarray(y) == positiva
        if yb.all() or not yb.any():
            continue
        pr, rc, _ = precision_recall_curve(yb, p)
        fpr, tpr, _ = roc_curve(yb, p)
        a1.plot(rc, pr, label=f"{et} (AP={average_precision_score(yb, p):.2f})", lw=1.5)
        a2.plot(fpr, tpr, label=f"{et} (AUC={roc_auc_score(yb, p):.2f})", lw=1.5)
        if len(val) > 2 and val[2] is not None:
            pb = p >= val[2]
            a1.plot(recall_score(yb, pb), precision_score(yb, pb, zero_division=0), "ko")
    prev = np.mean(np.asarray(next(iter(curvas.values()))[0]) == positiva)
    a1.axhline(prev, ls="--", c="gray", lw=1, label=f"azar ({prev:.2f})")
    a1.set(xlabel=f"Recall ({positiva})", ylabel=f"Precisión ({positiva})", title="Precisión-Recall", ylim=(0, 1.02))
    a2.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    a2.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title="ROC")
    for a in (a1, a2):
        a.legend(fontsize=7, loc="lower left" if a is a1 else "lower right")
    fig.suptitle(titulo)
    _guardar(fig, ruta)


def grafico_confiabilidad(y, proba, clases, positiva, titulo, ruta):
    """Diagrama de confiabilidad: si el modelo dice 80 %, ¿acierta el 80 % de las veces?"""
    fig, ax = plt.subplots(figsize=(5, 5))
    if positiva is not None:
        yb, p = np.asarray(y) == positiva, proba[:, clases.index(positiva)]
        etiqueta = f"p({positiva})"
    else:
        p = proba.max(axis=1)
        yb = np.array(clases)[proba.argmax(axis=1)] == np.asarray(y)
        etiqueta = "confianza (prob. máxima)"
    fr, pm = calibration_curve(yb, p, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "--", c="gray", lw=1, label="calibración perfecta")
    ax.plot(pm, fr, "o-", label=f"modelo (Brier={np.mean((yb - p) ** 2):.3f})")
    ax.set(xlabel=etiqueta, ylabel="frecuencia observada", title=titulo, xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
    _guardar(fig, ruta)


def grafico_personas(puntajes, titulo, ruta):
    """Boxplot del acierto balanceado por persona: muestra para quién falla cada modelo."""
    nombres = list(puntajes)
    fig, ax = plt.subplots(figsize=(1.1 * len(nombres) + 3, 4.5))
    ax.boxplot([puntajes[n] for n in nombres], showmeans=True)
    for i, n in enumerate(nombres, 1):
        ax.scatter(np.random.default_rng(i).normal(i, 0.05, len(puntajes[n])), puntajes[n], s=10, alpha=0.5)
    ax.set_xticks(range(1, len(nombres) + 1), nombres, rotation=30, ha="right")
    ax.set(ylabel="Acierto balanceado por persona", title=titulo, ylim=(0, 1.05))
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    _guardar(fig, ruta)


def grafico_importancia(imp, titulo, ruta, top=15):
    grupos = {k: v for k, v in imp.items() if k.startswith("[")}
    indiv = sorted(((k, v) for k, v in imp.items() if not k.startswith("[")), key=lambda kv: -kv[1][0])[:top]
    paneles = [("por señal", sorted(grupos.items(), key=lambda kv: kv[1][0]))]
    if indiv:
        paneles.append((f"top {top} rasgos", indiv[::-1]))
    fig, axs = plt.subplots(1, len(paneles), figsize=(6 * len(paneles), 0.32 * max(len(p[1]) for p in paneles) + 1.8))
    for ax, (sub, items) in zip(np.atleast_1d(axs), paneles):
        ax.barh([k for k, _ in items], [v[0] for _, v in items], xerr=[v[1] for _, v in items], color="tab:blue")
        ax.axvline(0, c="gray", lw=1)
        ax.set(xlabel="caída del acierto balanceado al permutar", title=sub)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(titulo)
    _guardar(fig, ruta)


def tabla(resumenes, claves, mejor):
    """Tabla de texto: media ± DE entre folds por modelo."""
    enc = f"  {'modelo':14s}" + "".join(f"{ETIQUETAS_METRICAS.get(c, c):>15s}" for c in claves)
    filas = [enc]
    for n, s in resumenes.items():
        celdas = "".join(f"{s['media'].get(c, np.nan):>8.3f}±{s['de'].get(c, np.nan):<6.3f}" for c in claves)
        filas.append(f"{'*' if n == mejor else ' '} {n:14s}{celdas}")
    return filas
