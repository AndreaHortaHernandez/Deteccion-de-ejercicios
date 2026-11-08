"""Entrena y evalúa los modelos a partir de los CSV de pose.

1. Modelo de ejercicio: clasifica ventanas de 2 s (sentadilla / lagartija / curl).
2. Modelo de técnica (uno por ejercicio): clasifica cada repetición (bien / mal).

Protocolo de evaluación:
- Antes de todo se aparta ~20 % de las personas como PRUEBA FINAL. No se usan para elegir nada
  y se evalúan una sola vez, al final.
- Con el resto, cada candidato (desde Dummy y una regla de un umbral hasta ensambles y ROCKET)
  se evalúa con validación cruzada ANIDADA y agrupada por persona: la CV externa
  (StratifiedGroupKFold repetida) estima el desempeño; la interna elige hiperparámetros, calibra
  las probabilidades y fija el umbral de p(mal). Se reporta media ± DE entre folds, IC 95 % por
  bootstrap de personas y métricas por video (la etiqueta del dataset es por video).
- Gana el de mejor F1 macro medio; se compara con los demás con Wilcoxon pareado por persona.
- El ganador se entrena con las personas de entrenamiento y se mide en la prueba final. Ese
  modelo se guarda en <modelos>/evaluacion/ (lo usan evaluar_sistema.py y experimentos.py).
- El modelo para usar en vivo se reentrena con todas las personas y se guarda en <modelos>/.

Uso:
    python entrenar.py --datos /ruta/datos --modelos /ruta/modelos --salidas /ruta/salidas
                       [--rapido] [--candidatos LogReg,RandomForest] [--sin-prueba]
"""
import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

import config
import evaluacion as ev
import pose
from modelos import ModeloCalibrado, candidatos, describir_regla

VISTAS_MENDELEY = {"side", "diagonal", "front"}


# ----------------------------------------------------------------- datos
def cargar_sesiones(dir_datos, vistas=config.VISTAS_ENTRENAR, crudo=False):
    """Lee los CSV de pose. Nombre: <persona>/<ejercicio>__<etiqueta>__<n o vista>.csv, con etiqueta
    bien, mal o un error específico de config.EJERCICIOS[...]["errores"] (cuenta como mal).
    Los videos propios (sin vista en el nombre) cuentan como vista de perfil (side)."""
    sesiones = []
    for ruta in sorted(Path(dir_datos).rglob("*.csv")):
        partes = ruta.stem.lower().split("__")
        clase = config.clasificar_etiqueta(partes[0], partes[1]) if len(partes) >= 2 else None
        if clase is None:
            continue  # conteos_reales.csv u otros archivos
        vista = partes[2] if len(partes) > 2 and partes[2] in VISTAS_MENDELEY else "side"
        if vistas and vista not in vistas:
            continue
        lm = pose.leer_landmarks(ruta)
        t, A = pose.senales_sesion(*lm)
        if len(A) < config.MIN_FRAMES_VENTANA or t[-1] - t[0] < config.VENTANA_S:
            print(f"  (se ignora {ruta.name}: muy pocos cuadros con pose)")
            continue
        # etiqueta = bien/mal (modelo de técnica); error = error específico si se grabó uno (o None)
        s = {"persona": ruta.parent.name, "id": f"{ruta.parent.name}/{ruta.stem}", "vista": vista,
             "ejercicio": partes[0], "etiqueta": clase[0], "error": clase[1], "t": t, "A": A}
        if crudo:
            s["crudo"] = lm
        sesiones.append(s)
    return sesiones


def dividir_personas(personas, semilla=config.SEMILLA):
    """Personas de prueba final (determinista: siempre las mismas para el mismo conjunto)."""
    personas = sorted(personas)
    if len(personas) < config.MIN_PERSONAS_TEST:
        return []
    n = max(2, round(config.FRACCION_TEST * len(personas)))
    return sorted(np.random.default_rng(semilla).choice(personas, n, replace=False).tolist())


def dataset_ejercicio(sesiones, por_persona):
    X, y, g, ids, per = [], [], [], [], []
    for s in sesiones:
        for W in pose.ventanas(s["t"], s["A"], paso=config.PASO_ENTRENAR_S):
            if not pose.hay_movimiento(W):
                continue
            X.append(pose.rasgos_ventana(W))
            y.append(s["ejercicio"])
            g.append(s["persona"] if por_persona else s["id"])
            ids.append(s["id"])
            per.append(s["persona"])
    return {"tabular": np.array(X)}, np.array(y), np.array(g), np.array(ids), np.array(per)


def dataset_tecnica(sesiones, ej, por_persona, errores=False):
    """Reps de un ejercicio. Con errores=True, solo las de videos con un error específico grabado
    y la clase es ese error (para el modelo que dice QUÉ error es)."""
    k = pose.SENALES.index(config.EJERCICIOS[ej]["angulo"])
    Xt, Xs, y, g, ids, per, reps_por_sesion = [], [], [], [], [], [], {}
    for s in sesiones:
        if s["ejercicio"] != ej or (errores and not s.get("error")):
            continue
        reps = pose.contar_reps(s["t"], s["A"], ej)
        for A, t in reps:
            Xt.append(pose.rasgos_rep(A, t, k))
            Xs.append(pose.secuencia_rep(A, t))
            y.append(s["error"] if errores else s["etiqueta"])
            g.append(s["persona"] if por_persona else s["id"])
            ids.append(s["id"])
            per.append(s["persona"])
        reps_por_sesion[s["id"]] = len(reps)
    return ({"tabular": np.array(Xt), "secuencia": np.array(Xs)}, np.array(y), np.array(g),
            np.array(ids), np.array(per), reps_por_sesion)


# ----------------------------------------------------------------- evaluación de una tarea
def _fmt(m, claves):
    return "   ".join(f"{ev.ETIQUETAS_METRICAS.get(c, c)} = {m.get(c, np.nan):.3f}" for c in claves)


def _paquete(mod, nombres, entrada, algoritmo, f1_cv, metricas_test=None):
    return {"modelo": mod, "clases": list(mod.classes_), "rasgos": nombres, "entrada": entrada,
            "algoritmo": algoritmo, "f1_cv": f1_cv, "umbral_mal": mod.umbral_,
            "metricas_test": metricas_test,
            "config": {k: getattr(config, k) for k in ("LADO_FIJO", "MEDIANA_FRAMES", "ESTADISTICOS",
                                                       "RASGOS_REP", "VENTANA_S", "PUNTOS_SECUENCIA")}}


def evaluar_tarea(nombre, slug, Xs, nombres, y, g, ids, per, prueba, cands, dir_salidas, reporte, n_rep):
    """Compara candidatos con CV anidada en las personas de entrenamiento, mide al ganador en la
    prueba final y devuelve (modelo final con todos, modelo solo con entrenamiento, filas CSV)."""
    clases = sorted(set(y))
    positiva = ev.positiva_de(clases)
    reporte.append(f"\n{'=' * 78}\n{nombre}\n{'=' * 78}")
    reporte.append(f"Muestras por clase: {dict(sorted(Counter(map(str, y)).items()))}   "
                   f"Videos: {len(set(ids))}   Grupos: {len(set(g))}")
    if len(clases) < 2:
        reporte.append("Solo hay una clase: no se puede entrenar un clasificador.")
        return None, None, []
    en_prueba = np.isin(per, prueba)
    tr = ~en_prueba
    if len(set(g[tr])) < 2:
        reporte.append("Solo hay 1 video/persona: no se puede validar. Graba más videos para tener métricas.")
        c = cands.get("RandomForest") or next(iter(cands.values()))
        X = Xs[c["entrada"]]
        mod = ModeloCalibrado(c["base"], {}, positiva, config.CRITERIO_UMBRAL).fit(X, y, groups=g)
        return _paquete(mod, nombres[c["entrada"]], c["entrada"], "RandomForest", None), None, []

    claves = ["f1_macro", "acierto_bal", "mcc"] + (
        ["recall_mal", "precision_mal", "pr_auc", "brier"] if positiva else ["roc_auc", "brier"])
    reporte.append(f"CV anidada: {min(config.FOLDS_EXTERNOS, len(set(g[tr])))} folds externos x {n_rep} "
                   f"repeticiones, {config.FOLDS_INTERNOS} internos. Personas de entrenamiento: {len(set(per[tr]))}")
    res, resu = {}, {}
    for nom, c in cands.items():
        X = Xs[c["entrada"]]
        t0 = time.time()
        res[nom] = ev.validar(c, X[tr], y[tr], g[tr], positiva, n_rep=n_rep, guardar_modelos=True)
        resu[nom] = ev.resumir(res[nom], y[tr], g[tr], ids[tr], positiva)
        print(f"  {nombre:22s} {nom:13s} F1 macro = {resu[nom]['media']['f1_macro']:.3f} "
              f"± {resu[nom]['de']['f1_macro']:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    mejor = max(resu, key=lambda n: resu[n]["media"]["f1_macro"])
    s = resu[mejor]

    reporte.append("\nMedia ± DE entre folds (ventanas/reps de personas no vistas):")
    reporte += ev.tabla(resu, claves, mejor)
    reporte.append("\nPor video (se promedia la probabilidad de sus ventanas/reps; la etiqueta del dataset es por video)"
                   + (" y pareado por persona (¿su video mal tiene más p(mal) que su video bien?):" if positiva else ":"))
    reporte += [f"  {n:14s} F1 macro = {r['video']['f1_macro']:.3f}   acierto bal. = {r['video']['acierto_bal']:.3f}"
                + (f"   pareado = {r['pareado']:.3f}" if positiva else "") for n, r in resu.items()]
    reporte.append(f"\nMejor modelo: {mejor}   F1 macro = {s['media']['f1_macro']:.3f} "
                   f"(IC 95 % por personas: {s['ic_f1'][0]:.3f}-{s['ic_f1'][1]:.3f})")
    reporte.append(f"Hiperparámetros más elegidos: {s['params']}")
    if s["umbral"] is not None:
        reporte.append(f"Umbral de p(mal) (mediana entre folds): {s['umbral']:.3f} "
                       f"(criterio: {config.CRITERIO_UMBRAL})")
    elif positiva:
        reporte.append("Decisión: la del propio modelo con clases balanceadas (sin umbral ajustado; ver config.CRITERIO_UMBRAL)")
    if positiva is None and slug == "ejercicio":
        cob, acc = ev.cobertura(res[mejor]["reps"][0]["proba"], y[tr], clases)
        reporte.append(f"En vivo (prob. >= {config.PROB_MIN_EJERCICIO}): decide en {cob:.1%} de las ventanas "
                       f"y acierta {acc:.1%} de esas")
    wil = ev.wilcoxon_holm({n: r["por_persona"] for n, r in resu.items()}, mejor)
    reporte.append(f"\nWilcoxon pareado por persona ({mejor} vs. cada uno; p ajustado por Holm):")
    reporte += [f"  vs {n:14s} p = {p:.4f}   p Holm = {ph:.4f}{'   (diferencia significativa)' if ph < 0.05 else ''}"
                for n, (p, ph) in wil.items()]
    if "Regla_umbral" in cands:
        X = Xs["tabular"]
        reporte.append("\nLínea base 'Regla_umbral' entrenada con las personas de entrenamiento:")
        reporte += ["  " + l for l in describir_regla(cands["Regla_umbral"]["base"], X[tr], y[tr],
                                                    nombres["tabular"]).splitlines()]

    # gráficos de la CV
    r0 = res[mejor]["reps"][0]
    ev.grafico_confusion(y[tr], r0["pred"], clases, f"{nombre} · {mejor} (CV)\nnormalizada por fila",
                         dir_salidas / f"confusion_{slug}_cv.png")
    ev.grafico_confiabilidad(y[tr], r0["proba"], clases, positiva, f"{nombre} · {mejor}",
                             dir_salidas / f"confiabilidad_{slug}.png")
    ev.grafico_personas({n: r["por_persona"] for n, r in resu.items()}, f"{nombre}: acierto por persona (CV)",
                        dir_salidas / f"personas_{slug}.png")
    if positiva:
        ip = clases.index(positiva)
        curvas = {n: (y[tr], res[n]["reps"][0]["proba"][:, ip], s["umbral"] if n == mejor else None) for n in res}
        ev.grafico_pr_roc(curvas, positiva, f"{nombre} (CV, todos los modelos)", dir_salidas / f"pr_roc_{slug}.png")
        por_fold = {f"fold {i + 1}": (y[tr][te], r0["proba"][te, ip]) for i, te in enumerate(r0["te"])}
        ev.grafico_pr_roc(por_fold, positiva, f"{nombre} · {mejor} por fold", dir_salidas / f"pr_roc_folds_{slug}.png")
    c = cands[mejor]
    X = Xs[c["entrada"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imp = ev.importancia_permutacion(r0["modelos"], X[tr], y[tr], nombres[c["entrada"]])
    ev.grafico_importancia(imp, f"{nombre} · {mejor}: importancia por permutación (CV)",
                           dir_salidas / f"importancia_{slug}.png")
    top = sorted(((k, v) for k, v in imp.items() if k.startswith("[")), key=lambda kv: -kv[1][0])
    reporte.append("\nImportancia por permutación (caída del acierto bal.) por señal:")
    reporte += [f"  {k:30s} {m:+.3f} ± {d:.3f}" for k, (m, d) in top]

    # prueba final
    mod_tr = ModeloCalibrado(c["base"], c["grid"], positiva, config.CRITERIO_UMBRAL,
                             config.FOLDS_INTERNOS, config.SEMILLA).fit(X[tr], y[tr], groups=g[tr])
    m_test = None
    if en_prueba.any():
        te = en_prueba
        p, pred = mod_tr.predict_proba(X[te]), mod_tr.predict(X[te])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m_test = ev.metricas(y[te], pred, p, clases, positiva)
            ic = ev.ic_bootstrap(y[te], pred, per[te], clases)
            _, yv, pv, predv = ev.agregar_por_video(ids[te], y[te], p, clases, mod_tr.umbral_, positiva)
            m_video = ev.metricas(yv, predv, pv, clases, positiva)
            m_video["pareado"], n_par = ev.acierto_pareado(ids[te], per[te], y[te], p, clases, positiva)
        m_test["video_f1_macro"] = m_video["f1_macro"]
        reporte.append(f"\nPRUEBA FINAL ({len(set(per[te]))} personas nunca vistas, {te.sum()} muestras, {len(yv)} videos):")
        reporte.append("  " + _fmt(m_test, claves))
        reporte.append(f"  IC 95 % del F1 macro (bootstrap de personas): {ic[0]:.3f}-{ic[1]:.3f}")
        reporte.append("  Por video: " + _fmt(m_video, ["f1_macro", "acierto_bal"] + (["recall_mal"] if positiva else []))
                       + (f"   pareado = {m_video['pareado']:.3f} ({n_par} personas)" if positiva else ""))
        if slug == "ejercicio":
            cob, acc = ev.cobertura(p, y[te], clases)
            reporte.append(f"  En vivo: decide en {cob:.1%} de las ventanas y acierta {acc:.1%} de esas")
        ev.grafico_confusion(y[te], pred, clases, f"{nombre} · {mejor} (prueba final)\nnormalizada por fila",
                             dir_salidas / f"confusion_{slug}_prueba.png")

    mod_final = ModeloCalibrado(c["base"], c["grid"], positiva, config.CRITERIO_UMBRAL,
                                config.FOLDS_INTERNOS, config.SEMILLA).fit(X, y, groups=g)
    filas = [{"tarea": nombre, "modelo": n, "elegido": n == mejor,
              **{f"{k}_media": r["media"].get(k) for k in claves},
              **{f"{k}_de": r["de"].get(k) for k in claves},
              "video_f1_macro": r["video"]["f1_macro"], "video_acierto_bal": r["video"]["acierto_bal"],
              "pareado": r.get("pareado"),
              "wilcoxon_p_holm": wil.get(n, (None, None))[1]} for n, r in resu.items()]
    if m_test:
        filas.append({"tarea": nombre, "modelo": f"{mejor} (prueba final)", "elegido": True,
                      **{f"{k}_media": m_test.get(k) for k in claves}, "video_f1_macro": m_test["video_f1_macro"]})
    return (_paquete(mod_final, nombres[c["entrada"]], c["entrada"], mejor, s["media"]["f1_macro"], m_test),
            _paquete(mod_tr, nombres[c["entrada"]], c["entrada"], mejor, s["media"]["f1_macro"], m_test)
            if m_test else None, filas)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", default=config.DIR_DATOS, type=Path)
    ap.add_argument("--modelos", default=config.DIR_MODELOS, type=Path)
    ap.add_argument("--salidas", default=config.DIR_SALIDAS, type=Path)
    ap.add_argument("--rapido", action="store_true", help="sin búsqueda de hiperparámetros y 1 repetición de la CV")
    ap.add_argument("--candidatos", default="", help="lista separada por coma (por defecto, todos)")
    ap.add_argument("--sin-prueba", action="store_true", help="no apartar personas de prueba final")
    args = ap.parse_args()
    dir_eval = args.modelos / "evaluacion"
    for d in (args.modelos, dir_eval, args.salidas):
        d.mkdir(parents=True, exist_ok=True)
    n_rep = 1 if args.rapido else config.REPETICIONES_CV
    elegidos = [c.strip() for c in args.candidatos.split(",") if c.strip()]
    todos = candidatos(2, con_secuencia=True)
    desconocidos = set(elegidos) - set(todos)
    if desconocidos:
        sys.exit(f"Candidatos desconocidos: {', '.join(desconocidos)}. Hay: {', '.join(todos)}")

    def filtrar(c):
        res = {n: v for n, v in c.items() if not elegidos or n in elegidos}
        return res or c  # p. ej. --candidatos ROCKET: el modelo de ejercicio usa todos

    print(f"Leyendo sesiones de {args.datos} ...")
    sesiones = cargar_sesiones(args.datos)
    if not sesiones:
        sys.exit("No hay CSV válidos. Corre primero extraer_videos.py.")
    personas = sorted({s["persona"] for s in sesiones})
    por_persona = len(personas) >= 3
    prueba = [] if args.sin_prueba or not por_persona else dividir_personas(personas)
    (dir_eval / "personas_prueba.json").write_text(json.dumps(prueba, indent=1), encoding="utf-8")
    reporte = [f"Sesiones (videos): {len(sesiones)}   Personas: {len(personas)}   Vistas: {', '.join(config.VISTAS_ENTRENAR)}",
               "Grupos de validación: " + ("persona (mide qué tan bien generaliza a gente nueva)"
                                           if por_persona else "video (con 3+ personas se agrupa por persona)"),
               f"Prueba final: {len(prueba)} personas ({', '.join(prueba) if prueba else 'ninguna'})"]
    tabla = Counter((s["ejercicio"], s["etiqueta"]) for s in sesiones)
    reporte.append("Videos por ejercicio y etiqueta:")
    reporte += [f"  {e:11s} {et:18s} {n}" for (e, et), n in sorted(tabla.items())]
    filas = []

    Xs, y, g, ids, per = dataset_ejercicio(sesiones, por_persona)
    final, de_eval, f = evaluar_tarea("Ejercicio", "ejercicio", Xs, {"tabular": pose.nombres_rasgos_ventana()},
                                      y, g, ids, per, prueba, filtrar(candidatos(len(set(y)), rapido=args.rapido)),
                                      args.salidas, reporte, n_rep)
    if final is None:
        sys.exit("\n".join(reporte) + "\n\nSe necesitan videos de al menos 2 ejercicios.")
    joblib.dump(final, args.modelos / "ejercicio.joblib")
    if de_eval:
        joblib.dump(de_eval, dir_eval / "ejercicio.joblib")
    filas += f

    nombres = {"tabular": pose.nombres_rasgos_rep(), "secuencia": pose.nombres_secuencia()}
    for ej in config.EJERCICIOS:
        Xs, y, g, ids, per, reps = dataset_tecnica(sesiones, ej, por_persona)
        sin_reps = [k for k, n in reps.items() if n == 0]
        if sin_reps:
            reporte.append(f"\n[!] {ej}: no se detectaron repeticiones en {', '.join(sin_reps)}")
        if len(y) == 0:
            reporte.append(f"\n{ej}: sin repeticiones, no se entrena modelo de técnica.")
            continue
        final, de_eval, f = evaluar_tarea(f"Tecnica {ej}", f"tecnica_{ej}", Xs, nombres, y, g, ids, per, prueba,
                                          filtrar(candidatos(len(set(y)), con_secuencia=True, rapido=args.rapido)),
                                          args.salidas, reporte, n_rep)
        if final is not None:
            joblib.dump(final, args.modelos / f"tecnica_{ej}.joblib")
        if de_eval:
            joblib.dump(de_eval, dir_eval / f"tecnica_{ej}.joblib")
        filas += f

        # tipo de error: solo con videos grabados con un error específico (PROTOCOLO_GRABACION.md)
        ruta_err = args.modelos / f"error_{ej}.joblib"
        Xs, y, g, ids, per, _ = dataset_tecnica(sesiones, ej, por_persona, errores=True)
        n_personas = len(set(per))
        if len(set(y)) < 2 or n_personas < 2:
            if len(y):
                reporte.append(f"\n[!] Tipo de error {ej}: se necesitan al menos 2 errores distintos grabados por "
                               f"2+ personas (hay {len(set(y))} errores, {n_personas} personas).")
            ruta_err.unlink(missing_ok=True)
            continue
        # con pocas personas no se aparta prueba: se reporta solo la validación cruzada por persona
        final, _, f = evaluar_tarea(f"Tipo de error {ej}", f"error_{ej}", Xs, nombres, y, g, ids, per, [],
                                    filtrar(candidatos(len(set(y)), con_secuencia=True, rapido=args.rapido)),
                                    args.salidas, reporte, n_rep)
        if final is not None:
            joblib.dump(final, ruta_err)
        filas += f

    if filas:
        campos = list(dict.fromkeys(k for fila in filas for k in fila))
        with open(args.salidas / "resultados.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
    texto = "\n".join(reporte)
    (args.salidas / "reporte_entrenamiento.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nModelos para usar en vivo en {args.modelos}"
          f"\nModelos sin las personas de prueba (para evaluar_sistema.py) en {dir_eval}"
          f"\nReporte, resultados.csv y gráficos en {args.salidas}")


if __name__ == "__main__":
    main()
