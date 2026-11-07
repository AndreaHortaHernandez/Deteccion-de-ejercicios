"""Detección de pose (MediaPipe), señales por cuadro, rasgos y conteo de repeticiones.

Todo lo que se usa al entrenar y en vivo sale de aquí, para que ambos calculen exactamente lo mismo.
"""
import csv
import math
import warnings
from collections import deque

import cv2
import numpy as np

import config

# Índices de MediaPipe Pose (33 puntos): (izquierdo, derecho)
PUNTOS = {
    "hombro": (11, 12), "codo": (13, 14), "muneca": (15, 16),
    "cadera": (23, 24), "rodilla": (25, 26), "tobillo": (27, 28),
}
PIE = (31, 32)  # punta del pie
ANGULOS = ["rodilla", "cadera", "codo", "hombro", "torso", "linea_cuerpo"]
# medidas normalizadas por la longitud del segmento (no dependen de la distancia a la cámara)
MEDIDAS = ["prof_cadera", "rodilla_pie"]
SENALES = ANGULOS + MEDIDAS
CONEXIONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
              (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]
N_PUNTOS = 33


# ----------------------------------------------------------------- detector
class DetectorPose:
    """Envuelve el PoseLandmarker de MediaPipe en modo video. Úsalo con `with`."""

    def __init__(self, modelo=config.MODELO_POSE):
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        asegurar_modelo_pose(modelo)
        opciones = vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(modelo)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opciones)
        self._ultimo_ts = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._lm.close()
        return False

    def detectar(self, frame_bgr, ts_ms):
        """Devuelve un arreglo (33, 4) con x, y (normalizados 0-1), z y visibilidad, o None."""
        import mediapipe as mp
        ts_ms = max(int(ts_ms), self._ultimo_ts + 1)  # MediaPipe exige timestamps crecientes
        self._ultimo_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        if not res.pose_landmarks:
            return None
        return np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks[0]], dtype=np.float32)


URL_MODELO_POSE = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def asegurar_modelo_pose(ruta=config.MODELO_POSE):
    """Descarga el modelo de pose de MediaPipe (≈9 MB) si todavía no está."""
    from pathlib import Path
    import urllib.request
    ruta = Path(ruta)
    if not ruta.exists():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando modelo de pose en {ruta} ...")
        urllib.request.urlretrieve(URL_MODELO_POSE, ruta)
    return ruta


# ----------------------------------------------------------------- señales
def _ang(a, b, c):
    """Ángulo ABC en grados (0-180)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def visibilidad_lados(lm):
    """Suma de visibilidad de hombro, codo, muñeca, cadera, rodilla y tobillo de cada lado."""
    return np.array([sum(lm[PUNTOS[p][lado], 3] for p in PUNTOS) for lado in (0, 1)])


def senales(lm, w, h, lado):
    """Calcula las señales (orden de SENALES) con el lado del cuerpo indicado (0 izq., 1 der.).

    Devuelve None si el tronco y el brazo no se ven bien (se descarta el frame). Las señales que
    usan la pierna quedan en NaN si la rodilla o el tobillo no se ven (en el curl a veces salen
    del cuadro): así no entra basura a los rasgos.
    """
    if lm is None:
        return None
    idx = {p: PUNTOS[p][lado] for p in PUNTOS}
    idx["pie"] = PIE[lado]
    if min(lm[idx[p], 3] for p in ("hombro", "codo", "muneca", "cadera")) < config.VIS_MIN:
        return None
    ve = {p: lm[i, 3] >= config.VIS_MIN for p, i in idx.items()}
    P = {p: lm[i, :2] * np.array([w, h], dtype=np.float32) for p, i in idx.items()}
    nan = float("nan")
    torso = P["hombro"] - P["cadera"]
    inclinacion = math.degrees(math.atan2(abs(torso[0]), -torso[1]))  # 0 = torso vertical, 90 = horizontal
    rodilla, tobillo = ve["rodilla"], ve["rodilla"] and ve["tobillo"]
    muslo = float(np.linalg.norm(P["rodilla"] - P["cadera"]))
    espinilla = float(np.linalg.norm(P["rodilla"] - P["tobillo"]))
    # profundidad: + si la cadera está más abajo que la rodilla (la y de la imagen crece hacia abajo)
    prof = (P["cadera"][1] - P["rodilla"][1]) / muslo if rodilla and muslo > 1e-6 else nan
    # rodilla delante de la punta del pie (+), en el sentido en que apunta el pie
    if tobillo and ve["pie"] and espinilla > 1e-6:
        sentido = 1.0 if P["pie"][0] >= P["tobillo"][0] else -1.0
        rod_pie = (P["rodilla"][0] - P["pie"][0]) * sentido / espinilla
    else:
        rod_pie = nan
    valores = [
        _ang(P["cadera"], P["rodilla"], P["tobillo"]) if tobillo else nan,
        _ang(P["hombro"], P["cadera"], P["rodilla"]) if rodilla else nan,
        _ang(P["hombro"], P["codo"], P["muneca"]),
        _ang(P["codo"], P["hombro"], P["cadera"]),
        inclinacion,
        _ang(P["hombro"], P["cadera"], P["tobillo"]) if ve["tobillo"] else nan,
        prof,
        rod_pie,
    ]
    if any(math.isnan(valores[SENALES.index(s)]) for s in ("codo", "hombro", "torso")):
        return None
    return np.array(valores, dtype=np.float32)


def _nan_mediana(H):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # columnas todas NaN -> NaN
        return np.nanmedian(H, axis=0).astype(np.float32)


class ProcesadorPose:
    """Landmarks de cada cuadro -> vector de señales. Se usa igual al entrenar y en vivo.

    - Lado del cuerpo: se fija con histéresis sobre la visibilidad suavizada. Elegirlo en cada
      cuadro (v1) hacía que los ángulos saltaran de un brazo al otro cuando la visibilidad oscilaba.
    - Mediana móvil causal de MEDIANA_FRAMES cuadros contra el temblor de MediaPipe (solo usa
      cuadros pasados, así que sirve en vivo).
    """

    def __init__(self):
        self.lado = None
        self.vis = None
        self.hist = deque(maxlen=max(1, config.MEDIANA_FRAMES))

    def __call__(self, lm, w, h):
        if lm is None:
            return None
        v = visibilidad_lados(lm)
        if not config.LADO_FIJO or self.lado is None:
            self.vis = v
            lado = int(np.argmax(v))
        else:
            self.vis = (1 - config.ALFA_LADO) * self.vis + config.ALFA_LADO * v
            lado = self.lado
            if self.vis[1 - lado] > self.vis[lado] + config.HISTERESIS_LADO:
                lado = 1 - lado
        if lado != self.lado:
            self.hist.clear()  # no mezclar en la mediana cuadros de los dos lados
            self.lado = lado
        s = senales(lm, w, h, lado)
        if s is None:
            return None
        self.hist.append(s)
        return s if len(self.hist) == 1 else _nan_mediana(np.array(self.hist))


# ----------------------------------------------------------------- rasgos
def _nombres_estad():
    bajo, alto = ("min", "max") if config.ESTADISTICOS == "minmax" else ("p05", "p95")
    return ["media", "std", bajo, alto]


def _estad(A):
    """Media, desviación y extremos (p5/p95 o mín./máx.) de cada columna, ignorando NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if config.ESTADISTICOS == "minmax":
            bajo, alto = np.nanmin(A, 0), np.nanmax(A, 0)
        else:
            bajo, alto = np.nanpercentile(A, 5, axis=0), np.nanpercentile(A, 95, axis=0)
        return [np.nanmean(A, 0), np.nanstd(A, 0), bajo, alto]


def nombres_rasgos_ventana():
    return [f"{s}_{e}" for e in _nombres_estad() for s in SENALES]


def rasgos_ventana(A):
    """A: (n, señales) de una ventana -> rasgos para reconocer el ejercicio."""
    return np.concatenate(_estad(np.asarray(A, dtype=np.float64))).astype(np.float32)


def ventanas(t, A, paso=None):
    """Corta una sesión en ventanas de VENTANA_S segundos cada `paso` segundos (PASO_S por defecto).
    Devuelve una lista de arreglos de señales (solo ventanas con suficientes cuadros)."""
    paso = config.PASO_S if paso is None else paso
    t = np.asarray(t)
    res = []
    if len(t) == 0:
        return res
    inicio = t[0]
    while inicio + config.VENTANA_S <= t[-1] + 1e-6:
        m = (t >= inicio) & (t < inicio + config.VENTANA_S)
        if m.sum() >= config.MIN_FRAMES_VENTANA:
            res.append(A[m])
        inicio += paso
    return res


FASE = ["fase_duracion_s", "fase_bajada_s", "fase_subida_s", "fase_fraccion_bajada",
        "fase_vel_bajada", "fase_vel_subida", "fase_recorrido"]


ESTAD_BASICOS = ["min", "max", "media", "std", "rango"]


def nombres_rasgos_rep():
    """Familias (config.RASGOS_REP):
    basicos -> mín., máx., media, desv. y rango de los 6 ángulos + duración (los de la versión 1)
    estad -> media, desv. y extremos de cada señal en toda la rep (rasgos genéricos)
    fase  -> duración de bajada y subida, velocidad angular máxima y recorrido del ángulo principal
    flex  -> valor de cada señal en el punto de máxima flexión (fondo de la sentadilla, pecho
             abajo en la lagartija, codo arriba en el curl): ahí se ve casi todo error de técnica
    """
    nombres = []
    if "basicos" in config.RASGOS_REP:
        nombres += [f"bas_{s}_{e}" for e in ESTAD_BASICOS for s in ANGULOS] + ["bas_duracion_s"]
    if "estad" in config.RASGOS_REP:
        nombres += [f"{s}_{e}" for e in _nombres_estad() for s in SENALES]
    if "fase" in config.RASGOS_REP:
        nombres += FASE
    if "flex" in config.RASGOS_REP:
        nombres += [f"flex_{s}" for s in SENALES]
    return nombres


def rasgos_rep(A, t, k):
    """A: (n, señales) de una repetición, t: tiempos en s, k: índice del ángulo principal."""
    A = np.asarray(A, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    a = A[:, k]
    i = int(np.nanargmin(a))  # máxima flexión
    partes = []
    if "basicos" in config.RASGOS_REP:
        B = A[:, :len(ANGULOS)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mn, mx = np.nanmin(B, 0), np.nanmax(B, 0)
            partes += [mn, mx, np.nanmean(B, 0), np.nanstd(B, 0), mx - mn, [t[-1] - t[0]]]
    if "estad" in config.RASGOS_REP:
        partes += _estad(A)
    if "fase" in config.RASGOS_REP:
        dur, bajada, subida = t[-1] - t[0], t[i] - t[0], t[-1] - t[i]
        dt = np.diff(t)
        v = np.diff(a) / np.where(dt > 0, dt, np.nan)  # grados por segundo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v_baja = -np.nanmin(v[:i]) if i > 0 else 0.0
            v_sube = np.nanmax(v[i:]) if i < len(v) else 0.0
        partes.append([dur, bajada, subida, bajada / dur if dur > 0 else 0.5,
                       v_baja, v_sube, np.nanmax(a) - np.nanmin(a)])
    if "flex" in config.RASGOS_REP:
        partes.append(A[i])
    return np.concatenate([np.ravel(p) for p in partes]).astype(np.float32)


def nombres_secuencia():
    return [f"{s}_t{j:02d}" for s in SENALES for j in range(config.PUNTOS_SECUENCIA)]


def secuencia_rep(A, t):
    """Remuestrea la rep a PUNTOS_SECUENCIA instantes (tiempo normalizado 0-1) -> vector plano
    (señal x tiempo) para modelos de series de tiempo como ROCKET."""
    A = np.asarray(A, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    u = (t - t[0]) / max(t[-1] - t[0], 1e-6)
    malla = np.linspace(0, 1, config.PUNTOS_SECUENCIA)
    S = np.full((A.shape[1], len(malla)), np.nan)
    for c in range(A.shape[1]):
        ok = ~np.isnan(A[:, c])
        if ok.sum() >= 2:
            S[c] = np.interp(malla, u[ok], A[ok, c])
    return S.ravel().astype(np.float32)


def hay_movimiento(A):
    """True si el codo o la rodilla se mueven lo suficiente en la ventana."""
    A = np.asarray(A, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = [np.nanstd(A[:, SENALES.index(s)]) for s in ("codo", "rodilla")]
    return float(np.max(np.nan_to_num(d))) >= config.UMBRAL_MOVIMIENTO


# ----------------------------------------------------------------- repeticiones
class ContadorReps:
    """Segmenta repeticiones con el ángulo principal del ejercicio (rodilla o codo).

    Una rep empieza cuando el ángulo baja DELTA_INICIO grados desde el máximo reciente
    (posición extendida) y termina cuando recupera RECUPERACION (70 %) del recorrido entre
    ese máximo y el mínimo de la rep. Así se cuentan también las reps mal hechas (poco
    profundas) y no se juntan dos reps cuando la persona no se estira por completo.
    Los cuadros en los que no se ve el ángulo principal se ignoran.
    """

    def __init__(self, nombre_angulo, delta_inicio=None, recuperacion=None):
        self.k = SENALES.index(nombre_angulo)
        self.delta = config.DELTA_INICIO if delta_inicio is None else delta_inicio
        self.recup = config.RECUPERACION if recuperacion is None else recuperacion
        self.reset()

    def reset(self):
        self.estado = "arriba"
        self.tope = -1.0
        self.minimo = 999.0
        self.desde_tope = []   # frames (señales, t) desde el último máximo
        self.rep = []

    def actualizar(self, A_frame, t):
        """Agrega un frame. Devuelve (señales_rep, tiempos_rep) cuando se completa una rep; si no, None."""
        a = float(A_frame[self.k])
        if math.isnan(a):
            return None
        if self.estado == "arriba":
            if a >= self.tope:
                self.tope = a
                self.desde_tope = [(A_frame, t)]
            else:
                self.desde_tope.append((A_frame, t))
            if a < self.tope - self.delta:
                self.estado = "abajo"
                # la rep empieza justo antes de bajar, no en el máximo (que puede estar en la pausa de arriba)
                cerca = [i for i, (f, _) in enumerate(self.desde_tope)
                         if float(f[self.k]) >= self.tope - config.TOLERANCIA_INICIO]
                self.rep = list(self.desde_tope[cerca[-1]:])
                self.minimo = a
            return None
        # estado "abajo"
        self.rep.append((A_frame, t))
        self.minimo = min(self.minimo, a)
        dur = t - self.rep[0][1]
        if dur > config.MAX_DUR_REP:
            self.reset()
            return None
        if a >= self.minimo + self.recup * (self.tope - self.minimo):
            rep = self.rep
            self.reset()
            self.tope = a
            self.desde_tope = [(A_frame, t)]
            if dur < config.MIN_DUR_REP:
                return None
            return np.array([r[0] for r in rep]), np.array([r[1] for r in rep])
        return None


def contar_reps(t, A, ejercicio, **kw):
    """Corre el contador sobre una sesión completa. Devuelve la lista de (señales_rep, tiempos_rep)."""
    cont = ContadorReps(config.EJERCICIOS[ejercicio]["angulo"], **kw)
    reps = []
    for a, tt in zip(A, t):
        r = cont.actualizar(a, tt)
        if r is not None:
            reps.append(r)
    return reps


# ----------------------------------------------------------------- análisis completo
def _comprobar_rasgos(m, esperados, ruta):
    if m.get("rasgos") != esperados:
        raise ValueError(f"{ruta} se entrenó con otros rasgos (otra versión o config distinta). "
                         "Vuelve a correr entrenar.py.")


class Analizador:
    """Junta todo: reconoce el ejercicio, cuenta reps y evalúa la técnica de cada una.

    Se alimenta frame por frame con `paso_pose(landmarks, w, h, t)` (o `paso(señales, t)` si ya
    se calcularon); lo usan analizar_video.py, en_vivo.py, la celda de cámara del notebook y
    evaluar_sistema.py.
    """

    def __init__(self, dir_modelos=config.DIR_MODELOS):
        import joblib
        from pathlib import Path
        dir_modelos = Path(dir_modelos)
        ruta = dir_modelos / "ejercicio.joblib"
        if not ruta.exists():
            raise FileNotFoundError(f"No hay modelo de ejercicio en {ruta}. Corre primero entrenar.py.")
        self.m_ej = joblib.load(ruta)
        _comprobar_rasgos(self.m_ej, nombres_rasgos_ventana(), ruta)
        self.m_tec = {}
        for ej in config.EJERCICIOS:
            r = dir_modelos / f"tecnica_{ej}.joblib"
            if r.exists():
                m = joblib.load(r)
                _comprobar_rasgos(m, nombres_secuencia() if m.get("entrada") == "secuencia"
                                  else nombres_rasgos_rep(), r)
                self.m_tec[ej] = m
        # modelo opcional que dice QUÉ error es (solo si se grabaron errores específicos)
        self.m_err = {}
        for ej in config.EJERCICIOS:
            r = dir_modelos / f"error_{ej}.joblib"
            if r.exists():
                m = joblib.load(r)
                _comprobar_rasgos(m, nombres_secuencia() if m.get("entrada") == "secuencia"
                                  else nombres_rasgos_rep(), r)
                self.m_err[ej] = m
        self.reiniciar()

    def reiniciar(self):
        """Vuelve a empezar el conteo (sin volver a cargar los modelos)."""
        self.proc = ProcesadorPose()
        self.buf = deque()      # (señales, t) de los últimos VENTANA_S segundos
        self.votos = deque(maxlen=config.SUAVIZADO)
        self.t_pred = -1e9
        self.ejercicio = None
        self.contador = None
        self.conteo = {ej: {"reps": 0, "bien": 0, "mal": 0} for ej in config.EJERCICIOS}
        self.ultimo = None      # veredicto de la última rep
        self.historial = []     # todas las reps evaluadas
        self.t_serie = -1e9     # inicio de la serie actual (cuando se reconoció el ejercicio)

    def paso_pose(self, lm, w, h, t):
        """Procesa los landmarks de un frame (o None si no se detectó a nadie)."""
        return self.paso(self.proc(lm, w, h), t)

    def _votar(self):
        validos = [v for v in self.votos if v is not None]
        if not validos:
            return None
        mejor = max(set(validos), key=validos.count)
        return mejor if validos.count(mejor) > len(self.votos) // 2 else None

    def paso(self, A, t):
        """Procesa las señales de un frame. Devuelve el veredicto si en este frame terminó una rep; si no, None."""
        if A is None:
            return None
        self.buf.append((A, t))
        while self.buf and t - self.buf[0][1] > config.VENTANA_S:
            self.buf.popleft()
        lleno = t - self.buf[0][1] >= 0.9 * config.VENTANA_S and len(self.buf) >= config.MIN_FRAMES_VENTANA
        if lleno and t - self.t_pred >= config.PASO_S:
            self.t_pred = t
            W = np.array([a for a, _ in self.buf])
            pred = None
            if hay_movimiento(W):
                p = self.m_ej["modelo"].predict_proba(rasgos_ventana(W)[None])[0]
                i = int(np.argmax(p))
                if p[i] >= config.PROB_MIN_EJERCICIO:
                    pred = self.m_ej["clases"][i]
            self.votos.append(pred)
            nuevo = self._votar()
            if nuevo is not None and nuevo != self.ejercicio:
                return self._cambiar_ejercicio(nuevo)
        if self.contador is None:
            return None
        rep = self.contador.actualizar(A, t)
        if rep is None:
            return None
        return self._evaluar(*rep)

    def _cambiar_ejercicio(self, nuevo):
        """El reconocimiento va 1-2 s atrasado: las reps del ejercicio anterior que terminaron
        dentro de la ventana actual en realidad eran del nuevo. Se deshacen y la ventana se
        vuelve a procesar con el contador del ejercicio nuevo."""
        t0 = self.buf[0][1]
        while self.historial and self.historial[-1]["ejercicio"] == self.ejercicio \
                and self.historial[-1]["t"] >= t0:
            v = self.historial.pop()
            c = self.conteo[v["ejercicio"]]
            c["reps"] -= 1
            if v["ok"] is not None:
                c["bien" if v["ok"] else "mal"] -= 1
        self.ultimo = self.historial[-1] if self.historial else None
        self.ejercicio = nuevo
        self.t_serie = t0
        self.contador = ContadorReps(config.EJERCICIOS[nuevo]["angulo"])
        veredicto = None
        for a, tt in list(self.buf):
            rep = self.contador.actualizar(a, tt)
            if rep is not None:
                veredicto = self._evaluar(*rep)
        return veredicto

    def _evaluar(self, A_rep, t_rep):
        ej = self.ejercicio
        c = self.conteo[ej]
        c["reps"] += 1
        v = {"ejercicio": ej, "rep": c["reps"], "etiqueta": None, "ok": None, "prob": 0.0, "p_mal": None,
             "error": None,
             "texto": "Sin modelo de técnica para este ejercicio", "t": float(t_rep[-1]),
             "t_inicio": float(t_rep[0])}
        if ej in self.m_tec:
            m = self.m_tec[ej]
            x = (secuencia_rep(A_rep, t_rep) if m.get("entrada") == "secuencia"
                 else rasgos_rep(A_rep, t_rep, self.contador.k))[None]
            p = m["modelo"].predict_proba(x)[0]
            et = m["modelo"].predict(x)[0]  # aplica el umbral de p(mal) elegido al entrenar
            ok = et == "bien"
            c["bien" if ok else "mal"] += 1
            v.update(etiqueta=et, ok=ok, texto=config.EJERCICIOS[ej]["etiquetas"].get(et, et),
                     prob=float(p[list(m["clases"]).index(et)]),
                     p_mal=float(p[list(m["clases"]).index("mal")]) if "mal" in m["clases"] else None)
            if not ok and ej in self.m_err:
                me = self.m_err[ej]
                xe = (secuencia_rep(A_rep, t_rep) if me.get("entrada") == "secuencia"
                      else rasgos_rep(A_rep, t_rep, self.contador.k))[None]
                err = me["modelo"].predict(xe)[0]
                v.update(error=err, texto=config.EJERCICIOS[ej].get("errores", {}).get(err, err))
        self.ultimo = v
        self.historial.append(v)
        return v

    def veredicto_serie(self):
        """Veredicto de la serie actual: promedio de p(mal) de sus reps.

        Con el dataset real, decidir por serie (o video) acierta bastante más que rep por rep
        (acierto balanceado ~0.70 vs ~0.60): una sola rep es ruidosa, pero el patrón de "mala técnica"
        se repite en toda la serie. Devuelve (ok, p_mal_media, n_reps) o None si aún no hay suficientes.
        """
        ps = [v["p_mal"] for v in self.historial
              if v["ejercicio"] == self.ejercicio and v["p_mal"] is not None and v["t"] >= self.t_serie]
        if len(ps) < config.MIN_REPS_SERIE:
            return None
        umbral = self.m_tec.get(self.ejercicio, {}).get("umbral_mal") or 0.5
        media = float(np.mean(ps))
        return media < umbral, media, len(ps)

    def dibujar_hud(self, img, t_actual):
        """Escribe sobre img (BGR) el ejercicio, el conteo y el veredicto de la última rep."""
        h, w = img.shape[:2]
        esc = max(0.5, w / 1100)
        ej = self.ejercicio
        nombre = config.EJERCICIOS[ej]["nombre"] if ej else "Buscando ejercicio..."
        texto(img, nombre, (15, int(38 * esc / 0.6)), 1.1 * esc, (0, 255, 255))
        if ej:
            c = self.conteo[ej]
            texto(img, f"Reps: {c['reps']}   Bien: {c['bien']}   Mal: {c['mal']}",
                  (15, int(75 * esc / 0.6)), 0.8 * esc)
            s = self.veredicto_serie()
            if s:
                ok, media, n = s
                texto(img, f"Serie ({n} reps): {'BUENA TECNICA' if ok else 'A CORREGIR'}  p(mal) {media:.0%}",
                      (15, int(110 * esc / 0.6)), 0.8 * esc, (80, 220, 80) if ok else (60, 60, 255))
        v = self.ultimo
        if v and t_actual - v["t"] < 3.0:
            color = (80, 220, 80) if v["ok"] else (60, 60, 255) if v["ok"] is False else (200, 200, 200)
            pref = "BIEN" if v["ok"] else "MAL" if v["ok"] is False else "?"
            texto(img, f"Rep {v['rep']}: {pref} - {v['texto']}", (15, h - int(25 * esc / 0.6)), 0.8 * esc, color)


# ----------------------------------------------------------------- archivos
def guardar_sesion(ruta, filas, w, h):
    """filas: lista de (t, lm (33,4)). Guarda un CSV con t, w, h y los 33 puntos."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    enc = ["t", "w", "h"] + [f"{c}{i}" for i in range(N_PUNTOS) for c in ("x", "y", "z", "v")]
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(enc)
        for t, lm in filas:
            wr.writerow([f"{t:.4f}", w, h] + [f"{v:.5f}" for v in lm.reshape(-1)])


def leer_landmarks(ruta):
    """Devuelve (t, LM (n, 33, 4), w, h) tal como están en el CSV de sesión."""
    datos = np.loadtxt(ruta, delimiter=",", skiprows=1, ndmin=2)
    return datos[:, 0], datos[:, 3:].reshape(-1, N_PUNTOS, 4), datos[:, 1], datos[:, 2]


def senales_sesion(t, LM, w, h):
    """Pasa los landmarks de una sesión por un ProcesadorPose nuevo -> (t, A) de los frames válidos."""
    proc = ProcesadorPose()
    ts, filas = [], []
    for ti, lm, wi, hi in zip(t, LM, w, h):
        a = proc(lm, wi, hi)
        if a is not None:
            ts.append(ti)
            filas.append(a)
    return np.array(ts), np.array(filas, dtype=np.float32).reshape(-1, len(SENALES))


def leer_sesion(ruta):
    """Devuelve (t, A): tiempos y señales de los frames válidos de un CSV de sesión."""
    return senales_sesion(*leer_landmarks(ruta))


# ----------------------------------------------------------------- dibujo
def dibujar(frame, lm, color=(0, 255, 0)):
    if lm is None:
        return
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in lm[:, :2]]
    for a, b in CONEXIONES:
        if lm[a, 3] > 0.3 and lm[b, 3] > 0.3:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i in {i for c in CONEXIONES for i in c}:
        if lm[i, 3] > 0.3:
            cv2.circle(frame, pts[i], 4, (255, 255, 255), -1, cv2.LINE_AA)


def texto(img, s, pos, escala=0.7, color=(255, 255, 255), grosor=2):
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor + 3, cv2.LINE_AA)
    cv2.putText(img, s, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)
