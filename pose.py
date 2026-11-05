"""Detección de pose (MediaPipe), cálculo de ángulos, rasgos y conteo de repeticiones.

Todo lo que se usa al entrenar y en vivo sale de aquí, para que ambos calculen exactamente lo mismo.
"""
import csv
import math
from collections import deque

import cv2
import numpy as np

import config

# Índices de MediaPipe Pose (33 puntos): (izquierdo, derecho)
PUNTOS = {
    "hombro": (11, 12), "codo": (13, 14), "muneca": (15, 16),
    "cadera": (23, 24), "rodilla": (25, 26), "tobillo": (27, 28),
}
ANGULOS = ["rodilla", "cadera", "codo", "hombro", "torso", "linea_cuerpo"]
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


# ----------------------------------------------------------------- ángulos
def _ang(a, b, c):
    """Ángulo ABC en grados (0-180)."""
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def angulos(lm, w, h):
    """Calcula los 6 ángulos (orden de ANGULOS) usando el lado del cuerpo más visible.

    Devuelve None si el tronco y el brazo no se ven bien (se descarta el frame).
    """
    if lm is None:
        return None
    # lado más visible: suma de visibilidad de hombro, codo, muñeca, cadera, rodilla y tobillo
    vis = [sum(lm[PUNTOS[p][lado], 3] for p in PUNTOS) for lado in (0, 1)]
    lado = 0 if vis[0] >= vis[1] else 1
    idx = {p: PUNTOS[p][lado] for p in PUNTOS}
    # el tronco y el brazo deben verse; las piernas pueden salir del cuadro (curl)
    if min(lm[idx[p], 3] for p in ("hombro", "codo", "muneca", "cadera")) < config.VIS_MIN:
        return None
    P = {p: lm[i, :2] * np.array([w, h], dtype=np.float32) for p, i in idx.items()}
    torso = P["hombro"] - P["cadera"]
    inclinacion = math.degrees(math.atan2(abs(torso[0]), -torso[1]))  # 0 = torso vertical, 90 = horizontal
    valores = [
        _ang(P["cadera"], P["rodilla"], P["tobillo"]),
        _ang(P["hombro"], P["cadera"], P["rodilla"]),
        _ang(P["hombro"], P["codo"], P["muneca"]),
        _ang(P["codo"], P["hombro"], P["cadera"]),
        inclinacion,
        _ang(P["hombro"], P["cadera"], P["tobillo"]),
    ]
    if any(math.isnan(v) for v in valores):
        return None
    return np.array(valores, dtype=np.float32)


# ----------------------------------------------------------------- rasgos
ESTAD_VENTANA = ["media", "std", "min", "max"]
NOMBRES_RASGOS_VENTANA = [f"{a}_{e}" for e in ESTAD_VENTANA for a in ANGULOS]


def rasgos_ventana(A):
    """A: (n, 6) ángulos de una ventana -> 24 rasgos para reconocer el ejercicio."""
    A = np.asarray(A, dtype=np.float32)
    return np.concatenate([A.mean(0), A.std(0), A.min(0), A.max(0)])


ESTAD_REP = ["min", "max", "media", "std", "rango"]
NOMBRES_RASGOS_REP = [f"{a}_{e}" for e in ESTAD_REP for a in ANGULOS] + ["duracion_s"]


def rasgos_rep(A, t):
    """A: (n, 6) ángulos de una repetición, t: tiempos en s -> 31 rasgos para evaluar la técnica."""
    A = np.asarray(A, dtype=np.float32)
    mn, mx = A.min(0), A.max(0)
    return np.concatenate([mn, mx, A.mean(0), A.std(0), mx - mn, [float(t[-1] - t[0])]])


def hay_movimiento(A):
    """True si el codo o la rodilla se mueven lo suficiente en la ventana."""
    A = np.asarray(A)
    return max(A[:, ANGULOS.index("codo")].std(), A[:, ANGULOS.index("rodilla")].std()) >= config.UMBRAL_MOVIMIENTO


# ----------------------------------------------------------------- repeticiones
class ContadorReps:
    """Segmenta repeticiones con el ángulo principal del ejercicio (rodilla o codo).

    Una rep empieza cuando el ángulo baja DELTA_INICIO grados desde el máximo reciente
    (posición extendida) y termina cuando vuelve a quedar a menos de DELTA_FIN de ese máximo.
    Así se cuentan también las reps mal hechas (p. ej. sentadillas poco profundas).
    """

    def __init__(self, nombre_angulo):
        self.k = ANGULOS.index(nombre_angulo)
        self.reset()

    def reset(self):
        self.estado = "arriba"
        self.tope = -1.0
        self.desde_tope = []   # frames (angulos, t) desde el último máximo
        self.rep = []

    def actualizar(self, A_frame, t):
        """Agrega un frame. Devuelve (angulos_rep, tiempos_rep) cuando se completa una rep; si no, None."""
        a = float(A_frame[self.k])
        if self.estado == "arriba":
            if a >= self.tope:
                self.tope = a
                self.desde_tope = [(A_frame, t)]
            else:
                self.desde_tope.append((A_frame, t))
            if a < self.tope - config.DELTA_INICIO:
                self.estado = "abajo"
                self.rep = list(self.desde_tope)
            return None
        # estado "abajo"
        self.rep.append((A_frame, t))
        dur = t - self.rep[0][1]
        if dur > config.MAX_DUR_REP:
            self.reset()
            return None
        if a > self.tope - config.DELTA_FIN:
            rep = self.rep
            self.reset()
            self.tope = a
            self.desde_tope = [(A_frame, t)]
            if dur < config.MIN_DUR_REP:
                return None
            return np.array([r[0] for r in rep]), np.array([r[1] for r in rep])
        return None


# ----------------------------------------------------------------- análisis completo
class Analizador:
    """Junta todo: reconoce el ejercicio, cuenta reps y evalúa la técnica de cada una.

    Se alimenta frame por frame con `paso(angulos, t)`; lo usan analizar_video.py,
    en_vivo.py y la celda de cámara del notebook.
    """

    def __init__(self, dir_modelos=config.DIR_MODELOS):
        import joblib
        from pathlib import Path
        dir_modelos = Path(dir_modelos)
        ruta = dir_modelos / "ejercicio.joblib"
        if not ruta.exists():
            raise FileNotFoundError(f"No hay modelo de ejercicio en {ruta}. Corre primero entrenar.py.")
        self.m_ej = joblib.load(ruta)
        self.m_tec = {}
        for ej in config.EJERCICIOS:
            r = dir_modelos / f"tecnica_{ej}.joblib"
            if r.exists():
                self.m_tec[ej] = joblib.load(r)
        self.buf = deque(maxlen=config.VENTANA)
        self.votos = deque(maxlen=config.SUAVIZADO)
        self.n = 0
        self.ejercicio = None
        self.contador = None
        self.conteo = {ej: {"reps": 0, "bien": 0, "mal": 0} for ej in config.EJERCICIOS}
        self.ultimo = None      # veredicto de la última rep
        self.historial = []     # todas las reps evaluadas

    def _votar(self):
        validos = [v for v in self.votos if v is not None]
        if not validos:
            return None
        mejor = max(set(validos), key=validos.count)
        return mejor if validos.count(mejor) > len(self.votos) // 2 else None

    def paso(self, A, t):
        """Procesa un frame. Devuelve el veredicto si en este frame terminó una rep; si no, None."""
        if A is None:
            return None
        self.buf.append((A, t))
        self.n += 1
        if len(self.buf) == config.VENTANA and self.n % config.PASO_VENTANA == 0:
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
                self.ejercicio = nuevo
                self.contador = ContadorReps(config.EJERCICIOS[nuevo]["angulo"])
                for a, tt in list(self.buf)[:-1]:   # recupera la rep que ya iba empezando
                    self.contador.actualizar(a, tt)
        if self.contador is None:
            return None
        rep = self.contador.actualizar(A, t)
        if rep is None:
            return None
        return self._evaluar(*rep)

    def _evaluar(self, A_rep, t_rep):
        ej = self.ejercicio
        c = self.conteo[ej]
        c["reps"] += 1
        if ej not in self.m_tec:
            v = {"ejercicio": ej, "rep": c["reps"], "etiqueta": None, "ok": None,
                 "texto": "Sin modelo de técnica para este ejercicio", "prob": 0.0, "t": float(t_rep[-1])}
        else:
            m = self.m_tec[ej]
            p = m["modelo"].predict_proba(rasgos_rep(A_rep, t_rep)[None])[0]
            i = int(np.argmax(p))
            et = m["clases"][i]
            ok = et == "bien"
            c["bien" if ok else "mal"] += 1
            v = {"ejercicio": ej, "rep": c["reps"], "etiqueta": et, "ok": ok,
                 "texto": config.EJERCICIOS[ej]["etiquetas"].get(et, et), "prob": float(p[i]),
                 "t": float(t_rep[-1])}
        self.ultimo = v
        self.historial.append(v)
        return v

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


def leer_sesion(ruta):
    """Devuelve (t, A): tiempos y ángulos de los frames válidos de un CSV de sesión."""
    datos = np.loadtxt(ruta, delimiter=",", skiprows=1, ndmin=2)
    ts, filas = [], []
    for fila in datos:
        a = angulos(fila[3:].reshape(N_PUNTOS, 4), fila[1], fila[2])
        if a is not None:
            ts.append(fila[0])
            filas.append(a)
    return np.array(ts), np.array(filas).reshape(-1, len(ANGULOS))


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
