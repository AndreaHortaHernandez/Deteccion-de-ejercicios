"""Configuración compartida por capturar.py, entrenar.py y en_vivo.py."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DIR_DATOS = RAIZ / "datos"
DIR_MODELOS = RAIZ / "modelos"
DIR_SALIDAS = RAIZ / "salidas"
MODELO_POSE = DIR_MODELOS / "pose_landmarker_full.task"
MODELO_EJERCICIO = DIR_MODELOS / "ejercicio.joblib"

CAMARA = 0

# Para cada ejercicio:
#   angulo    -> ángulo principal con el que se cuentan las repeticiones
#   mendeley  -> nombre del ejercicio en el dataset de Mendeley (importar_mendeley.py)
#   etiquetas -> clases del modelo de técnica y el mensaje que se muestra en vivo.
#                Mejora futura: cambiar "mal" por errores específicos (p. ej. "cadera_caida"),
#                grabando videos de cada error; el resto del código ya lo soporta.
EJERCICIOS = {
    "sentadilla": {
        "nombre": "Sentadilla",
        "angulo": "rodilla",
        "mendeley": "squat",
        "etiquetas": {
            "bien": "Buena tecnica",
            "mal": "Corrige: baja con control y manten la espalda recta",
        },
    },
    "lagartija": {
        "nombre": "Lagartija",
        "angulo": "codo",
        "mendeley": "push_up",
        "etiquetas": {
            "bien": "Buena tecnica",
            "mal": "Corrige: cuerpo recto y baja hasta ~90 grados de codo",
        },
    },
    "curl": {
        "nombre": "Curl de biceps",
        "angulo": "codo",
        "mendeley": "bicep_curl",
        "etiquetas": {
            "bien": "Buena tecnica",
            "mal": "Corrige: codo pegado al cuerpo y sin balancear el torso",
        },
    },
}

# Reconocimiento del ejercicio: ventana deslizante de ángulos, medida en SEGUNDOS para que
# funcione igual con videos de 15 o 30 fps y con la cámara de Colab (pocos cuadros por segundo)
VENTANA_S = 2.0         # duración de cada ventana
PASO_S = 0.33           # cada cuánto se toma una ventana al entrenar / se predice en vivo
MIN_FRAMES_VENTANA = 8  # una ventana con menos cuadros válidos no se usa
SUAVIZADO = 5           # predicciones que se votan en vivo para que no "parpadee" el ejercicio
PROB_MIN_EJERCICIO = 0.6
UMBRAL_MOVIMIENTO = 6.0  # desviación estándar (grados) mínima de codo o rodilla para considerar que hay ejercicio

# Conteo de repeticiones con el ángulo principal
DELTA_INICIO = 25.0     # la rep empieza cuando el ángulo baja esto (grados) desde el máximo reciente
RECUPERACION = 0.7      # y termina cuando recupera esta fracción del recorrido (del mínimo al máximo)
MIN_DUR_REP = 0.5       # segundos; reps más cortas se descartan (ruido)
MAX_DUR_REP = 10.0      # segundos; si una rep tarda más, se reinicia el conteo

VIS_MIN = 0.5           # visibilidad mínima de los puntos del lado que se usa
