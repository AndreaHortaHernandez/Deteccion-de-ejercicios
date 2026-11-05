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
#   angulo  -> ángulo principal con el que se cuentan las repeticiones
#   etiquetas -> "bien" + los errores que se graban; el texto es lo que se muestra en vivo
EJERCICIOS = {
    "sentadilla": {
        "nombre": "Sentadilla",
        "angulo": "rodilla",
        "etiquetas": {
            "bien": "Bien hecha",
            "poco_profunda": "No bajaste lo suficiente",
            "espalda_inclinada": "Inclinas demasiado el torso",
        },
        "como_grabar": "De perfil, cuerpo completo (pies a cabeza). Brazos al frente.",
    },
    "lagartija": {
        "nombre": "Lagartija",
        "angulo": "codo",
        "etiquetas": {
            "bien": "Bien hecha",
            "cadera_caida": "Se te cae la cadera",
            "poco_profunda": "No bajaste lo suficiente",
        },
        "como_grabar": "De perfil, cámara a la altura del piso, cuerpo completo.",
    },
    "curl": {
        "nombre": "Curl de biceps",
        "angulo": "codo",
        "etiquetas": {
            "bien": "Bien hecho",
            "balanceo": "Balanceas el torso o mueves el codo",
            "rango_incompleto": "No subes la mancuerna completa",
        },
        "como_grabar": "De perfil, al menos de la cabeza a las rodillas.",
    },
}

# Reconocimiento del ejercicio: ventana deslizante de ángulos
VENTANA = 30            # frames (~1 s a 30 fps)
PASO_VENTANA = 5        # cada cuántos frames se toma una ventana al entrenar / se predice en vivo
SUAVIZADO = 5           # predicciones que se votan en vivo para que no "parpadee" el ejercicio
PROB_MIN_EJERCICIO = 0.6
UMBRAL_MOVIMIENTO = 6.0  # desviación estándar (grados) mínima de codo o rodilla para considerar que hay ejercicio

# Conteo de repeticiones (histeresis relativa al punto más alto/extendido)
DELTA_INICIO = 25.0     # la rep empieza cuando el ángulo baja esto (grados) desde el máximo reciente
DELTA_FIN = 15.0        # y termina cuando regresa a menos de esto del máximo
MIN_DUR_REP = 0.5       # segundos; reps más cortas se descartan (ruido)
MAX_DUR_REP = 10.0      # segundos; si una rep tarda más, se reinicia el conteo

VIS_MIN = 0.5           # visibilidad mínima de los puntos del lado que se usa
