"""Configuración compartida por todos los scripts (extraer, entrenar, evaluar y en vivo)."""
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
#   etiquetas -> clases del modelo de técnica (bien/mal) y el mensaje que se muestra en vivo
#   errores   -> errores específicos que se pueden grabar (ver PROTOCOLO_GRABACION.md). Un video
#                <ejercicio>__<error>__<n>.mp4 cuenta como "mal" para el modelo de técnica y,
#                además, entrena un segundo modelo que dice QUÉ error es (con su mensaje).
EJERCICIOS = {
    "sentadilla": {
        "nombre": "Sentadilla",
        "angulo": "rodilla",
        "mendeley": "squat",
        "etiquetas": {
            "bien": "Buena tecnica",
            "mal": "Corrige: baja con control y manten la espalda recta",
        },
        "errores": {
            "poco_profunda": "Baja mas: muslos al menos paralelos al piso",
            "torso_inclinado": "Pecho arriba: no inclines tanto el torso",
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
        "errores": {
            "cadera_caida": "Aprieta abdomen: la cadera se cae",
            "poco_profunda": "Baja mas: codo hasta ~90 grados",
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
        "errores": {
            "balanceo": "Sin impulso: no balancees el torso",
            "recorrido_parcial": "Recorrido completo: extiende y sube del todo",
        },
    },
}


def clasificar_etiqueta(ejercicio, etiqueta):
    """Etiqueta del nombre del archivo -> (etiqueta bien/mal, error específico o None).
    Devuelve None si la etiqueta no es válida para ese ejercicio."""
    cfg = EJERCICIOS.get(ejercicio)
    if cfg is None:
        return None
    if etiqueta in cfg["etiquetas"]:
        return etiqueta, None
    if etiqueta in cfg.get("errores", {}):
        return "mal", etiqueta
    return None


# Reconocimiento del ejercicio: ventana deslizante de ángulos, medida en SEGUNDOS para que
# funcione igual con videos de 15 o 30 fps y con la cámara de Colab (pocos cuadros por segundo)
VENTANA_S = 2.0         # duración de cada ventana
PASO_S = 0.33           # cada cuánto se toma una ventana al entrenar / se predice en vivo
MIN_FRAMES_VENTANA = 8  # una ventana con menos cuadros válidos no se usa
SUAVIZADO = 5           # predicciones que se votan en vivo para que no "parpadee" el ejercicio
MIN_REPS_SERIE = 3      # reps necesarias para dar el veredicto de la serie (promedio de p(mal) de sus reps)
PROB_MIN_EJERCICIO = 0.6
UMBRAL_MOVIMIENTO = 6.0  # desviación estándar (grados) mínima de codo o rodilla para considerar que hay ejercicio

# Conteo de repeticiones con el ángulo principal
DELTA_INICIO = 25.0     # la rep empieza cuando el ángulo baja esto (grados) desde el máximo reciente
RECUPERACION = 0.7      # y termina cuando recupera esta fracción del recorrido (del mínimo al máximo)
MIN_DUR_REP = 0.5       # segundos; reps más cortas se descartan (ruido)
MAX_DUR_REP = 10.0      # segundos; si una rep tarda más, se reinicia el conteo
TOLERANCIA_INICIO = 5.0  # la rep empieza en el último cuadro a menos de esto (grados) del máximo: así no
                         # incluye la pausa arriba (con el temblor, el máximo puede quedar segundos antes)

VIS_MIN = 0.5           # visibilidad mínima de los puntos del lado que se usa
ANCHO_POSE = 640        # la pose se detecta con el cuadro reducido a este ancho, igual al entrenar y al analizar

# Señal por cuadro (pose.ProcesadorPose)
LADO_FIJO = True        # elegir el lado del cuerpo con histéresis (False = el más visible en cada cuadro, como en v1)
ALFA_LADO = 0.1         # suavizado exponencial de la visibilidad de cada lado
HISTERESIS_LADO = 0.5   # el otro lado tiene que verse esto mejor (suma de 6 visibilidades) para cambiar
MEDIANA_FRAMES = 5      # mediana móvil causal de las señales contra el temblor de MediaPipe (1 = sin suavizado)

# Rasgos
ESTADISTICOS = "percentiles"           # "percentiles" (p5/p95, robustos) o "minmax" (como en v1)
# Familias de rasgos por repetición (ver pose.nombres_rasgos_rep). Por defecto "basicos" (los de v1):
# con el dataset real, en la comparación pareada por persona, sumar "estad", "fase" y "flex" (47 rasgos)
# no mejoró y en algunos modelos empeoró (con ~20 personas, más rasgos = más ruido). Ver HALLAZGOS.md.
RASGOS_REP = ("basicos",)
PUNTOS_SECUENCIA = 64                   # una rep se remuestrea a este número de puntos para ROCKET

# Entrenamiento y evaluación (entrenar.py, experimentos.py, evaluar_sistema.py)
VISTAS_ENTRENAR = ("side",)  # vistas del dataset de Mendeley con las que se entrena (los videos propios cuentan como side)
PASO_ENTRENAR_S = 1.0   # paso entre ventanas al entrenar: con 0.33 s las ventanas se traslapan 83 % y casi se repiten
FRACCION_TEST = 0.2     # personas que se apartan como prueba final (no se usan para elegir nada)
MIN_PERSONAS_TEST = 10  # con menos personas no se aparta prueba (solo validación cruzada)
FOLDS_EXTERNOS = 5      # validación cruzada externa (estima el desempeño)
FOLDS_INTERNOS = 3      # validación cruzada interna (hiperparámetros, calibración y umbral)
REPETICIONES_CV = 2     # repeticiones de la CV externa con distinta partición
# Umbral de p(mal). None = la decisión del propio modelo (class_weight="balanced"); es lo que mejor
# funcionó con el dataset real. Alternativas (se elige con la CV interna):
#   "acierto_bal": maximiza el acierto balanceado (J de Youden).
#   "f2": maximiza F2 (no detectar una rep mal pesa más). OJO: con un modelo débil el óptimo de F2
#         es decir "mal" a todo (recall 1, F2 = 0.83) y el F1 macro se desploma.
# Con ~20 personas, ambos umbrales salieron inestables y bajaron el F1 macro entre 0.02 y 0.15.
CRITERIO_UMBRAL = None
SEMILLA = 0
