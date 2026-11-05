# Detección de ejercicios y técnica

Práctica 3 · Machine Learning · 3.er parcial

Reconoce tres ejercicios (**sentadilla**, **lagartija** y **curl de bíceps**), cuenta las repeticiones y dice si cada una está **bien hecha** o qué **error** tiene. Corre en **Google Colab**, así que no hace falta una computadora potente.

## Cómo funciona

```
video / cámara ──► MediaPipe Pose (33 puntos) ──► 6 ángulos por cuadro
                                                     │
                     ┌───────────────────────────────┴──────────────────────────┐
                     ▼                                                          ▼
     Ventanas de ~1 s (media, desv., mín., máx.)            Corte de repeticiones con el ángulo principal
                     │                                       (rodilla o codo: baja ≥ 25° y regresa)
                     ▼                                                          ▼
       Modelo de EJERCICIO                                  Rasgos por repetición (mín., máx., media,
   (sentadilla / lagartija / curl)                           desv., rango de cada ángulo + duración)
                                                                                ▼
                                                   Modelo de TÉCNICA del ejercicio detectado
                                                       (bien / tipo de error)
```

Ángulos usados (del lado del cuerpo que mejor se ve): rodilla, cadera, codo, hombro, inclinación del torso respecto a la vertical y línea hombro-cadera-tobillo.

Para cada modelo se comparan **Random Forest**, **SVM (RBF)** y **MLP** con validación cruzada agrupada (por persona si hay 3 o más; si no, por video) y se guarda el de mejor F1 macro. Agrupar evita que cuadros del mismo video queden en entrenamiento y prueba a la vez.

| Ejercicio | Ángulo para contar | Etiquetas |
|---|---|---|
| Sentadilla | rodilla | `bien`, `poco_profunda`, `espalda_inclinada` |
| Lagartija | codo | `bien`, `cadera_caida`, `poco_profunda` |
| Curl de bíceps | codo | `bien`, `balanceo`, `rango_incompleto` |

## 1. Grabar los videos (con el celular)

**Cada integrante** graba un video por combinación de ejercicio y etiqueta, **mejor 2**, con **8 a 10 repeticiones** cada uno. Son 9 combinaciones → 18 videos de ~30 s por persona.

- **De perfil** a la cámara, con el **cuerpo completo** en el cuadro (en el curl basta de la cabeza a las rodillas).
- Cámara **fija** (tripié o recargada), en **horizontal**, buena luz y **una sola persona** en cuadro.
- En los videos de error, **todas** las repeticiones deben tener ese error y solo ese.
- Varíen un poco el fondo, la ropa y el lado (izquierdo o derecho) entre videos.

Cómo hacer cada error:

| Video | Qué hacer |
|---|---|
| `sentadilla__bien` | Bajar hasta que el muslo quede paralelo al piso, espalda recta. |
| `sentadilla__poco_profunda` | Bajar solo un poco (rodilla a ~120°). |
| `sentadilla__espalda_inclinada` | Bajar llevando el pecho muy hacia el frente. |
| `lagartija__bien` | Cuerpo recto de hombros a tobillos, bajar hasta ~90° de codo. |
| `lagartija__cadera_caida` | Dejar caer la cadera hacia el piso. |
| `lagartija__poco_profunda` | Bajar solo un poco. |
| `curl__bien` | Codo pegado al cuerpo, subir completo y bajar completo. |
| `curl__balanceo` | Ayudarse con el torso o adelantar el codo. |
| `curl__rango_incompleto` | Subir solo hasta la mitad. |

Graben también algunos videos **mezclando ejercicios** para probar al final; esos van en `pruebas/` y no se usan para entrenar.

## 2. Subirlos a Google Drive

```
MyDrive/deteccion_ejercicios/
  videos/
    apellido_nombre/
      sentadilla__bien__1.mp4
      sentadilla__bien__2.mp4
      lagartija__cadera_caida__1.mp4
      ...
  pruebas/
    prueba1.mp4
```

El nombre del archivo es `ejercicio__etiqueta__número` con **doble guion bajo**. Si un nombre está mal, el notebook lo avisa y se salta ese video.

## 3. Correr el notebook en Colab

Abre [`Deteccion_ejercicios_Colab.ipynb`](Deteccion_ejercicios_Colab.ipynb) en Colab (en GitHub: botón *Open in Colab*, o en Colab: *Archivo → Abrir cuaderno → GitHub* y pega la URL del repo) y corre las celdas en orden:

1. Instala MediaPipe y clona este repo.
2. Conecta Google Drive.
3. Extrae la pose de los videos (solo procesa los nuevos).
4. Entrena y evalúa: reporte y matrices de confusión en `salidas/`.
5. Analiza un video de prueba y lo muestra con el resultado dibujado.
6. Cámara en vivo desde el navegador (unos cuantos cuadros por segundo).

## Correrlo en una computadora (opcional)

Con Python 3.10 a 3.14:

```bash
pip install opencv-python numpy scikit-learn matplotlib joblib flatbuffers
pip install --no-deps mediapipe==1.2.2      # sin --no-deps instala otra copia de OpenCV

python extraer_videos.py --videos ruta/a/videos
python entrenar.py
python analizar_video.py --video prueba.mp4
python en_vivo.py                            # webcam local; ESC para salir, R para reiniciar
```

Sin argumentos usa las carpetas `datos/`, `modelos/` y `salidas/` del repo. El modelo de pose de MediaPipe se descarga solo la primera vez.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Ejercicios, etiquetas, mensajes y parámetros (ventana, umbrales de repetición). |
| `pose.py` | Detector de pose, ángulos, rasgos, conteo de repeticiones y el `Analizador` que junta todo. |
| `extraer_videos.py` | Videos → CSV con los 33 puntos por cuadro. |
| `entrenar.py` | Entrena y evalúa los modelos de ejercicio y de técnica. |
| `analizar_video.py` | Analiza un video y genera otro con el resultado dibujado. |
| `colab_camara.py` | Cámara en vivo dentro de Colab. |
| `en_vivo.py` | Cámara en vivo en una computadora. |

## Ajustes

- Si no cuenta repeticiones muy cortas o cuenta de más, ajusta `DELTA_INICIO` y `DELTA_FIN` en `config.py`.
- Si el ejercicio "parpadea" en vivo, sube `SUAVIZADO` o `PROB_MIN_EJERCICIO`.
- Para agregar otro error, agrégalo en `etiquetas` del ejercicio en `config.py`, graba sus videos y vuelve a entrenar.
