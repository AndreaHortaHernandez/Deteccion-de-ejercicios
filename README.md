# Detección de ejercicios y técnica

Práctica 3 · Machine Learning · 3.er parcial

Reconoce tres ejercicios (**sentadilla**, **lagartija** y **curl de bíceps**), cuenta las repeticiones y avisa si cada una está **bien** o **mal** hecha. Corre en **Google Colab**, así que no hace falta una computadora potente.

## Cómo funciona

```
video / cámara ──► MediaPipe Pose (33 puntos) ──► 8 señales por cuadro
                                                     │
                     ┌───────────────────────────────┴──────────────────────────┐
                     ▼                                                          ▼
     Ventanas de 2 s (media, desv., p5, p95)                 Corte de repeticiones con el ángulo principal
                     │                                       (rodilla o codo: baja ≥ 25° y recupera 70 %)
                     ▼                                                          ▼
       Modelo de EJERCICIO                                  Rasgos por repetición (mín., máx., media,
   (sentadilla / lagartija / curl)                           desv., rango de cada ángulo + duración)
                                                                                ▼
                                                   Modelo de TÉCNICA del ejercicio detectado
                                                                (bien / mal)
```

- **Señales:** 6 ángulos (rodilla, cadera, codo, hombro, inclinación del torso respecto a la vertical y línea hombro-cadera-tobillo) y 2 medidas normalizadas por la longitud del segmento: profundidad de la cadera respecto a la rodilla y rodilla delante de la punta del pie. Al usar ángulos y proporciones en lugar de píxeles, el modelo no depende de la distancia a la cámara, del lado ni de la estatura.
- **Preprocesamiento** (`pose.ProcesadorPose`, igual al entrenar y en vivo): el lado del cuerpo se fija con histéresis (para que los ángulos no salten de un brazo al otro); una mediana móvil causal de 5 cuadros quita el temblor de MediaPipe; si la rodilla o el tobillo no se ven, las señales de la pierna quedan vacías (NaN) en lugar de meter basura.
- **Corte de repeticiones:** la rep empieza en el último cuadro cerca del máximo, justo antes de bajar. Antes empezaba en el máximo, que con el temblor podía quedar segundos atrás (en la pausa de arriba); las reps pasaban de 10 s y el contador se reiniciaba: en lagartija se perdían ~60 repeticiones del dataset.
- **Ventanas por tiempo** (2 s), no por número de cuadros, para que funcione igual con videos de 15 o 30 fps y con la cámara de Colab, que va más lenta. Al entrenar se toma una cada 1 s (con 0.33 s se traslapaban 83 % y casi se repetían); en vivo, una cada 0.33 s.
- **Rasgos de técnica:** por defecto, mínimo, máximo, media, desviación y rango de los 6 ángulos más la duración de la rep (31 rasgos, los mismos de la versión 1). También están implementadas tres familias más (estadísticas robustas de las 8 señales, fase de la repetición y valores en el punto de máxima flexión), pero con el dataset real no mejoraron y en algunos modelos empeoraron, así que quedaron como opción (`config.RASGOS_REP`) y como ablación.

## Evaluación

Las ventanas y repeticiones de una persona no son independientes: la unidad real es la **persona** (26) y la etiqueta es por **video**. Todo el protocolo parte de eso.

1. **Prueba final:** antes de todo se aparta el 20 % de las personas. No se usa para elegir nada y se evalúa una sola vez.
2. **Validación cruzada anidada** (`StratifiedGroupKFold` por persona): la externa (5 folds × 2 repeticiones) estima el desempeño; la interna elige hiperparámetros y **calibra** las probabilidades (regresión logística sobre predicciones fuera de fold, descartada si invierte o empeora las predicciones). La decisión es la del propio modelo con clases balanceadas; un umbral ajustado (F2 o acierto balanceado) quedó como opción en `config.CRITERIO_UMBRAL` porque con los datos reales resultó inestable. Lo que se evalúa es exactamente la regla que se usa en vivo (`modelos.ModeloCalibrado`).
3. **Candidatos**, de lo más simple a lo más complejo: `Dummy`, `Regla_umbral` (una regla "si rasgo < x" aprendida), regresión logística, Random Forest, HistGradientBoosting, SVM RBF, MLP y, para técnica, **ROCKET** (convoluciones aleatorias sobre la secuencia completa de la rep).
4. **Métricas:** F1 macro, acierto balanceado, MCC y, para técnica, recall/precisión/F2 de `mal`, PR-AUC, Brier y **acierto pareado** (¿el video mal de cada persona sale con más p(mal) que su video bien?). Media ± DE entre folds, IC 95 % por bootstrap de personas, métricas por video y Wilcoxon pareado por persona (con Holm) para saber si la diferencia entre modelos es real.
5. **Sistema completo** (`evaluar_sistema.py`): cada video de prueba pasa por el analizador real; se mide el ejercicio por video, el conteo de repeticiones (contra el conteo real si lo anotan) y la técnica por video.
6. **Experimentos** (`experimentos.py`): comparación pareada v1 vs. v2, ablaciones, curva de aprendizaje por número de personas y robustez a la vista diagonal.
7. **Veredicto por serie:** en vivo, además de cada rep, se muestra el veredicto de la serie (promedio de p(mal) de sus reps, a partir de 3). Con el dataset real decidir por serie acierta ~0.70 contra ~0.60 rep por rep.

Los resultados con el dataset real y lo que significan están en [HALLAZGOS.md](HALLAZGOS.md).

| Ejercicio | Ángulo para contar repeticiones |
|---|---|
| Sentadilla | rodilla |
| Lagartija | codo |
| Curl de bíceps | codo |

## Datos

Se entrena con un dataset público:

> Duddela Sai Prashanth, Priya Kamath, Rakshith Bhandary, S Ashlesh Pai, Pratham PN, Advith P, Abhishek Mashetty (2026). *A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels*. Mendeley Data, v2. https://doi.org/10.17632/kgbb3yn47p.2 — Licencia CC BY 4.0.

26 personas grabadas con celular en un gimnasio, 7 ejercicios, cada uno con forma buena y mala, desde 3 vistas. Usamos los 3 ejercicios del proyecto en vista de **perfil**: ~148 videos (~512 MB). `importar_mendeley.py` los descarga directo de la API pública de Mendeley; no hace falta bajar el ZIP a mano.

**Videos propios con errores específicos:** es la mejora que más mueve el resultado. Cómo grabarlos (cámara, ejercicios, errores, nombres y conteo de repeticiones) está en [PROTOCOLO_GRABACION.md](PROTOCOLO_GRABACION.md). Van en la misma carpeta:

```
videos/apellido_nombre/sentadilla__bien__1.mp4
videos/apellido_nombre/sentadilla__poco_profunda__1.mp4
videos/apellido_nombre/curl__balanceo__1.mp4
```

Nombre: `ejercicio__etiqueta__n.mp4` (doble guion bajo). La etiqueta es `bien`, `mal` o uno de los errores de `config.py`:

| Ejercicio | Errores |
|---|---|
| `sentadilla` | `poco_profunda`, `torso_inclinado` |
| `lagartija` | `cadera_caida`, `poco_profunda` |
| `curl` | `balanceo`, `recorrido_parcial` |

Un video con error cuenta como `mal` para el modelo de técnica y además entrena un modelo que dice **qué** error es, para que en vivo salga el mensaje específico ("Baja mas...", "Aprieta abdomen..."). Se necesitan al menos 2 errores grabados por 2 o más personas. `extraer_videos.py` crea `datos/conteos_reales.csv` con una fila por video para anotar las repeticiones reales.

## Correrlo en Colab

Abre [`Deteccion_ejercicios_Colab.ipynb`](https://colab.research.google.com/github/AndreaHortaHernandez/Deteccion-de-ejercicios/blob/main/Deteccion_ejercicios_Colab.ipynb) y corre las celdas en orden:

1. Instala MediaPipe y clona este repo.
2. Conecta Google Drive (los datos y modelos se guardan en `MyDrive/deteccion_ejercicios/`).
3. Descarga el dataset de Mendeley.
4. Extrae la pose de los videos (solo procesa los nuevos).
5. Entrena y evalúa: reporte, `resultados.csv` y gráficos en `salidas/`.
6. Evalúa el sistema completo con las personas de prueba.
7. Experimentos para el reporte: ablaciones, curva de aprendizaje y vista diagonal.
8. Analiza un video de prueba y lo muestra con el resultado dibujado.
9. Cámara en vivo desde el navegador (unos cuantos cuadros por segundo).

## Correrlo en una computadora (opcional)

Con Python 3.10 a 3.14:

```bash
pip install -r requirements.txt
pip install --no-deps mediapipe==1.2.2      # sin --no-deps instala otra copia de OpenCV

python importar_mendeley.py --videos videos
python extraer_videos.py --videos videos
python entrenar.py                           # --rapido para una corrida de prueba
python evaluar_sistema.py
python experimentos.py
python analizar_video.py --video prueba.mp4
python en_vivo.py                            # webcam local; ESC para salir, R para reiniciar
```

Sin argumentos usa las carpetas `datos/`, `modelos/` y `salidas/` del repo. El modelo de pose de MediaPipe (~9 MB) se descarga solo la primera vez.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Ejercicios, etiquetas, mensajes y parámetros (señal, rasgos, ventana, conteo, evaluación). |
| `pose.py` | Detector de pose, señales, rasgos, conteo de repeticiones y el `Analizador` que junta todo. |
| `importar_mendeley.py` | Descarga el dataset de Mendeley con los nombres del proyecto. |
| `extraer_videos.py` | Videos → CSV con los 33 puntos por cuadro. |
| `modelos.py` | Candidatos, `ModeloCalibrado` (hiperparámetros + calibración) y ROCKET. |
| `evaluacion.py` | Validación cruzada anidada, métricas, intervalos de confianza, Wilcoxon y gráficos. |
| `entrenar.py` | Compara candidatos, mide al ganador en la prueba final y guarda los modelos (incluido el de tipo de error si hay videos propios). |
| `evaluar_sistema.py` | Evalúa el sistema completo (como en vivo) con las personas de prueba. |
| `experimentos.py` | Comparación v1 vs. v2, ablaciones, curva de aprendizaje y robustez al punto de vista. |
| `HALLAZGOS.md` | Resultados con el dataset real, diagnóstico y recomendaciones. |
| `PROTOCOLO_GRABACION.md` | Cómo grabar videos propios con errores específicos y conteo real. |
| `analizar_video.py` | Analiza un video y genera otro con el resultado dibujado. |
| `colab_camara.py` | Cámara en vivo dentro de Colab. |
| `en_vivo.py` | Cámara en vivo en una computadora. |

## Limitaciones

- El dataset de Mendeley solo dice si el video está bien o mal, no qué error tiene. El tipo de error sale solo de los videos propios; sin ellos, el mensaje de "mal" es general por ejercicio.
- La etiqueta es por video: se asume que todas las repeticiones de un video `mal` están mal.
- En el curl alternado del dataset, de perfil solo se ve bien un brazo, así que se cuentan las repeticiones de ese brazo.
- El corte de repeticiones es por reglas (umbrales en `config.py`), no aprendido. `evaluar_sistema.py` mide su error y su sensibilidad a los umbrales si se anota el conteo real.
- Con 26 personas, la prueba final tiene ~5: su intervalo de confianza es ancho. El reporte lo muestra; no hay que leer el número solo.

## Mejoras posibles

- Usar también la vista diagonal del dataset (`--vistas side,diagonal`) para que funcione con la cámara menos de perfil.
- Probar una red sobre la secuencia de cada repetición (CNN 1D o LSTM) contra ROCKET, si se consiguen más datos.
