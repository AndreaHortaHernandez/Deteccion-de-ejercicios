# Detección de ejercicios y técnica

Práctica 3 · Machine Learning · 3.er parcial

Reconoce tres ejercicios (**sentadilla**, **lagartija** y **curl de bíceps**), cuenta las repeticiones y avisa si cada una está **bien** o **mal** hecha. Corre en **Google Colab**, así que no hace falta una computadora potente.

## Cómo funciona

```
video / cámara ──► MediaPipe Pose (33 puntos) ──► 6 ángulos por cuadro
                                                     │
                     ┌───────────────────────────────┴──────────────────────────┐
                     ▼                                                          ▼
     Ventanas de 2 s (media, desv., mín., máx.)              Corte de repeticiones con el ángulo principal
                     │                                       (rodilla o codo: baja ≥ 25° y recupera 70 %)
                     ▼                                                          ▼
       Modelo de EJERCICIO                                  Rasgos por repetición (mín., máx., media,
   (sentadilla / lagartija / curl)                           desv., rango de cada ángulo + duración)
                                                                                ▼
                                                   Modelo de TÉCNICA del ejercicio detectado
                                                                (bien / mal)
```

- **Ángulos** (del lado del cuerpo que mejor se ve): rodilla, cadera, codo, hombro, inclinación del torso respecto a la vertical y línea hombro-cadera-tobillo. Al usar ángulos en lugar de píxeles, el modelo no depende de la distancia a la cámara, del lado ni de la estatura.
- **Ventanas por tiempo** (2 s cada 0.33 s), no por número de cuadros, para que funcione igual con videos de 15 o 30 fps y con la cámara de Colab, que va más lenta.
- **Modelos:** para cada problema se comparan **Random Forest**, **SVM (RBF)** y **MLP** con validación cruzada agrupada **por persona** (`GroupKFold`): la métrica indica qué tan bien funciona con gente que el modelo nunca vio. Se guarda el de mejor F1 macro.

| Ejercicio | Ángulo para contar repeticiones |
|---|---|
| Sentadilla | rodilla |
| Lagartija | codo |
| Curl de bíceps | codo |

## Datos

Se entrena con un dataset público:

> Duddela Sai Prashanth, Priya Kamath, Rakshith Bhandary, S Ashlesh Pai, Pratham PN, Advith P, Abhishek Mashetty (2026). *A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels*. Mendeley Data, v2. https://doi.org/10.17632/kgbb3yn47p.2 — Licencia CC BY 4.0.

26 personas grabadas con celular en un gimnasio, 7 ejercicios, cada uno con forma buena y mala, desde 3 vistas. Usamos los 3 ejercicios del proyecto en vista de **perfil**: ~148 videos (~512 MB). `importar_mendeley.py` los descarga directo de la API pública de Mendeley; no hace falta bajar el ZIP a mano.

**Videos propios (opcional):** se pueden agregar a la misma carpeta para tener más variedad:

```
videos/apellido_nombre/sentadilla__bien__1.mp4
videos/apellido_nombre/curl__mal__1.mp4
```

Nombre: `ejercicio__etiqueta__n.mp4` (doble guion bajo), con ejercicio `sentadilla`, `lagartija` o `curl` y etiqueta `bien` o `mal`. Grábenlos de perfil, con el cuerpo completo en cuadro, cámara fija y una sola persona. En un video `mal`, todas las repeticiones deben estar mal hechas.

## Correrlo en Colab

Abre [`Deteccion_ejercicios_Colab.ipynb`](https://colab.research.google.com/github/AndreaHortaHernandez/Deteccion-de-ejercicios/blob/main/Deteccion_ejercicios_Colab.ipynb) y corre las celdas en orden:

1. Instala MediaPipe y clona este repo.
2. Conecta Google Drive (los datos y modelos se guardan en `MyDrive/deteccion_ejercicios/`).
3. Descarga el dataset de Mendeley.
4. Extrae la pose de los videos (solo procesa los nuevos).
5. Entrena y evalúa: reporte y matrices de confusión en `salidas/`.
6. Analiza un video de prueba y lo muestra con el resultado dibujado.
7. Cámara en vivo desde el navegador (unos cuantos cuadros por segundo).

## Correrlo en una computadora (opcional)

Con Python 3.10 a 3.14:

```bash
pip install -r requirements.txt
pip install --no-deps mediapipe==1.2.2      # sin --no-deps instala otra copia de OpenCV

python importar_mendeley.py --videos videos
python extraer_videos.py --videos videos
python entrenar.py
python analizar_video.py --video prueba.mp4
python en_vivo.py                            # webcam local; ESC para salir, R para reiniciar
```

Sin argumentos usa las carpetas `datos/`, `modelos/` y `salidas/` del repo. El modelo de pose de MediaPipe (~9 MB) se descarga solo la primera vez.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Ejercicios, etiquetas, mensajes y parámetros (ventana, conteo de repeticiones). |
| `pose.py` | Detector de pose, ángulos, rasgos, conteo de repeticiones y el `Analizador` que junta todo. |
| `importar_mendeley.py` | Descarga el dataset de Mendeley con los nombres del proyecto. |
| `extraer_videos.py` | Videos → CSV con los 33 puntos por cuadro. |
| `entrenar.py` | Entrena y evalúa los modelos de ejercicio y de técnica. |
| `analizar_video.py` | Analiza un video y genera otro con el resultado dibujado. |
| `colab_camara.py` | Cámara en vivo dentro de Colab. |
| `en_vivo.py` | Cámara en vivo en una computadora. |

## Limitaciones

- El dataset solo dice si la repetición está bien o mal, no qué error tiene. El mensaje de "mal" da una indicación general por ejercicio.
- La etiqueta es por video: se asume que todas las repeticiones de un video `mal` están mal.
- En el curl alternado del dataset, de perfil solo se ve bien un brazo, así que se cuentan las repeticiones de ese brazo.
- El corte de repeticiones es por reglas (umbrales en `config.py`), no aprendido.

## Mejoras posibles

- **Decir qué error tiene la repetición** (cadera caída, poco profunda, balanceo…): cambiar `"mal"` por errores específicos en `config.py` y grabar videos de cada error. El resto del código ya lo soporta.
- Usar también la vista diagonal del dataset (`--vistas side,diagonal`) para que funcione con la cámara menos de perfil.
- Reemplazar los rasgos resumidos por una red que lea la secuencia completa de ángulos de cada repetición (CNN 1D o LSTM).
- Buscar hiperparámetros con `GridSearchCV` anidado.
