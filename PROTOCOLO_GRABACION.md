# Protocolo de grabación de videos propios

## Para qué

Con el dataset de Mendeley el modelo de técnica no pasa de ~0.65 de F1 con gente nueva. El límite
son las etiquetas: "mal" es genérico, se asigna por video y a mano, sin criterios (ver
[HALLAZGOS.md](HALLAZGOS.md)). Estos videos agregan lo que falta:

1. **Errores específicos y consistentes.** El sistema aprende a decir *qué* error es
   ("baja más", "aprieta abdomen"), no solo "mal".
2. **El conteo real de repeticiones**, para medir el error del contador y ajustar sus umbrales
   con evidencia.
3. **Más personas** para el modelo bien/mal: cada video con error cuenta como "mal".

**Tiempo:** unos 20 min por persona (9 videos de ~40 s más la preparación). Con 5 personas, unas 2 horas.

## 1. Personas

- **Mínimo 4, ideal 6 o más.** Importa más el número de personas que el de videos por persona: la
  evaluación agrupa por persona, así que 6 personas con 9 videos valen más que 3 personas con 18.
- Variedad de estatura, complexión y sexo, si se puede.
- **Consentimiento:** cada persona acepta, idealmente por escrito, que se graben y usen sus videos
  para la práctica. Los videos se quedan en el Drive privado: `videos/` está en `.gitignore`, así
  que no se suben al repositorio. En el reporte usen cuadros con el esqueleto dibujado o difuminen
  la cara.
- **Seguridad:** calentamiento breve, peso ligero en el curl y errores *moderados*. Que nadie se
  fuerce: si algo duele, se para. Si alguien no puede hacer 5 lagartijas completas, se salta ese
  ejercicio (anótenlo).

## 2. Cámara y lugar

| Qué | Cómo |
|---|---|
| Orientación | **Horizontal** (apaisada) para los 3 ejercicios. La lagartija necesita el ancho. |
| Vista | **Perfil estricto**: la cámara perpendicular a la persona, como en la vista *side* de Mendeley. Todos con el **mismo lado** hacia la cámara (el izquierdo, como la mayoría de Mendeley). Los ángulos no cambian con el lado, pero así es más parejo. |
| Altura | A la altura de la cadera (~80–100 cm), sobre un tripié o apoyada en algo fijo. **No a mano.** |
| Distancia | 2.5–4 m: el cuerpo completo, de la cabeza a los pies, con margen en todo el movimiento. Para la lagartija, revisen que entre de la cabeza a los talones. |
| Calidad | 720p o 1080p a 30 fps. Sin cámara lenta, sin filtros, sin zoom digital. |
| Luz | De frente o de lado. **Nada de contraluz** (una ventana detrás de la persona). |
| Fondo | Liso, sin espejos y **sin otras personas en cuadro**. |
| Ropa | Ajustada y que contraste con el fondo. Nada holgado: MediaPipe pierde las articulaciones. |

Antes de empezar, graben 5 segundos de prueba y revisen que se vea todo el cuerpo.

## 3. Qué grabar

Por persona, **9 videos**: los 3 ejercicios con la técnica correcta y con 2 errores cada uno.

Cada video lleva:
- **8–10 repeticiones** a ritmo normal (~2 s por rep);
- **2 segundos quieto** al principio y al final (de pie, o en posición de plancha en la lagartija);
- **un archivo por video**: se detiene la grabación entre uno y otro;
- en un video con error, **todas las repeticiones llevan ese error**, y solo ese.

### Sentadilla (`sentadilla`)
| Etiqueta | Cómo se hace | Qué cambia en las señales |
|---|---|---|
| `bien` | Pies al ancho de hombros, bajar hasta muslos paralelos al piso o más (rodilla ≤ ~90°), espalda neutra, talones en el piso | — |
| `poco_profunda` | Igual, pero bajar solo hasta la mitad (rodilla ~120–130°) | Rodilla mínima, profundidad de cadera |
| `torso_inclinado` | Profundidad normal, pero llevar el pecho hacia adelante (torso casi horizontal) | Inclinación del torso, ángulo de cadera |

### Lagartija (`lagartija`)
| Etiqueta | Cómo se hace | Qué cambia en las señales |
|---|---|---|
| `bien` | Cuerpo en línea recta de la cabeza a los talones, manos bajo los hombros, bajar hasta codo ~90° | — |
| `cadera_caida` | Profundidad normal, pero con el abdomen flojo: la cadera se cae (arco en la espalda baja) | Línea hombro-cadera-tobillo, ángulo de cadera |
| `poco_profunda` | Cuerpo recto, pero bajar solo un tercio (codo ~130°) | Codo mínimo, recorrido |

### Curl de bíceps (`curl`)
Mancuerna ligera (2–5 kg) o una botella con agua o arena. **Los dos brazos al mismo tiempo** (no
alternado), para que el brazo del lado de la cámara haga todas las repeticiones.

| Etiqueta | Cómo se hace | Qué cambia en las señales |
|---|---|---|
| `bien` | Codos pegados al torso, torso quieto, extender casi del todo abajo y subir completo | — |
| `balanceo` | Echar el torso hacia atrás para subir con impulso (~15–20°) | Inclinación del torso, ángulo de hombro |
| `recorrido_parcial` | Solo la mitad del movimiento: no extender abajo ni subir del todo | Recorrido del codo |

### Orden
Para que el cansancio no se confunda con la técnica, **cambien el orden entre personas**:

| Persona | Orden dentro de cada ejercicio |
|---|---|
| 1, 4 | bien → error 1 → error 2 |
| 2, 5 | error 1 → error 2 → bien |
| 3, 6 | error 2 → bien → error 1 |

## 4. Nombres de archivo

```
videos/<apellido_nombre>/<ejercicio>__<etiqueta>__<n>.mp4
```
En minúsculas, sin acentos ni espacios, **con doble guion bajo**. `<n>` es el número de toma (1, o
2 si repiten). Los 9 archivos de una persona:

```
videos/perez_ana/sentadilla__bien__1.mp4
videos/perez_ana/sentadilla__poco_profunda__1.mp4
videos/perez_ana/sentadilla__torso_inclinado__1.mp4
videos/perez_ana/lagartija__bien__1.mp4
videos/perez_ana/lagartija__cadera_caida__1.mp4
videos/perez_ana/lagartija__poco_profunda__1.mp4
videos/perez_ana/curl__bien__1.mp4
videos/perez_ana/curl__balanceo__1.mp4
videos/perez_ana/curl__recorrido_parcial__1.mp4
```

Los errores válidos están en `config.py` (`EJERCICIOS[...]["errores"]`). Si agregan otro error, por
ejemplo `curl__codo_adelante`, primero agréguenlo ahí con su mensaje.

## 5. Contar las repeticiones

Mientras una persona hace el ejercicio, **otra cuenta las repeticiones** y las anota en la hoja
(abajo). Después se pasan a `datos/conteos_reales.csv`: `extraer_videos.py` lo crea con una fila
vacía por video, y solo hay que llenar la columna `reps_reales`.

Cuenta como repetición cada bajada y subida completa *según la etiqueta del video*: en
`poco_profunda` cuenta cada bajada corta.

Vale la pena contar también ~20 videos de Mendeley (viéndolos en Drive): son pocos minutos y
duplican los datos para medir el contador.

## 6. Después de grabar

1. Suban los videos a `MyDrive/deteccion_ejercicios/videos/<apellido_nombre>/`.
2. Notebook, paso 4 (**extraer pose**). Revisen al final:
   - "video(s) con problemas": casi siempre es un nombre mal escrito;
   - "[!] en N frames no se vio a la persona": si es más del 10 %, revisen el encuadre o
     regraben ese video.
3. Llenen `reps_reales` en `datos/conteos_reales.csv`.
4. Paso 5 (**entrenar**). En `reporte_entrenamiento.txt` aparece una sección nueva, **"Tipo de
   error <ejercicio>"**, con validación cruzada por persona y su matriz de confusión
   (`confusion_error_<ejercicio>_cv.png`). Con 4–6 personas no se aparta prueba final: se reporta
   la validación cruzada con su intervalo.
5. Paso 6 (**evaluar sistema**): error de conteo (MAE, % exacto) y mapa de sensibilidad de los
   umbrales del contador.
6. Revisen `linea_de_tiempo.png`: cada repetición marcada debe coincidir con una bajada real.

## 7. Qué reportar

- **Matriz de confusión del tipo de error** por ejercicio: qué errores se confunden entre sí.
- **Conteo:** MAE y % exacto contra el conteo real. Si el mapa de sensibilidad sugiere cambiar
  `DELTA_INICIO` o `RECUPERACION`, digan que se eligieron con esos mismos datos.
- **Limitaciones** que hay que decir:
  - errores *actuados* y algo exagerados, más fáciles que los naturales;
  - pocas personas y un solo lugar;
  - el mismo lado hacia la cámara.

## Hoja para imprimir

| Persona | Archivo | Orden | Reps contadas | Notas (¿se vio todo? ¿repetir?) |
|---|---|---|---|---|
| | sentadilla__bien__1 | | | |
| | sentadilla__poco_profunda__1 | | | |
| | sentadilla__torso_inclinado__1 | | | |
| | lagartija__bien__1 | | | |
| | lagartija__cadera_caida__1 | | | |
| | lagartija__poco_profunda__1 | | | |
| | curl__bien__1 | | | |
| | curl__balanceo__1 | | | |
| | curl__recorrido_parcial__1 | | | |
