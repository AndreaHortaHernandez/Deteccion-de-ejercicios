# Hallazgos con el dataset real

Resultados de correr la versión 1 (v1, commit `f47c907`) y la versión 2 (v2, este código) con los
mismos CSV de pose del dataset de Mendeley: 148 videos de perfil, 25 personas.

## Resumen

- **Reconocer el ejercicio es trivial**: F1 ≈ 0.998 en v1 y en v2, y una regla de un umbral llega a 0.992.
- **Clasificar la técnica (bien/mal) con gente nueva es difícil**: F1 ≈ 0.60–0.67 por repetición
  y ≈ 0.63–0.73 por video. v2 **no predice mejor que v1**: en la comparación pareada (mismos
  modelos, mismas particiones por persona) las diferencias son de ±0.04 y casi nunca significativas.
- Lo que v2 sí mejora:
  1. **Una estimación honesta del desempeño**, con prueba final, CV anidada, IC por personas,
     líneas base y métricas por video. Con v1 el mismo código daba 0.66 o 0.59 en sentadilla
     según la versión de scikit-learn (Colab vs. local): una sola partición varía ±0.07.
  2. **El conteo de repeticiones**: arreglamos un error que perdía reps (≈60 en lagartija).
  3. **Un veredicto por serie**, que acierta más que el veredicto rep por rep.
  4. **Encontramos dónde está el límite: en los datos**, no en el modelo.

## 1. ¿Coinciden con los datos sintéticos?

No. Con los datos sintéticos que usamos para probar el código, la técnica daba F1 0.97–1.00; con
los reales da 0.60–0.67. Los sintéticos tenían errores exagerados, sin ruido de etiqueta y con
poca variación entre personas. Sirvieron para comprobar que el código corre, pero **escondieron
dos fallas** que solo aparecieron con los datos reales (sección 4). El modelo de ejercicio sí
coincide (≈1.0 en ambos).

## 2. v1 contra v2 (comparación pareada)

Mismos modelos y mismas particiones por persona: 5 repeticiones × 5 folds con las 20 personas de
entrenamiento. "p" es Wilcoxon pareado del acierto balanceado por persona. Sale de
`experimentos.py` (sección 0).

| Técnica | Modelo | v1 F1 | v2 F1 | p |
|---|---|---|---|---|
| sentadilla | LogReg | 0.547 | 0.555 | 0.79 |
| sentadilla | Random Forest | 0.656 | 0.652 | 0.96 |
| sentadilla | SVM | 0.591 | 0.624 | 0.05 |
| lagartija | LogReg | 0.586 | 0.543 | 0.03 |
| lagartija | Random Forest | 0.612 | 0.599 | 0.84 |
| lagartija | SVM | 0.612 | 0.602 | 0.30 |
| curl | LogReg | 0.582 | 0.575 | 0.59 |
| curl | Random Forest | 0.587 | 0.569 | 0.17 |
| curl | SVM | 0.568 | 0.566 | 0.82 |

De 9 comparaciones, 2 dan p ≤ 0.05, una a favor de cada versión: es lo que se espera por azar.
**Ninguna de las dos versiones predice mejor que la otra.**

**Ablaciones** (sección 1 de `experimentos.py`): las 47 características nuevas de v2 (estadísticas
robustas de 8 señales, fase de la rep y valores en la máxima flexión) dieron **igual o peor** que
las 31 de v1 (Random Forest en sentadilla: 0.589 contra 0.697). Con unas 20 personas, más
características suman más ruido que señal. Por eso v2 usa por defecto las de v1
(`config.RASGOS_REP = ("basicos",)`); las otras quedaron como opción. El preprocesamiento de v2
(lado fijo, suavizado) es neutral para la técnica.

## 3. Resultado final de v2

CV anidada con 20 personas (5 × 2) y prueba final con 5 personas nunca vistas.

| Técnica | Modelo elegido | F1 por rep (IC 95 %) | F1 por video | Pareado | Prueba final: rep / video |
|---|---|---|---|---|---|
| sentadilla | Random Forest | 0.66 (0.53–0.77) | 0.72 | 0.79 | 0.42 / 0.45 |
| lagartija | SVM | 0.67 (0.54–0.79) | 0.73 | 0.82 | 0.78 / 0.80 |
| curl | ROCKET | 0.61 (0.56–0.72) | 0.63 | 0.83 | 0.83 / 0.90 |

"Pareado" es la fracción de personas cuyo video mal sale con más p(mal) que su video bien.

**Ojo con la prueba final**: con 5 personas varía muchísimo. La sentadilla da 0.42 (las 5 personas
son difíciles) y el curl 0.90 (son fáciles). El número de referencia es el de la CV con su
intervalo; la prueba final solo confirma que no hay fuga.

## 4. Fallas que encontramos al probar con datos reales (ya corregidas)

1. **El umbral por F2 colapsaba.** Con un modelo débil, el óptimo de F2 es decir "mal" a todo
   (recall 1, F2 = 0.83), y el F1 caía a 0.34. Ahora la decisión es la del modelo con clases
   balanceadas; el umbral quedó como opción (`config.CRITERIO_UMBRAL`).
2. **La calibración a veces invertía las predicciones.** Con unas 13 personas en la CV interna y
   una señal débil, la regresión logística de calibración salía con pendiente negativa (había
   folds con F1 de 0.27). Ahora se descarta si invierte o empeora las predicciones.
3. **El corte de repeticiones perdía reps.** La rep empezaba en el último máximo, que con el
   temblor podía quedar segundos antes, en la pausa de arriba. Entonces la rep pasaba de 10 s y el
   contador se reiniciaba. En subject_021 (lagartija mal) había 14 reps visibles y se contaba 1.
   Ahora la rep empieza justo antes de bajar: lagartija pasó de 325 a 387 reps.

## 5. Por qué la técnica no pasa de ~0.65 (diagnóstico)

- **La diferencia entre bien y mal existe, pero es chica comparada con la diferencia entre
  personas.** Comparando el video bien y el mal de cada persona:
  - lagartija: en el video mal la cadera baja ~6.6° más (en 91 % de las personas);
  - sentadilla: los brazos suben ~12° menos (88 %);
  - curl: subida más rápida (92 %) y menos recorrido (72 %).

  Pero entre personas esos ángulos varían mucho más que 6–12°.
- **Por eso dentro de una persona se distingue mejor**: el acierto pareado es 0.79–0.83, contra
  0.63–0.73 en absoluto por video.
- **Curvas de aprendizaje** (`curva_aprendizaje_*.png`): entrenamiento ≈ 0.85–0.99 y validación ≈
  0.55–0.65, **plana** de 3 a 15 personas. Los modelos memorizan a cada persona. **Más personas
  grabadas igual no van a resolverlo.**
- **Las etiquetas son débiles.** Según la página del dataset, cada video se revisó a mano y se
  marcó como bueno o malo, sin criterios por ejercicio, sin tipos de error y sin etiquetas por
  repetición. Suponemos que todas las reps de un video "mal" están mal, y no tiene por qué ser así.
- El corte de reps no cuenta lagartijas muy poco profundas (~20° de recorrido del codo) porque
  `DELTA_INICIO` es 25°. Ajustarlo requiere el conteo real (`conteos_reales.csv`).

## 6. ¿Hace falta otro dataset?

Para **reconocer el ejercicio y contar reps**, el dataset actual alcanza. Para **calificar la
técnica**, el límite está en los datos. En orden de impacto:

1. **Grabar videos propios con errores específicos y contar las repeticiones**, siguiendo
   [PROTOCOLO_GRABACION.md](PROTOCOLO_GRABACION.md). El código ya los usa: un video
   `sentadilla__poco_profunda__1.mp4` cuenta como "mal" para el modelo de técnica y además entrena
   un modelo que dice qué error es, con su mensaje en vivo. Un error concreto y consistente es
   mucho más fácil de aprender que un "mal" genérico, y le dice a la persona qué corregir.
2. **Calibración por usuario.** El acierto pareado (0.79–0.83) muestra que comparar contra una serie
   de referencia de la misma persona funciona mejor que un umbral universal. Para usarlo en vivo
   hace falta una serie de referencia que de verdad esté bien hecha (con un entrenador), y
   validarlo con datos propios.
3. **Más vistas del mismo dataset**: la vista diagonal (`--vistas side,diagonal`) sirve para medir
   robustez al ángulo de cámara. Como más datos de entrenamiento no es seguro que ayude, porque
   los ángulos 2D cambian con la vista.
4. **Otros datasets públicos** (ninguno cubre los mismos 3 ejercicios con etiqueta bien/mal):
   - [EC3D](https://github.com/Jacoo-Zhao/3D-Pose-Based-Feedback-For-Physical-Exercises): sentadilla
     (y zancada y plancha) con 11 tipos de error etiquetados, en 3D; solo 4 personas.
   - [QEVD](https://www.qualcomm.com/developer/software/qevd-dataset) (Qualcomm, NeurIPS 2024):
     cientos de horas con errores comunes y retroalimentación de entrenador; solo para
     investigación.
   - [REHAB24-6](https://zenodo.org/records/13305826): ejercicios de rehabilitación con etiqueta
     de correcto/incorrecto y segmentación por repetición.

## Cómo reproducir

```bash
python entrenar.py --datos datos --modelos modelos --salidas salidas
python evaluar_sistema.py --datos datos --modelos modelos --salidas salidas
python experimentos.py --datos datos --modelos modelos --salidas salidas
```

La corrida completa tarda ~10 min (entrenar) + ~5 min (experimentos) en una computadora de 4 núcleos.
