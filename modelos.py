"""Modelos candidatos y el envoltorio que se despliega (hiperparámetros + calibración + umbral).

`ModeloCalibrado` es exactamente lo que usa el Analizador en vivo, y es también lo que se evalúa
en la validación cruzada: así la métrica del reporte corresponde a la regla de decisión real.
"""
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

import pose


# ----------------------------------------------------------------- ROCKET
class Rocket(BaseEstimator, TransformerMixin):
    """ROCKET (Dempster, Petitjean y Webb, 2020): convoluciones aleatorias sobre la serie de
    señales de cada repetición. De cada kernel salen 2 rasgos: proporción de valores positivos
    (PPV) y máximo. Con un clasificador lineal encima es de lo mejor que hay para series cortas
    con pocos datos, y no necesita GPU.

    X: (n, canales * T), como lo da pose.secuencia_rep. Cada canal se estandariza con la media y
    desviación del entrenamiento (no por serie, para conservar la amplitud: la profundidad importa).
    """

    def __init__(self, n_kernels=1000, n_canales=len(pose.SENALES), random_state=0):
        self.n_kernels = n_kernels
        self.n_canales = n_canales
        self.random_state = random_state

    def _series(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X.reshape(len(X), self.n_canales, -1)

    def fit(self, X, y=None):
        S = self._series(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.media_ = np.nan_to_num(np.nanmean(S, axis=(0, 2)))
            self.desv_ = np.nan_to_num(np.nanstd(S, axis=(0, 2)), nan=1.0)
        self.desv_[self.desv_ < 1e-6] = 1.0
        T = S.shape[2]
        rng = np.random.default_rng(self.random_state)
        self.kernels_ = []
        for _ in range(self.n_kernels):
            L = int(rng.choice([7, 9, 11]))
            nc = int(rng.integers(1, min(3, self.n_canales) + 1))
            canales = rng.choice(self.n_canales, nc, replace=False)
            w = rng.normal(0, 1, (nc, L))
            w -= w.mean(axis=1, keepdims=True)
            b = float(rng.uniform(-1, 1))
            d = int(2 ** rng.uniform(0, np.log2((T - 1) / (L - 1))))
            pad = ((L - 1) * d) // 2 if rng.random() < 0.5 else 0
            self.kernels_.append((canales, w, b, d, pad))
        return self

    def transform(self, X):
        S = (self._series(X) - self.media_[None, :, None]) / self.desv_[None, :, None]
        S = np.nan_to_num(S)  # NaN -> media del entrenamiento
        F = np.empty((len(S), 2 * len(self.kernels_)), dtype=np.float32)
        for j, (canales, w, b, d, pad) in enumerate(self.kernels_):
            Z = S[:, canales]
            if pad:
                Z = np.pad(Z, ((0, 0), (0, 0), (pad, pad)))
            L = w.shape[1]
            t_out = Z.shape[2] - (L - 1) * d
            out = np.full((len(S), t_out), b)
            for q in range(L):
                out += np.einsum("c,nct->nt", w[:, q], Z[:, :, q * d:q * d + t_out])
            F[:, 2 * j] = (out > 0).mean(axis=1)
            F[:, 2 * j + 1] = out.max(axis=1)
        return F


# ----------------------------------------------------------------- candidatos
def candidatos(n_clases, con_secuencia=False, rapido=False):
    """{nombre: {"base", "grid", "entrada"}} de lo más simple a lo más complejo.

    - Dummy: predice siempre según la proporción de clases (piso: cualquier modelo debe superarlo).
    - Regla_umbral: un árbol de profundidad 1 (2 con 3 clases) = una regla "si rasgo < x" aprendida.
      Si un modelo complejo no le gana, no se justifica.
    - LogReg: lineal e interpretable (coeficientes).
    - RandomForest / HistGB: ensambles de árboles. SVM_RBF y MLP: no lineales.
    - ROCKET (solo técnica): lee la secuencia completa de la rep en lugar de rasgos resumidos.
    Los grids son chicos a propósito: con ~26 personas un grid grande sobreajusta la selección.
    """
    def imp():
        return SimpleImputer(strategy="median")  # dentro del pipeline: se ajusta solo con el fold de entrenamiento

    c = {
        "Dummy": (DummyClassifier(strategy="prior"), {}),
        "Regla_umbral": (make_pipeline(imp(), DecisionTreeClassifier(
            max_depth=1 if n_clases == 2 else 2, class_weight="balanced", random_state=0)), {}),
        "LogReg": (make_pipeline(imp(), StandardScaler(), LogisticRegression(
            max_iter=5000, class_weight="balanced")), {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "RandomForest": (make_pipeline(imp(), RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0)),
            {"randomforestclassifier__min_samples_leaf": [1, 5, 20]}),
        # HistGB acepta NaN directamente. Árboles poco profundos y sin early stopping: con unos
        # cientos de reps, apartar 10 % para detenerlo deja muy pocos datos
        "HistGB": (HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                                  max_iter=100, max_depth=3, random_state=0),
                   {"learning_rate": [0.05, 0.2]}),
        "SVM_RBF": (make_pipeline(imp(), StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
                    {"svc__C": [0.3, 1, 10]}),
        # early_stopping=False: en scikit-learn 1.8 falla con etiquetas de texto.
        # Una sola capa chica: con unos cientos de reps, (64, 32) tenía más parámetros que datos.
        "MLP": (make_pipeline(imp(), StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=1000, random_state=0)),
            {"mlpclassifier__alpha": [1e-3, 1e-1, 1.0]}),
    }
    res = {n: {"base": b, "grid": {} if rapido else g, "entrada": "tabular"} for n, (b, g) in c.items()}
    if con_secuencia:
        res["ROCKET"] = {"base": make_pipeline(Rocket(n_kernels=1000), StandardScaler(),
                                               RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                                                                 class_weight="balanced")),
                         "grid": {}, "entrada": "secuencia"}
    return res


def describir_regla(base, X, y, nombres):
    """Entrena la línea base de regla y la devuelve como texto legible."""
    m = clone(base).fit(X, y)
    arbol = m[-1] if hasattr(m, "steps") else m
    return export_text(arbol, feature_names=list(nombres), class_names=[str(c) for c in arbol.classes_])


# ----------------------------------------------------------------- lo que se despliega
def elegir_umbral(es_pos, p, criterio="acierto_bal"):
    """Umbral de p(positiva) que maximiza el criterio: "acierto_bal" (promedio del recall de las dos
    clases, J de Youden) o "f2" (F-beta con beta=2: el recall de la positiva pesa el doble).

    Se prueban los puntos medios entre probabilidades consecutivas. Si varios umbrales empatan
    (pasa cuando las clases se separan bien) se toma el del centro: el borde de esa meseta queda
    pegado a los datos de entrenamiento y con gente nueva falla.
    """
    es_pos, p = np.asarray(es_pos, dtype=bool), np.asarray(p, dtype=np.float64)
    if es_pos.all() or not es_pos.any():
        return 0.5
    u = np.unique(p)
    cand = np.concatenate([[u[0]], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-9]])
    pred = p[None, :] >= cand[:, None]
    tp = (pred & es_pos).sum(axis=1)
    fp = (pred & ~es_pos).sum(axis=1)
    fn = es_pos.sum() - tp
    if criterio == "f2":
        f = 5 * tp / np.maximum(5 * tp + 4 * fn + fp, 1)
    else:
        f = tp / es_pos.sum() + ((~es_pos).sum() - fp) / (~es_pos).sum()
    mejores = cand[f >= f.max() - 1e-12]
    return float(np.median(mejores))


class ModeloCalibrado(BaseEstimator, ClassifierMixin):
    """Envuelve un candidato:

    1. Busca hiperparámetros (`grid`) con validación cruzada interna agrupada por persona.
    2. Calibra las probabilidades que se muestran: regresión logística (con clases balanceadas)
       sobre las predicciones fuera de fold de esa CV interna. La calibración de
       SVC(probability=True) usaba una CV interna sin agrupar y salía demasiado confiada.
       Salvaguarda: si la calibración sale con pendiente negativa o empeora el acierto fuera de
       fold, se descarta. Con el dataset real pasaba: con ~13 personas y una señal débil, la
       calibración a veces invertía las predicciones.
    3. La decisión (`predict`) es la del propio modelo (con class_weight="balanced"). Solo si se
       pide un `criterio` ("acierto_bal" o "f2") se elige además un umbral de p(positiva) con esas
       predicciones fuera de fold. Con los datos reales el umbral elegido así resultó inestable
       y bajaba el F1 macro, por eso viene apagado (config.CRITERIO_UMBRAL = None).
    4. Entrena el modelo final con todos los datos que recibe.

    `predict` es exactamente lo que usa el Analizador en vivo.
    """

    def __init__(self, base, grid=None, positiva=None, criterio=None, n_internos=3, random_state=0):
        self.base = base
        self.grid = grid
        self.positiva = positiva
        self.criterio = criterio
        self.n_internos = n_internos
        self.random_state = random_state

    @staticmethod
    def _z(S, metodo):
        S = np.asarray(S, dtype=np.float64)
        if S.ndim == 1:
            S = S[:, None]
        if metodo == "predict_proba":
            S = np.clip(S, 1e-4, 1 - 1e-4)
            S = np.log(S[:, [1]] / S[:, [0]]) if S.shape[1] == 2 else np.log(S)
        return S

    def fit(self, X, y, groups):
        X, y, g = np.asarray(X), np.asarray(y), np.asarray(groups)
        self.classes_ = np.unique(y)
        modelo = clone(self.base)
        self.mejores_params_, self.calibrador_, self.umbral_ = {}, None, None
        k = min(self.n_internos, len(np.unique(g)))
        if k >= 2 and len(self.classes_) >= 2:
            splits = list(StratifiedGroupKFold(k, shuffle=True, random_state=self.random_state).split(X, y, g))
            if self.grid:
                gs = GridSearchCV(clone(self.base), self.grid, cv=splits, scoring="f1_macro").fit(X, y)
                self.mejores_params_ = gs.best_params_
                modelo = clone(self.base).set_params(**gs.best_params_)
            self.metodo_ = "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"
            S = cross_val_predict(clone(modelo), X, y, cv=splits, method=self.metodo_)
            Z = self._z(S, self.metodo_)
            cal = LogisticRegression(C=10.0, max_iter=5000, class_weight="balanced").fit(Z, y)
            crudo = self.classes_[np.argmax(S, axis=1)] if np.ndim(S) == 2 and S.shape[1] > 1 \
                else self.classes_[(np.ravel(S) > (0.5 if self.metodo_ == "predict_proba" else 0)).astype(int)]
            pendiente_ok = Z.shape[1] > 1 or cal.coef_[0, 0] > 0
            if pendiente_ok and (cal.predict(Z) == y).mean() >= (crudo == y).mean() - 0.02:
                self.calibrador_ = cal
            if self.calibrador_ is not None and self.positiva in self.classes_ and self.criterio:
                ip = list(cal.classes_).index(self.positiva)
                self.umbral_ = elegir_umbral(y == self.positiva, cal.predict_proba(Z)[:, ip], self.criterio)
        self.modelo_ = modelo.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.calibrador_ is None:
            if hasattr(self.modelo_, "predict_proba"):
                return self.modelo_.predict_proba(X)
            pred = self.modelo_.predict(X)
            return (pred[:, None] == self.classes_[None, :]).astype(float)
        return self.calibrador_.predict_proba(self._z(getattr(self.modelo_, self.metodo_)(X), self.metodo_))

    def predict(self, X):
        if self.umbral_ is not None and len(self.classes_) == 2:
            p = self.predict_proba(X)
            ip = list(self.classes_).index(self.positiva)
            otra = self.classes_[1 - ip]
            return np.where(p[:, ip] >= self.umbral_, self.positiva, otra)
        return self.modelo_.predict(X)
