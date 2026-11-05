"""Cámara en vivo dentro de Google Colab.

Colab corre en un servidor, así que no puede abrir tu webcam con OpenCV. Aquí el navegador
captura la cámara con JavaScript, manda cada cuadro a Python, Python calcula la pose y el
análisis, y regresa una capa transparente (esqueleto + resultado) que se dibuja encima del video.
Va a unos cuantos cuadros por segundo: suficiente para la demo.

Uso en el notebook:
    import colab_camara
    colab_camara.correr("/content/drive/MyDrive/deteccion_ejercicios/modelos")
Haz clic sobre el video para detenerlo.
"""
import base64
import json
import time

import cv2
import numpy as np

import config
import pose

_JS = r"""
var video = null, div = null, stream = null, captureCanvas = null, imgElement = null,
    labelElement = null, pendingResolve = null, shutdown = false;

function removeDom() {
  if (stream) stream.getVideoTracks()[0].stop();
  if (div) div.remove();
  video = null; div = null; stream = null; imgElement = null; captureCanvas = null; labelElement = null;
}

function onAnimationFrame() {
  if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
  if (pendingResolve) {
    var result = "";
    if (!shutdown) {
      captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
      result = captureCanvas.toDataURL('image/jpeg', 0.8);
    }
    var lp = pendingResolve; pendingResolve = null; lp(result);
  }
}

async function createDom() {
  if (div !== null) return stream;
  div = document.createElement('div');
  div.style.cssText = 'border:2px solid #333;padding:4px;width:100%;max-width:640px;position:relative';
  document.body.appendChild(div);
  const estado = document.createElement('div');
  estado.innerHTML = '<span>Estado: </span>';
  labelElement = document.createElement('span');
  labelElement.style.fontWeight = 'bold';
  labelElement.innerText = 'Iniciando...';
  estado.appendChild(labelElement);
  div.appendChild(estado);
  video = document.createElement('video');
  video.style.display = 'block';
  video.width = div.clientWidth - 8;
  video.setAttribute('playsinline', '');
  video.onclick = () => { shutdown = true; };
  stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
  div.appendChild(video);
  imgElement = document.createElement('img');
  imgElement.style.cssText = 'position:absolute;z-index:1;pointer-events:none';
  div.appendChild(imgElement);
  const ayuda = document.createElement('div');
  ayuda.innerHTML = '<span style="color:#c00;font-weight:bold">Haz clic en el video para detener</span>';
  ayuda.onclick = () => { shutdown = true; };
  div.appendChild(ayuda);
  video.srcObject = stream;
  await video.play();
  captureCanvas = document.createElement('canvas');
  captureCanvas.width = 640; captureCanvas.height = 480;
  window.requestAnimationFrame(onAnimationFrame);
  return stream;
}

async function stream_frame(label, imgData) {
  if (shutdown) { removeDom(); shutdown = false; return ''; }
  stream = await createDom();
  if (label !== '') labelElement.innerHTML = label;
  if (imgData !== '') {
    imgElement.style.top = video.offsetTop + 'px';
    imgElement.style.left = video.offsetLeft + 'px';
    imgElement.style.width = video.offsetWidth + 'px';
    imgElement.style.height = video.offsetHeight + 'px';
    imgElement.src = imgData;
  }
  var result = await new Promise(function (resolve) { pendingResolve = resolve; });
  shutdown = false;
  return {'img': result};
}
"""

_CLAVE = np.array([255, 0, 255], np.uint8)  # color "transparente" del lienzo


def _decodificar(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def _capa_png(lienzo):
    alpha = np.where(np.all(lienzo == _CLAVE, axis=2), 0, 255).astype(np.uint8)
    bgra = np.dstack([lienzo, alpha])
    ok, buf = cv2.imencode(".png", bgra)
    return "data:image/png;base64," + base64.b64encode(buf).decode() if ok else ""


def correr(dir_modelos=config.DIR_MODELOS, max_segundos=900):
    """Abre la webcam del navegador y analiza en vivo. Devuelve el Analizador con el historial."""
    from google.colab.output import eval_js
    from IPython.display import Javascript, display

    display(Javascript(_JS))
    analizador = pose.Analizador(dir_modelos)
    etiqueta, capa = "Iniciando...", ""
    t0 = time.monotonic()
    with pose.DetectorPose() as det:
        while time.monotonic() - t0 < max_segundos:
            r = eval_js(f"stream_frame({json.dumps(etiqueta)}, {json.dumps(capa)})")
            if not r or not r.get("img"):
                break
            frame = _decodificar(r["img"])
            if frame is None:
                continue
            h, w = frame.shape[:2]
            t = time.monotonic() - t0
            lm = det.detectar(frame, t * 1000)
            analizador.paso(pose.angulos(lm, w, h), t)
            lienzo = np.empty((h, w, 3), np.uint8)
            lienzo[:] = _CLAVE
            pose.dibujar(lienzo, lm)
            analizador.dibujar_hud(lienzo, t)
            capa = _capa_png(lienzo)
            ej = analizador.ejercicio
            if ej:
                c = analizador.conteo[ej]
                etiqueta = f"{config.EJERCICIOS[ej]['nombre']} | reps {c['reps']} | bien {c['bien']} | mal {c['mal']}"
            else:
                etiqueta = "Buscando ejercicio..." if lm is not None else "No te veo: colócate de perfil y completo"
    print("Cámara detenida.")
    for ej, c in analizador.conteo.items():
        if c["reps"]:
            print(f"  {config.EJERCICIOS[ej]['nombre']:15s} reps: {c['reps']:2d}   bien: {c['bien']:2d}   mal: {c['mal']:2d}")
    return analizador
