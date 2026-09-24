/**
 * Wrapper de inferencia + postprocesado para YOLO11-seg (ONNX, formato Ultralytics).
 * Entradas del modelo: "images" [1,3,640,640]
 * Salidas: "output0" [1,116,8400] (4 caja + 80 clases + 32 coef. de máscara)
 *          "output1" [1,32,160,160] (prototipos de máscara)
 */
const YOLO_INPUT_SIZE = 640;
const YOLO_NUM_CLASSES = 80;
const YOLO_NUM_MASKS = 32;
const PROTO_SIZE = 160;
// Soporte float16 nativo (Chrome/Edge recientes); si no existe, sólo se puede usar un model.onnx en FP32.
const HAS_NATIVE_FLOAT16 = typeof Float16Array !== "undefined";

class YoloSegModel {
  constructor() {
    this.session = null;
    this.backend = null;
    this.inputType = "float32"; // detectado en load() mediante una pasada de calentamiento
    this._letterboxCanvas = document.createElement("canvas");
    this._letterboxCanvas.width = YOLO_INPUT_SIZE;
    this._letterboxCanvas.height = YOLO_INPUT_SIZE;
    this._letterboxCtx = this._letterboxCanvas.getContext("2d", { willReadFrequently: true });
  }

  async load(modelUrl, onStatus) {
    this._modelUrl = modelUrl;
    const cacheKey = `yolo-backend:${modelUrl}`;
    const cached = safeLocalStorageGet(cacheKey);

    const providerAttempts = [
      { name: "webgpu", options: [{ executionProviders: ["webgpu"] }] },
      { name: "wasm", options: [{ executionProviders: ["wasm"] }] },
    ];
    // Si ya sabemos qué combinación funcionó antes, probamos esa primero para
    // evitar repetir en cada carga la sonda fallida de WebGPU (kernels no
    // soportados registran errores en consola aunque el fallback funcione bien).
    if (cached) {
      providerAttempts.sort((a, b) => (a.name === cached.backend ? -1 : b.name === cached.backend ? 1 : 0));
    }

    for (const attempt of providerAttempts) {
      try {
        onStatus?.(`Cargando modelo (${attempt.name})…`);
        this.session = await ort.InferenceSession.create(modelUrl, attempt.options[0]);
        this.backend = attempt.name;
        const preferredType = cached?.backend === attempt.name ? cached.inputType : undefined;
        await this._detectInputType(onStatus, preferredType);
        safeLocalStorageSet(cacheKey, { backend: this.backend, inputType: this.inputType });
        return attempt.name;
      } catch (err) {
        console.warn(`Fallo backend ${attempt.name}:`, err);
      }
    }
    throw new Error("No se pudo inicializar ningún backend de onnxruntime-web");
  }

  /**
   * El export con quantize=16 deja el modelo (entradas y salidas) en float16.
   * Se detecta con una pasada de calentamiento en vez de asumirlo, para que la
   * app funcione igual con un model.onnx en fp32 o en fp16. Si se conoce el tipo
   * que funcionó antes para este backend, se prueba primero para minimizar ruido.
   */
  async _detectInputType(onStatus, preferredType) {
    const dummyShape = [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE];
    const plane = 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
    let candidates = HAS_NATIVE_FLOAT16 ? ["float32", "float16"] : ["float32"];
    if (preferredType && candidates.includes(preferredType)) {
      candidates = [preferredType, ...candidates.filter((t) => t !== preferredType)];
    }
    for (const type of candidates) {
      try {
        const data = type === "float32" ? new Float32Array(plane) : new Float16Array(plane);
        await this.session.run({ images: new ort.Tensor(type, data, dummyShape) });
        this.inputType = type;
        onStatus?.(`Modelo listo (${type})`);
        return;
      } catch (err) {
        console.warn(`Entrada ${type} no aceptada:`, err.message);
      }
    }
    throw new Error(
      HAS_NATIVE_FLOAT16
        ? "El modelo no acepta entradas float32 ni float16"
        : "El modelo requiere float16, pero este navegador no soporta Float16Array (usa un model.onnx en FP32)"
    );
  }

  /**
   * Recarga el modelo forzando el backend WASM (algunos kernels, p.ej. el
   * Softmax del módulo DFL de YOLO, no están soportados por el JSEP de WebGPU).
   */
  async _fallbackToWasm(onStatus) {
    onStatus?.("WebGPU falló en un kernel, recargando con WASM…");
    this.session = await ort.InferenceSession.create(this._modelUrl, { executionProviders: ["wasm"] });
    this.backend = "wasm";
    await this._detectInputType(onStatus);
  }

  /**
   * Prepara el tensor de entrada a partir de un frame de vídeo, aplicando letterbox.
   * Devuelve también los parámetros para deshacer el letterbox tras la inferencia.
   */
  _preprocess(videoEl) {
    const srcW = videoEl.videoWidth;
    const srcH = videoEl.videoHeight;
    const scale = Math.min(YOLO_INPUT_SIZE / srcW, YOLO_INPUT_SIZE / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const padX = Math.floor((YOLO_INPUT_SIZE - newW) / 2);
    const padY = Math.floor((YOLO_INPUT_SIZE - newH) / 2);

    const ctx = this._letterboxCtx;
    ctx.fillStyle = "#727272";
    ctx.fillRect(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    ctx.drawImage(videoEl, 0, 0, srcW, srcH, padX, padY, newW, newH);

    const { data } = ctx.getImageData(0, 0, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE);
    const plane = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
    const chw = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const o = i * 4;
      chw[i] = data[o] / 255;
      chw[plane + i] = data[o + 1] / 255;
      chw[2 * plane + i] = data[o + 2] / 255;
    }

    const inputData = this.inputType === "float16" ? new Float16Array(chw) : chw;

    return {
      tensor: new ort.Tensor(this.inputType, inputData, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
      scale, padX, padY, srcW, srcH,
    };
  }

  async infer(videoEl, { confThres = 0.45, iouThres = 0.45, maskThres = 0.5 } = {}, onStatus) {
    const pre = this._preprocess(videoEl);
    const feeds = { images: pre.tensor };

    let results;
    try {
      results = await this.session.run(feeds);
    } catch (err) {
      if (this.backend === "webgpu") {
        await this._fallbackToWasm(onStatus);
        results = await this.session.run(this._rebuildFeeds(pre));
      } else {
        throw err;
      }
    }

    const output0 = toFloat32(results.output0);
    const output1 = toFloat32(results.output1);

    const { detections, debug } = this._decode(output0, pre, confThres);
    const kept = this._nms(detections, iouThres);
    for (const det of kept) {
      det.mask = this._buildMaskCanvas(det, output1, pre, maskThres);
    }
    kept.debug = debug;
    return kept;
  }

  // El calentamiento tras el fallback puede cambiar this.inputType; reconstruye el tensor si hace falta.
  _rebuildFeeds(pre) {
    const raw = pre.tensor.type === "float16" ? Float32Array.from(pre.tensor.data) : pre.tensor.data;
    const data = this.inputType === "float16" ? new Float16Array(raw) : Float32Array.from(raw);
    return { images: new ort.Tensor(this.inputType, data, pre.tensor.dims) };
  }

  _decode(output0, pre, confThres) {
    const N = 8400;
    const dets = [];
    const { scale, padX, padY, srcW, srcH } = pre;
    const debug = { bestOverall: { score: 0, classId: -1 }, bestPerson: 0 };

    for (let p = 0; p < N; p++) {
      let bestScore = 0;
      let bestClass = -1;
      for (let c = 0; c < YOLO_NUM_CLASSES; c++) {
        const s = output0[(4 + c) * N + p];
        if (s > bestScore) {
          bestScore = s;
          bestClass = c;
        }
      }
      if (bestScore > debug.bestOverall.score) {
        debug.bestOverall = { score: bestScore, classId: bestClass };
      }
      if (bestClass === 0 && bestScore > debug.bestPerson) {
        debug.bestPerson = bestScore;
      }
      if (bestScore < confThres) continue;

      const cx = output0[0 * N + p];
      const cy = output0[1 * N + p];
      const w = output0[2 * N + p];
      const h = output0[3 * N + p];

      const x1_640 = cx - w / 2;
      const y1_640 = cy - h / 2;
      const x2_640 = cx + w / 2;
      const y2_640 = cy + h / 2;

      const x1 = clamp((x1_640 - padX) / scale, 0, srcW);
      const y1 = clamp((y1_640 - padY) / scale, 0, srcH);
      const x2 = clamp((x2_640 - padX) / scale, 0, srcW);
      const y2 = clamp((y2_640 - padY) / scale, 0, srcH);
      if (x2 - x1 < 1 || y2 - y1 < 1) continue;

      const maskCoeffs = new Float32Array(YOLO_NUM_MASKS);
      for (let k = 0; k < YOLO_NUM_MASKS; k++) {
        maskCoeffs[k] = output0[(4 + YOLO_NUM_CLASSES + k) * N + p];
      }

      dets.push({
        classId: bestClass,
        score: bestScore,
        box640: [x1_640, y1_640, x2_640, y2_640],
        box: [x1, y1, x2, y2],
        maskCoeffs,
      });
    }
    return { detections: dets, debug };
  }

  _nms(dets, iouThres) {
    dets.sort((a, b) => b.score - a.score);
    const kept = [];
    const used = new Array(dets.length).fill(false);
    for (let i = 0; i < dets.length; i++) {
      if (used[i]) continue;
      const a = dets[i];
      kept.push(a);
      for (let j = i + 1; j < dets.length; j++) {
        if (used[j] || dets[j].classId !== a.classId) continue;
        if (iou(a.box, dets[j].box) > iouThres) used[j] = true;
      }
    }
    return kept;
  }

  _buildMaskCanvas(det, protoData, pre, maskThres) {
    const [x1_640, y1_640, x2_640, y2_640] = det.box640;
    const ratio = PROTO_SIZE / YOLO_INPUT_SIZE; // 0.25
    const ix1 = clamp(Math.floor(x1_640 * ratio), 0, PROTO_SIZE);
    const iy1 = clamp(Math.floor(y1_640 * ratio), 0, PROTO_SIZE);
    const ix2 = clamp(Math.ceil(x2_640 * ratio), 0, PROTO_SIZE);
    const iy2 = clamp(Math.ceil(y2_640 * ratio), 0, PROTO_SIZE);
    const w = Math.max(1, ix2 - ix1);
    const h = Math.max(1, iy2 - iy1);

    const plane = PROTO_SIZE * PROTO_SIZE;
    const color = classColorRgb(det.classId);
    const imgData = new ImageData(w, h);
    for (let yy = 0; yy < h; yy++) {
      for (let xx = 0; xx < w; xx++) {
        const py = iy1 + yy;
        const px = ix1 + xx;
        let sum = 0;
        const protoIdx = py * PROTO_SIZE + px;
        for (let c = 0; c < YOLO_NUM_MASKS; c++) {
          sum += det.maskCoeffs[c] * protoData[c * plane + protoIdx];
        }
        const v = sigmoid(sum);
        const o = (yy * w + xx) * 4;
        if (v > maskThres) {
          imgData.data[o] = color[0];
          imgData.data[o + 1] = color[1];
          imgData.data[o + 2] = color[2];
          imgData.data[o + 3] = 130;
        }
      }
    }

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d").putImageData(imgData, 0, 0);

    // Rectángulo destino (coords fuente) a partir del rectángulo de máscara en espacio 640
    const { scale, padX, padY, srcW, srcH } = pre;
    const destX1 = clamp((ix1 / ratio - padX) / scale, 0, srcW);
    const destY1 = clamp((iy1 / ratio - padY) / scale, 0, srcH);
    const destX2 = clamp((ix2 / ratio - padX) / scale, 0, srcW);
    const destY2 = clamp((iy2 / ratio - padY) / scale, 0, srcH);

    return { canvas, dest: [destX1, destY1, destX2 - destX1, destY2 - destY1] };
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function safeLocalStorageGet(key) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function safeLocalStorageSet(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // almacenamiento no disponible (modo privado, cuota, etc.): no es crítico
  }
}

// Conversión IEEE-754 float32 <-> float16 para navegadores sin Float16Array nativo.
// Ya no se usa para construir tensores de entrada (onnxruntime-web exige un
// Float16Array real), pero se deja como referencia/lectura de salidas si hiciera falta.
function float16BitsToFloat32(h) {
  const sign = (h & 0x8000) >> 15;
  const exp = (h & 0x7c00) >> 10;
  const frac = h & 0x03ff;
  if (exp === 0) return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  if (exp === 0x1f) return frac ? NaN : (sign ? -1 : 1) * Infinity;
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

// Normaliza cualquier tensor de salida (float32 o float16) a un Float32Array plano.
// Con Float16Array nativo, indexar ya devuelve números JS normales.
function toFloat32(tensor) {
  if (tensor.type !== "float16") return tensor.data;
  return HAS_NATIVE_FLOAT16 ? Float32Array.from(tensor.data) : Float32Array.from(tensor.data, float16BitsToFloat32);
}


function iou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

function classColorRgb(classId) {
  const hue = (classId * 37) % 360;
  return hslToRgb(hue / 360, 0.85, 0.55);
}

function hslToRgb(h, s, l) {
  let r, g, b;
  if (s === 0) {
    r = g = b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}
