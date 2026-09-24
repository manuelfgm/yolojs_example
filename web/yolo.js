/**
 * Wrapper de inferencia + postprocesado para YOLO11-seg, usando TensorFlow.js
 * (grafo convertido desde el SavedModel de Ultralytics con tensorflowjs_converter).
 * Entrada del modelo: "images" [1,640,640,3] float32 NHWC
 * Salidas: output_0 [1,116,8400] (4 caja + 80 clases + 32 coef. de máscara, igual que en ONNX)
 *          output_1 [1,160,160,32] (prototipos de máscara, NHWC)
 */
const YOLO_INPUT_SIZE = 640;
const YOLO_NUM_CLASSES = 80;
const YOLO_NUM_MASKS = 32;
const PROTO_SIZE = 160;

class YoloSegModel {
  constructor() {
    this.model = null;
    this.backend = null;
    this._letterboxCanvas = document.createElement("canvas");
    this._letterboxCanvas.width = YOLO_INPUT_SIZE;
    this._letterboxCanvas.height = YOLO_INPUT_SIZE;
    this._letterboxCtx = this._letterboxCanvas.getContext("2d", { willReadFrequently: true });
  }

  async load(modelUrl, onStatus) {
    // Los nombres reales de los tensores de salida no coinciden siempre con el orden
    // output_0/output_1 declarado en la firma, así que se leen explícitamente del model.json.
    const manifest = await (await fetch(modelUrl)).json();
    const outputs = manifest.signature?.outputs ?? manifest.userDefinedMetadata?.signature?.outputs;
    this._outputNames = [outputs.output_0.name, outputs.output_1.name];

    // WebGPU es el "delegate" de GPU real de tf.js; wasm (SIMD/threads) y cpu son fallback universal.
    const backendAttempts = ["webgpu", "wasm", "cpu"];
    for (const backend of backendAttempts) {
      try {
        onStatus?.(`Cargando modelo (${backend})…`);
        await tf.setBackend(backend);
        await tf.ready();
        this.model = await tf.loadGraphModel(modelUrl);
        this.backend = tf.getBackend();
        return this.backend;
      } catch (err) {
        console.warn(`Fallo backend ${backend}:`, err);
      }
    }
    throw new Error("No se pudo inicializar ningún backend de TensorFlow.js");
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

    // NHWC directo desde el canvas: más simple y rápido que empaquetar CHW a mano.
    const tensor = tf.tidy(() =>
      tf.browser.fromPixels(this._letterboxCanvas).toFloat().div(255).expandDims(0)
    );

    return { tensor, scale, padX, padY, srcW, srcH };
  }

  async infer(videoEl, { confThres = 0.45, iouThres = 0.45, maskThres = 0.5 } = {}) {
    const pre = this._preprocess(videoEl);

    const [out0, out1] = this.model.execute({ images: pre.tensor }, this._outputNames);

    const output0 = await out0.data(); // [116,8400] (mismo layout que en ONNX)
    const output1 = await out1.data(); // [160,160,32] NHWC

    pre.tensor.dispose();
    out0.dispose();
    out1.dispose();

    const { detections, debug } = this._decode(output0, pre, confThres);
    const kept = this._nms(detections, iouThres);
    for (const det of kept) {
      det.mask = this._buildMaskCanvas(det, output1, pre, maskThres);
    }
    kept.debug = debug;
    return kept;
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

    const color = classColorRgb(det.classId);
    const imgData = new ImageData(w, h);
    if (protoData) {
      for (let yy = 0; yy < h; yy++) {
        for (let xx = 0; xx < w; xx++) {
          const py = iy1 + yy;
          const px = ix1 + xx;
          let sum = 0;
          // Prototipos NHWC: el canal es el eje más rápido -> índice (y*W+x)*C + c
          const base = (py * PROTO_SIZE + px) * YOLO_NUM_MASKS;
          for (let c = 0; c < YOLO_NUM_MASKS; c++) {
            sum += det.maskCoeffs[c] * protoData[base + c];
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
