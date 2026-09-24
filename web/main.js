// Configuración de onnxruntime-web (WebGPU con fallback a WASM)
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
// El multi-hilo (SharedArrayBuffer) sólo funciona en un contexto "cross-origin isolated"
// (cabeceras COOP/COEP, servidas por server.js). Si no está disponible, cae a 1 hilo.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 4) : 1;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");
const debugScoresEl = document.getElementById("debugScores");
const btnStart = document.getElementById("btnStart");
const btnSwitch = document.getElementById("btnSwitch");
const confSlider = document.getElementById("confSlider");
const confVal = document.getElementById("confVal");

const model = new YoloSegModel();
let running = false;
let facingMode = "environment";
let currentStream = null;
let confThres = parseFloat(confSlider.value);

confSlider.addEventListener("input", () => {
  confThres = parseFloat(confSlider.value);
  confVal.textContent = confThres.toFixed(2);
});

btnStart.addEventListener("click", async () => {
  btnStart.disabled = true;
  try {
    await startCamera(facingMode);
    await ensureModelLoaded();
    btnSwitch.disabled = false;
    running = true;
    statusEl.textContent = `Detectando (${model.backend})`;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    statusEl.textContent = `Error: ${err.message}`;
    btnStart.disabled = false;
  }
});

btnSwitch.addEventListener("click", async () => {
  facingMode = facingMode === "environment" ? "user" : "environment";
  await startCamera(facingMode);
});

async function ensureModelLoaded() {
  if (model.session) return;
  const backend = await model.load("model.onnx", (msg) => (statusEl.textContent = msg));
  console.log("Backend activo:", backend);
}

async function startCamera(mode) {
  if (currentStream) {
    currentStream.getTracks().forEach((t) => t.stop());
  }
  statusEl.textContent = "Solicitando cámara…";
  currentStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { ideal: mode },
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  });
  video.srcObject = currentStream;
  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play();
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      resolve();
    };
  });
}

let lastTime = performance.now();
let frames = 0;
let fpsAccum = 0;

async function loop() {
  if (!running) return;
  const t0 = performance.now();

  try {
    const detections = await model.infer(
      video,
      { confThres, iouThres: 0.45, maskThres: 0.5 },
      (msg) => (statusEl.textContent = msg)
    );
    draw(detections);
    if (running) statusEl.textContent = `Detectando (${model.backend})`;
    if (detections.debug) {
      const { bestOverall, bestPerson } = detections.debug;
      debugScoresEl.textContent = `mejor: ${COCO_CLASSES[bestOverall.classId] ?? "-"} ${(bestOverall.score * 100).toFixed(0)}% · persona: ${(bestPerson * 100).toFixed(0)}%`;
    }
  } catch (err) {
    console.error("Error de inferencia:", err);
  }

  const t1 = performance.now();
  frames++;
  fpsAccum += 1000 / (t1 - t0);
  if (t1 - lastTime > 500) {
    fpsEl.textContent = `${(fpsAccum / frames).toFixed(1)} FPS`;
    frames = 0;
    fpsAccum = 0;
    lastTime = t1;
  }

  requestAnimationFrame(loop);
}

function draw(detections) {
  octx.clearRect(0, 0, overlay.width, overlay.height);

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const color = classColor(det.classId);
    const label = `${COCO_CLASSES[det.classId]} ${(det.score * 100).toFixed(0)}%`;

    if (det.mask) {
      const [dx, dy, dw, dh] = det.mask.dest;
      octx.drawImage(det.mask.canvas, dx, dy, dw, dh);
    }

    octx.strokeStyle = color;
    octx.lineWidth = 2;
    octx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    octx.font = "16px sans-serif";
    const textW = octx.measureText(label).width;
    octx.fillStyle = color;
    octx.fillRect(x1, Math.max(0, y1 - 20), textW + 8, 20);
    octx.fillStyle = "#000";
    octx.fillText(label, x1 + 4, Math.max(14, y1 - 5));
  }
}
