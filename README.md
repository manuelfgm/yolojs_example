# YOLO11-seg en el navegador (WebGPU + onnxruntime-web)

Segmentación de instancias en tiempo real usando la cámara del dispositivo, pensada para
abrirse principalmente desde un **móvil**. Corre 100% en el cliente (no hay backend de
inferencia): carga `model.onnx` con `onnxruntime-web`, probando primero **WebGPU** y
haciendo *fallback* automático a **WASM** si algún kernel no está soportado.

## Estructura

```
web/
├── index.html        # UI (vídeo + canvas overlay + controles)
├── style.css
├── main.js            # cámara, bucle de inferencia, dibujado
├── yolo.js            # carga del modelo, preprocesado, decodificación, NMS, máscaras
├── coco-classes.js    # nombres de clases COCO y colores
├── model.onnx         # copia de yolo11n-seg.onnx
├── server.js           # servidor HTTPS estático para pruebas (necesario en móvil)
└── certs/             # certificado autofirmado usado por server.js
```

## Cómo funciona

1. **Captura**: `getUserMedia` obtiene el stream de la cámara (trasera por defecto,
   botón para alternar a la frontal).
2. **Preprocesado**: cada frame se redimensiona con *letterbox* a 640×640 (relleno gris,
   igual que en el entrenamiento) y se normaliza a `float32` CHW.
3. **Inferencia**: `onnxruntime-web` ejecuta `model.onnx`, que devuelve:
   - `output0` `[1,116,8400]` → 4 coords de caja + 80 puntuaciones de clase + 32
     coeficientes de máscara, por cada una de las 8400 celdas.
   - `output1` `[1,32,160,160]` → prototipos de máscara.
4. **Postprocesado** (`yolo.js`): se filtra por confianza, se aplica NMS por clase, y
   para cada detección se reconstruye su máscara combinando los coeficientes con los
   prototipos (sigmoide + umbral).
5. **Dibujado**: cajas, etiqueta+score y máscara semitransparente sobre un `<canvas>`
   superpuesto al `<video>` (mismo tamaño y `object-fit` para que coincidan píxel a píxel).

### Fallback WebGPU → WASM

Algunos kernels (p. ej. el `Softmax` del módulo DFL de YOLO) no están implementados en
el backend JSEP de WebGPU en la versión actual de `onnxruntime-web` y lanzan un error en
tiempo de ejecución. Si esto ocurre, `yolo.js` recarga automáticamente la sesión forzando
WASM y reintenta, sin necesidad de recargar la página. El HUD indica el backend activo
("Detectando (webgpu)" / "Detectando (wasm)").

### HUD de depuración

Junto al estado y los FPS se muestra `mejor: <clase> <%> · persona: <%>`: la puntuación
más alta detectada en el frame (de cualquier clase) y la puntuación máxima para la clase
"person", **aunque estén por debajo del umbral de confianza**. Útil para saber si hay que
bajar el slider de confianza o si el modelo realmente no ve nada.

## Uso

### 1. Servir por HTTPS

`getUserMedia` exige un contexto seguro. En `localhost` basta con HTTP, pero para abrir
la app **desde el móvil** (misma red WiFi) hace falta HTTPS, por eso se incluye un
servidor mínimo con certificado autofirmado:

```bash
cd web
node server.js
```

Salida esperada:

```
Servidor HTTPS escuchando en el puerto 8443
  https://<ip-de-tu-equipo>:8443   ← ábrelo desde el móvil (misma WiFi)
  https://localhost:8443           ← para probar en este equipo
```

El navegador avisará de certificado no confiable (es autofirmado): acepta el riesgo para
continuar. Si haces cambios en el código, recarga con caché deshabilitada o sube el
número de versión (`?v=N`) en los `<script>` de `index.html`, ya que el servidor no cachea
pero el navegador puede hacerlo igualmente.

### 2. Usar la app

1. Pulsa **"Iniciar cámara"** y concede el permiso.
2. Ajusta el slider **"Confianza"** si no ves detecciones (bájalo a ~0.15–0.25).
3. **"Cambiar cámara"** alterna entre trasera/frontal.

### 3. Actualizar el modelo

Si reexportas el `.onnx`, sustituye `web/model.onnx` (el nombre de archivo y las
entradas/salidas `images` / `output0` / `output1` deben mantenerse, o habrá que ajustar
`yolo.js`).

## Sobre el flag `half` al exportar

```
yolo export model=yolo11n-seg.pt format=onnx half=True opset=17
```

En la versión de `ultralytics` instalada aquí (**8.4.160**) el argumento `half` está
**obsoleto**: el exportador ONNX ahora usa `quantize=16` para pedir FP16, y solo se aplica
si el export se ejecuta en GPU (`device.type != "cpu"`). Como el export corrió en CPU,
`half=True` no tuvo efecto y el `model.onnx` resultante quedó en **FP32** (verificado:
`images`, `output0`, `output1` son todos `float32`). Esto explica por qué no hubo
problemas de tipos con el `Float32Array` que se envía desde JavaScript.

Recomendación:
- Para servir en el navegador, **FP32 es lo correcto**: `onnxruntime-web` (tanto WASM
  como WebGPU) tiene mucho mejor soporte y rendimiento con FP32 que con FP16, y evita
  conversiones de tipo en el cliente.
- Si en el futuro quieres forzar FP16 realmente (por ejemplo para reducir el tamaño del
  archivo a la mitad), usa el flag actual y ejecuta el export en GPU:
  ```bash
  yolo export model=yolo11n-seg.pt format=onnx quantize=16 opset=17 device=0
  ```
  Ten en cuenta que entonces el input/output del modelo pasaría a ser `float16`, y habría
  que adaptar `yolo.js` (crear el tensor de entrada como `"float16"` con un
  `Uint16Array` empaquetado, y decodificar las salidas igual). Para este caso de uso
  (un solo modelo `nano`, tamaño ya pequeño) no compensa la complejidad añadida: quédate
  con el export por defecto en FP32.

## GPU en el navegador: qué garantiza (y qué no) `onnxruntime-web`

- **WebGPU *es* el "delegate" de GPU** en el navegador: cuando `onnxruntime-web` usa el
  execution provider `webgpu`, los kernels corren realmente sobre la GPU del sistema a
  través de la API WebGPU del navegador (no es un fallback de CPU disfrazado). El "Error
  de inferencia" que veíamos antes (`Softmax` del módulo DFL) era justo eso: un kernel
  concreto sin implementar en el JSEP de WebGPU de esta versión de `onnxruntime-web`, no
  una señal de que WebGPU no se estuviera usando.
- **No se puede "forzar" GPU de forma universal en todos los móviles**: a diferencia de
  delegates nativos (NNAPI/CoreML en apps nativas), WebGPU depende del navegador y del
  dispositivo. Soporte real a día de hoy: Chrome/Edge en Android (bastante maduro),
  Safari/iOS (soporte más reciente y limitado según versión de iOS), navegadores basados
  en Firefox (aún experimental). Cuando WebGPU no está disponible, la app cae a WASM
  (CPU) automáticamente — es el único "delegate" universal que garantiza que la app
  funcione en cualquier navegador.
- **Para que el fallback de CPU sea lo más eficiente posible**, `server.js` envía las
  cabeceras `Cross-Origin-Opener-Policy`/`Cross-Origin-Embedder-Policy`, que habilitan
  `SharedArrayBuffer` y por tanto WASM **multi-hilo** (`ort.env.wasm.numThreads` se ajusta
  automáticamente al número de núcleos si el contexto está "cross-origin isolated").
- El HUD muestra siempre el backend real en uso: `Detectando (webgpu)` o
  `Detectando (wasm)`, para verificarlo en cada dispositivo.

## Soporte FP16 (`quantize=16`)

`yolo.js` ya no asume FP32: al cargar el modelo hace una pasada de calentamiento
probando primero `float32` y luego `float16`, y usa el que el modelo acepte. Los
navegadores no tienen `Float16Array` nativo, así que los tensores fp16 se empaquetan/
desempaquetan a mano (bits IEEE-754 half-precision) tanto en la entrada (`images`) como
en las salidas (`output0`, `output1`) antes del postprocesado. Esto permite usar
directamente el `model.onnx` exportado con `quantize=16 device=0` sin tocar código.

Ventaja real de FP16 aquí: modelo más pequeño de descargar (importante en móvil) y,
donde el backend WebGPU soporte bien los kernels en fp16, menor uso de memoria/ancho de
banda en la GPU. El coste es la conversión fp16↔fp32 en JS en cada frame (CPU), que para
un modelo `nano` es asumible.

## ¿Hace falta el fichero `.pt`?

No, para la app web **no se usa en ningún momento** — solo lee `web/model.onnx`. El
`.pt` (pesos de PyTorch) sólo es necesario si quieres **volver a exportar** el modelo más
adelante (otro `opset`, `imgsz`, `quantize`, cuantización INT8, TensorRT, etc.), ya que el
`.onnx` no se puede "reexportar" a otro formato con la misma fidelidad. Si no vas a tocar
el modelo, puedes borrar `yolo11n-seg.pt` con seguridad; si crees que volverás a exportar,
consérvalo (o vuelve a descargarlo con `yolo` cuando lo necesites, ya que Ultralytics lo
descarga automáticamente si falta).

