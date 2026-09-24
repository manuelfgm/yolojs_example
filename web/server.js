// Servidor HTTPS estático simple, para poder abrir la app desde el móvil
// (getUserMedia exige un contexto seguro salvo en localhost).
const https = require("https");
const fs = require("fs");
const path = require("path");
const os = require("os");

const ROOT = __dirname;
const PORT = process.env.PORT || 8443;

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".onnx": "application/octet-stream",
  ".wasm": "application/wasm",
  ".json": "application/json",
};

const options = {
  key: fs.readFileSync(path.join(ROOT, "certs", "key.pem")),
  cert: fs.readFileSync(path.join(ROOT, "certs", "cert.pem")),
};

const server = https.createServer(options, (req, res) => {
  let reqPath = decodeURIComponent(req.url.split("?")[0]);
  if (reqPath === "/") reqPath = "/index.html";
  const filePath = path.join(ROOT, reqPath);

  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, {
      "Content-Type": MIME[ext] || "application/octet-stream",
      "Cache-Control": "no-store, no-cache, must-revalidate",
      // Habilita SharedArrayBuffer (WASM multi-hilo) marcando la página como "cross-origin isolated".
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    });
    res.end(data);
  });
});

server.listen(PORT, "0.0.0.0", () => {
  const nets = os.networkInterfaces();
  console.log(`Servidor HTTPS escuchando en el puerto ${PORT}`);
  console.log("Abre en el móvil (misma red WiFi):");
  for (const iface of Object.values(nets)) {
    for (const net of iface) {
      if (net.family === "IPv4" && !net.internal) {
        console.log(`  https://${net.address}:${PORT}`);
      }
    }
  }
  console.log(`También en este equipo: https://localhost:${PORT}`);
  console.log("El navegador mostrará aviso de certificado no confiable: acepta el riesgo para continuar.");
});
