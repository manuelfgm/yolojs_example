"""Convierte un modelo ONNX de Ultralytics a TensorFlow SavedModel.

Entorno requerido:
    source .venv/bin/activate

Instala las dependencias del entorno principal antes de ejecutar este script:
    pip install -r requirements.txt

Uso:
    python scripts/onnx_to_saved_model.py best.onnx best_saved_model
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics.utils.export.tensorflow import onnx2saved_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convierte ONNX a TensorFlow SavedModel para TF.js.")
    parser.add_argument("onnx_model", type=Path, help="Ruta al modelo ONNX de entrada.")
    parser.add_argument("output_dir", type=Path, help="Carpeta de destino para el SavedModel.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Elimina la carpeta de destino antes de generar el SavedModel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.onnx_model.is_file():
        raise SystemExit(f"No existe el modelo ONNX: {args.onnx_model}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Ya existe {args.output_dir}; usa --overwrite para sustituirlo.")
        shutil.rmtree(args.output_dir)

    # Descompone las convoluciones agrupadas/depthwise para que TF.js las soporte.
    onnx2saved_model(
        str(args.onnx_model),
        str(args.output_dir),
        disable_group_convolution=True,
        cuda=False,
    )


if __name__ == "__main__":
    main()
