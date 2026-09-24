"""Reescribe el Softmax del módulo DFL de YOLO para que opere sobre el último eje.

El export de Ultralytics genera, dentro de `model.23.dfl` (Distribution Focal Loss),
un `Softmax(axis=1)` sobre un tensor `[N, 16, 4, 8400]`. El execution provider WebGPU
de onnxruntime-web (JSEP) sólo soporta Softmax sobre el último eje, así que ese nodo
falla en el navegador con WebGPU y obliga a caer siempre a WASM (CPU).

Este script sustituye ese único nodo por:
    Transpose(perm=[0,2,3,1])  -> Softmax(axis=-1)  -> Transpose(perm=[0,3,1,2])
que es matemáticamente equivalente (softmax es invariante al eje siempre que se
deshaga la permutación), pero sí soporta WebGPU.

Uso:
    python scripts/fix_dfl_softmax_axis.py web/model.onnx web/model.onnx
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def patch(model: onnx.ModelProto) -> onnx.ModelProto:
    graph = model.graph
    target = next((n for n in graph.node if n.op_type == "Softmax" and "dfl" in n.name.lower()), None)
    if target is None:
        raise SystemExit("No se encontró el nodo Softmax del módulo DFL en el grafo")

    axis = next((helper.get_attribute_value(a) for a in target.attribute if a.name == "axis"), -1)
    rank = 4  # conocido por inspección: [N, 16, 4, 8400]
    if axis in (-1, rank - 1):
        print("El Softmax ya opera sobre el último eje; no hace falta parchear.")
        return model

    perm_to_last = [i for i in range(rank) if i != axis] + [axis]
    perm_back = [0] * rank
    for new_pos, old_axis in enumerate(perm_to_last):
        perm_back[old_axis] = new_pos

    x_name = target.input[0]
    y_name = target.output[0]
    pre_name = f"{target.name}/pre_transpose_output"
    post_softmax_name = f"{target.name}/softmax_last_axis_output"

    t1 = helper.make_node("Transpose", [x_name], [pre_name], perm=perm_to_last, name=f"{target.name}_pre")
    # axis=-1 en vez de rank-1: el kernel JSEP de WebGPU sólo reconoce el literal -1 como "último eje".
    softmax = helper.make_node("Softmax", [pre_name], [post_softmax_name], axis=-1, name=f"{target.name}_last")
    t2 = helper.make_node("Transpose", [post_softmax_name], [y_name], perm=perm_back, name=f"{target.name}_post")

    idx = list(graph.node).index(target)
    del graph.node[idx]
    for offset, node in enumerate((t1, softmax, t2)):
        graph.node.insert(idx + offset, node)

    onnx.checker.check_model(model)
    return model


def verify_equivalence(original_path: str, patched_model: onnx.ModelProto) -> None:
    """Compara numéricamente el modelo original y el parcheado con la misma entrada aleatoria."""
    dtype = np.float16 if patched_model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16 else np.float32
    x = np.random.rand(1, 3, 640, 640).astype(dtype)

    sess_orig = ort.InferenceSession(original_path, providers=["CPUExecutionProvider"])
    out_orig = sess_orig.run(None, {"images": x})

    patched_bytes = patched_model.SerializeToString()
    sess_new = ort.InferenceSession(patched_bytes, providers=["CPUExecutionProvider"])
    out_new = sess_new.run(None, {"images": x})

    for a, b, name in zip(out_orig, out_new, [o.name for o in sess_orig.get_outputs()]):
        atol = 1e-2 if dtype == np.float16 else 1e-4
        if not np.allclose(a, b, atol=atol):
            raise SystemExit(f"Salida '{name}' difiere tras el parche (max diff={np.abs(a - b).max()})")
    print("Verificado: salidas idénticas antes/después del parche.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"Uso: {sys.argv[0]} <entrada.onnx> <salida.onnx>")
    in_path, out_path = sys.argv[1], sys.argv[2]

    model = onnx.load(in_path)
    patched = patch(model)
    verify_equivalence(in_path, patched)
    onnx.save(patched, out_path)
    print(f"Guardado: {out_path}")
