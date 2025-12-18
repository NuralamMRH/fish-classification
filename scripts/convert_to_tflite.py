import os
import argparse
import shutil
import tempfile
from typing import List

import torch
import onnx

# Lazy imports for heavy deps
def _import_tf():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception:
        raise RuntimeError("TensorFlow is required for TFLite conversion")

def _import_onnx_tf():
    try:
        from onnx_tf.backend import prepare  # type: ignore
        return prepare
    except Exception:
        raise RuntimeError("onnx-tf is required to convert ONNX to TensorFlow")


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _get_outputs(model, dummy):
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (list, tuple)):
        return list(out)
    return [out]


def export_to_onnx(ts_path: str, onnx_path: str, input_shape: List[int], opset: int = 13) -> None:
    model = torch.jit.load(ts_path)
    model.eval()
    dummy = torch.randn(*input_shape)

    example_outputs = _get_outputs(model, dummy)
    output_names = [f"output{i}" for i in range(len(example_outputs))]

    _ensure_dir(onnx_path)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,  # static shapes are simpler for TFLite
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def onnx_to_tflite(onnx_path: str, tflite_path: str) -> None:
    prepare = _import_onnx_tf()
    tf = _import_tf()

    # Convert ONNX -> TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)

    tmp_dir = tempfile.mkdtemp(prefix="onnx_tf_")
    try:
        saved_model_dir = os.path.join(tmp_dir, "saved_model")
        tf_rep.export_graph(saved_model_dir)

        # TF SavedModel -> TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        # Broader op coverage to reduce conversion failures
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter.experimental_new_converter = True
        # Optional optimizations
        try:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        except Exception:
            pass

        tflite_model = converter.convert()
        _ensure_dir(tflite_path)
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def convert_model(kind: str):
    # Map model kind to paths and input shape
    root = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(root, "models")

    if kind == "classification":
        ts_path = os.path.join(models_dir, "classification", "model.ts")
        onnx_path = os.path.join(models_dir, "classification", "model.onnx")
        tflite_path = os.path.join(models_dir, "classification", "model.tflite")
        input_shape = [1, 3, 224, 224]
    elif kind == "segmentation":
        ts_path = os.path.join(models_dir, "segmentation", "model.ts")
        onnx_path = os.path.join(models_dir, "segmentation", "model.onnx")
        tflite_path = os.path.join(models_dir, "segmentation", "model.tflite")
        input_shape = [1, 3, 416, 416]
    elif kind == "detection":
        ts_path = os.path.join(models_dir, "detection", "model.ts")
        onnx_path = os.path.join(models_dir, "detection", "model.onnx")
        tflite_path = os.path.join(models_dir, "detection", "model.tflite")
        input_shape = [1, 3, 640, 640]
    elif kind == "face":
        ts_path = os.path.join(models_dir, "face_detector", "model.ts")
        onnx_path = os.path.join(models_dir, "face_detector", "model.onnx")
        tflite_path = os.path.join(models_dir, "face_detector", "model.tflite")
        input_shape = [1, 3, 640, 640]
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    if not os.path.exists(ts_path):
        raise FileNotFoundError(f"Model not found: {ts_path}")

    print(f"[ONNX] Exporting {kind} from {ts_path} -> {onnx_path}")
    export_to_onnx(ts_path, onnx_path, input_shape)
    print(f"[ONNX] Done: {onnx_path}")

    print(f"[TFLite] Converting {onnx_path} -> {tflite_path}")
    try:
        onnx_to_tflite(onnx_path, tflite_path)
        print(f"[TFLite] Done: {tflite_path}")
    except Exception as e:
        print(f"[TFLite] Conversion failed for {kind}: {e}")
        print(f"[TFLite] ONNX file is available at {onnx_path} for further processing.")


def main():
    parser = argparse.ArgumentParser(description="Convert TorchScript models to TFLite via ONNX and TensorFlow")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["classification", "segmentation", "detection", "face"],
        help="Models to convert: classification segmentation detection face",
    )
    args = parser.parse_args()

    for m in args.models:
        convert_model(m)


if __name__ == "__main__":
    main()
