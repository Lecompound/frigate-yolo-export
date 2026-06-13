#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "ultralytics>=8.3",
#     "onnx>=1.16",
#     "onnxruntime>=1.18",
#     "onnxslim>=0.1.40",
#     "opencv-python-headless>=4.9",
#     "numpy>=1.24",
# ]
# ///
"""Build a pretrained-COCO YOLO ONNX detector for Frigate's OpenVINO/ONNX backend.

Exports an Ultralytics YOLO model (YOLO11, YOLOv8, ...) to ONNX at a chosen
resolution, validates it locally against a sample image, and prints the matching
Frigate `model:` config block.

Examples
--------
    # Square 640, YOLO11s (default)
    python build_frigate_yolo.py

    # Non-square to match 4:3 cameras (no letterbox)
    python build_frigate_yolo.py --model yolo11s --imgsz 640x480

    # Lighter model for a busy GPU
    python build_frigate_yolo.py --model yolo11n --imgsz 320

The output ONNX feeds Frigate's `model_type: yolo-generic` decoder. See README
for deploy steps and the YOLOv9 (official-repo) alternative.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# COCO 80-class order — matches Frigate's built-in /labelmap/coco-80.txt
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def parse_imgsz(spec: str) -> tuple[int, int]:
    """Parse '640' or '640x480' (WIDTHxHEIGHT) into (width, height)."""
    spec = spec.lower().strip()
    if "x" in spec:
        w_str, h_str = spec.split("x", 1)
        w, h = int(w_str), int(h_str)
    else:
        w = h = int(spec)
    for name, v in (("width", w), ("height", h)):
        if v % 32 != 0:
            raise SystemExit(f"error: {name} {v} must be divisible by 32 (YOLO requirement)")
    return w, h


def model_stem(model: str, w: int, h: int) -> str:
    base = Path(model).stem  # strip .pt / path if given
    return f"{base}-{w}" if w == h else f"{base}-{w}x{h}"


def export_onnx(model: str, w: int, h: int, opset: int, out_dir: Path) -> Path:
    """Export the model to ONNX and move it to out_dir with a canonical name."""
    from ultralytics import YOLO

    weights = model if model.endswith(".pt") else f"{model}.pt"
    print(f"==> Exporting {weights} at {w}x{h} (WxH), opset {opset} ...")
    yolo = YOLO(weights)
    # NMS-free / end-to-end models (YOLO26, YOLOv10) emit [1, max_det, 6] final
    # detections, which Frigate's yolo-generic decoder can't read. Disabling the
    # end2end head exposes the standard [1, 84, N] grid Frigate expects.
    # No-op for grid models (yolo11/v8/v9) where end2end is absent/False.
    head = yolo.model.model[-1]
    if getattr(head, "end2end", False):
        head.end2end = False
        if hasattr(yolo.model, "end2end"):
            yolo.model.end2end = False
        print("    (disabled end2end head -> grid output for Frigate yolo-generic)")
    # Ultralytics imgsz is [height, width]; pass an int when square.
    imgsz = w if w == h else [h, w]
    produced = Path(yolo.export(format="onnx", imgsz=imgsz,
                                simplify=True, opset=opset))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{model_stem(model, w, h)}.onnx"
    shutil.move(str(produced), dest)
    print(f"==> Saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def validate(onnx_path: Path, w: int, h: int, conf_thres: float = 0.25) -> None:
    """Load the ONNX, check shapes, and run one decode on a sample image."""
    import cv2
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    print(f"==> ONNX input  {inp.name} {inp.shape}")
    print(f"==> ONNX output {out.name} {out.shape}")
    if list(inp.shape[-2:]) != [h, w]:
        print(f"!!  warning: input H,W {inp.shape[-2:]} != expected [{h},{w}]")

    # Sample image bundled with ultralytics (people + bus).
    try:
        from ultralytics.utils import ASSETS
        img = cv2.imread(str(ASSETS / "bus.jpg"))
    except Exception:
        img = None
    if img is None:
        print("==> No sample image available; skipping inference decode.")
        return

    h0, w0 = img.shape[:2]
    r = min(w / w0, h / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    canvas = np.full((h, w, 3), 114, np.uint8)
    dw, dh = (w - nw) // 2, (h - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(img, (nw, nh))
    blob = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    blob = blob.transpose(2, 0, 1)[None]

    pred = sess.run(None, {inp.name: blob})[0][0]
    if pred.shape[0] < pred.shape[1]:  # [84, N] -> [N, 84]
        pred = pred.T
    boxes, scores = pred[:, :4], pred[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf > conf_thres
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_thres, 0.45)
    idxs = np.array(idxs).flatten()
    if len(idxs) == 0:
        raise SystemExit("error: validation produced 0 detections — model looks broken")

    from collections import defaultdict
    grouped: dict[str, list[float]] = defaultdict(list)
    for i in idxs:
        grouped[COCO80[int(cls[i])]].append(round(float(conf[i]), 3))
    print(f"==> Decode OK: {len(idxs)} detections on sample image")
    for name in sorted(grouped):
        cs = grouped[name]
        print(f"      {name:12s} x{len(cs)}  conf {min(cs)}-{max(cs)}")


def print_frigate_config(onnx_path: Path, w: int, h: int) -> None:
    print("\n" + "=" * 60)
    print("Frigate config — replace your `model:` block with:")
    print("=" * 60)
    print(f"""
detectors:
  ov:
    type: openvino
    device: GPU          # or NPU, or 'cpu' for the cpu detector

model:
  model_type: yolo-generic
  width: {w}
  height: {h}
  path: /config/model_cache/{onnx_path.name}
  input_tensor: nchw
  input_dtype: float
  labelmap_path: /labelmap/coco-80.txt
""")


def prompt_menu(title: str, options: list[tuple[str, str]], default: int) -> str:
    """Ask the user to pick from options [(value, description), ...]. Returns the value.

    Non-interactive (no TTY): silently returns the default.
    """
    if not sys.stdin.isatty():
        return options[default][0]
    print(f"\n{title}")
    for i, (val, desc) in enumerate(options, 1):
        mark = "  (default)" if i - 1 == default else ""
        print(f"  {i}) {val:10s} {desc}{mark}")
    while True:
        raw = input(f"Choose [1-{len(options)}, Enter={default + 1}]: ").strip()
        if raw == "":
            return options[default][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print("  ! enter a number from the list")


def resolve_choices(model: str | None, imgsz: str | None) -> tuple[str, str]:
    """Fill in model / imgsz, prompting interactively when not given on the CLI."""
    if model is None:
        model = prompt_menu(
            "Which model? (Intel iGPU via OpenVINO)",
            [("yolo11n", "fastest, lightest — busy GPU / many cameras"),
             ("yolo11s", "balanced — recommended"),
             ("yolo11m", "most accurate, heaviest"),
             ("yolo26s", "newest architecture — NMS-free head handled automatically"),
             ("yolov8n", "classic lightweight — broad hardware support"),
             ("yolov8s", "classic balanced — broad hardware support")],
            default=1)
    if imgsz is None:
        imgsz = prompt_menu(
            "Which input size?",
            [("320", "lightest, lower accuracy on distant objects"),
             ("640", "accurate, square (letterboxes 4:3 cameras)"),
             ("640x480", "accurate, matches 4:3 cameras (no letterbox)")],
            default=1)
    return model, imgsz


def copy_to_model_cache(onnx_path: Path, dest_dir: Path) -> Path | None:
    dest_dir = dest_dir.expanduser()
    if not dest_dir.is_dir():
        print(f"!!  --copy-to: {dest_dir} is not a directory; skipping copy")
        return None
    dest = dest_dir / onnx_path.name
    shutil.copy2(onnx_path, dest)
    print(f"==> Copied to {dest}")
    return dest


def print_next_steps(onnx_path: Path, copied_to: Path | None) -> None:
    print("=" * 60)
    print("Next steps — deploy to Frigate")
    print("=" * 60)
    if copied_to is not None:
        step1 = f"Model already copied to {copied_to} ✓"
    else:
        step1 = (f"Copy {onnx_path} into Frigate's model_cache/:\n"
                 "       - HA add-on: addon_configs/<frigate>/model_cache/ (via Samba),\n"
                 "         which is /config/model_cache/ inside the container\n"
                 "       - Docker:    the host dir you bind-mount to /config")
    steps = [
        step1,
        "Paste the model: block above into your Frigate config, replacing the\n"
        "       old model block. Keep ONE OpenVINO detector — a 2nd on the same GPU\n"
        "       contends for compute and makes inference slower, not faster.",
        "Restart Frigate (add-on: Restart; Docker: docker compose restart frigate).",
        "Verify: System -> Logs (clean load, no shape/labelmap errors), then\n"
        "       System -> Metrics -> Detectors for the inference speed.",
        "If inference is too slow, re-run this tool at a smaller --imgsz (e.g. 320)\n"
        "       — the ONNX input size is fixed, so it must be re-exported, not reconfigured.",
    ]
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="Ultralytics model name or .pt path (prompts if omitted)")
    ap.add_argument("--imgsz", default=None,
                    help="'640' (square) or 'WIDTHxHEIGHT' e.g. '640x480' (prompts if omitted)")
    ap.add_argument("--opset", type=int, default=12, help="ONNX opset (default: 12)")
    ap.add_argument("--output-dir", default="models", type=Path,
                    help="Where to write the ONNX (default: ./models)")
    ap.add_argument("--copy-to", default=None, type=Path,
                    help="Also copy the ONNX into this model_cache dir (skips the manual move)")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip the local inference validation step")
    args = ap.parse_args(argv)

    model, imgsz = resolve_choices(args.model, args.imgsz)
    w, h = parse_imgsz(imgsz)
    onnx_path = export_onnx(model, w, h, args.opset, args.output_dir)
    if not args.no_validate:
        validate(onnx_path, w, h)
    print_frigate_config(onnx_path, w, h)
    copied = copy_to_model_cache(onnx_path, args.copy_to) if args.copy_to else None
    print_next_steps(onnx_path, copied)
    return 0


if __name__ == "__main__":
    sys.exit(main())
