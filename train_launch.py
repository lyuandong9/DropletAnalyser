#!/usr/bin/env python3
"""
Training launch script — works on both NVIDIA CUDA and AMD ROCm (via HIP).

Usage (inside WSL2 or native Linux):
    source ~/rocm_venv/bin/activate
    python train_launch.py --data dataset.yaml

Usage (Windows NVIDIA):
    python train_launch.py --data dataset.yaml --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from app.model.train import train_model, validate_model, export_to_onnx, get_best_device


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg for droplet/bubble detection"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to dataset.yaml (YOLO format)"
    )
    parser.add_argument(
        "--model", default="yolov8x-seg",
        choices=["yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg"],
        help="YOLO model variant (default: yolov8x-seg)"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (default: 1280)")
    parser.add_argument("--batch", type=int, default=0,
                        help="Batch size (0 = auto based on VRAM)")
    parser.add_argument("--device", default=None,
                        help="Device override (e.g. cuda:0, cpu)")
    parser.add_argument("--project", default="runs/train",
                        help="Output directory for training runs")
    parser.add_argument("--name", default="droplet_v1",
                        help="Experiment name")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export ONNX model after training")

    args = parser.parse_args()

    # --- Device detection ---
    device = args.device or get_best_device()

    if device == "cpu":
        print("\n[WARNING] No GPU detected! Training on CPU is extremely slow.")
        print("For AMD GPU on WSL2, ensure ROCm is properly installed.")
        print("  $ sudo ldconfig")
        print("  $ wsl --shutdown  (from Windows PowerShell)")
        print("  $ wsl")
        resp = input("\nContinue with CPU training? (y/N): ")
        if resp.lower() != "y":
            sys.exit(0)

    # --- Auto batch size ---
    if args.batch == 0:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Conservative batch size for instance segmentation at 1280px
                if vram_gb >= 20:
                    batch = 16
                elif vram_gb >= 15:
                    batch = 8
                elif vram_gb >= 10:
                    batch = 4
                else:
                    batch = 2
            else:
                batch = 2
        except Exception:
            batch = 4
        print(f"Auto batch_size = {batch} (based on VRAM)")
    else:
        batch = args.batch

    # --- Verify dataset ---
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset YAML not found: {args.data}")
        sys.exit(1)

    print(f"""
{'='*55}
  Training Configuration
{'='*55}
  Dataset:       {args.data}
  Model:         {args.model}
  Epochs:        {args.epochs}
  Image size:    {args.imgsz}
  Batch size:    {batch}
  Device:        {device}
  Output:        {args.project}/{args.name}
{'='*55}
""")

    # --- Train ---
    best_path = train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=batch,
        device=device,
        project_dir=args.project,
        experiment_name=args.name,
    )

    # --- Validate ---
    print("\n=== Running validation on best weights ===")
    metrics = validate_model(
        weights_path=best_path,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=device,
    )

    # --- Export ONNX ---
    if args.export_onnx:
        onnx_path = export_to_onnx(best_path, imgsz=args.imgsz)
        print(f"\nONNX model for inference: {onnx_path}")

    print(f"\nDone. Best weights: {best_path}")


if __name__ == "__main__":
    main()
