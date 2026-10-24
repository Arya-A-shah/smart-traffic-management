"""Fine-tune YOLOv8 to add emergency-vehicle classes.

1. Download a labelled emergency-vehicle dataset in YOLO format (for
   example from Roboflow Universe or Kaggle; search "emergency vehicle
   detection"), and edit perception/data_template.yaml to point at it.
2. python -m perception.train_emergency --data my_data.yaml --epochs 50
3. The best weights are copied to models/emergency.pt and used with
   --emergency-model models/emergency.pt

Brightness/colour augmentation (hsv_v) and the synthetic night/rain/fog
images from augment_weather.py improve robustness to adverse conditions.
"""
import argparse
import shutil

from ultralytics import YOLO

from config import MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    a = ap.parse_args()
    model = YOLO(a.base)
    res = model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                      device=a.device, hsv_v=0.6, hsv_s=0.7, degrees=5, mosaic=1.0,
                      project=str(MODELS / "yolo_runs"), name="emergency", exist_ok=True)
    best = MODELS / "yolo_runs" / "emergency" / "weights" / "best.pt"
    shutil.copy(best, MODELS / "emergency.pt")
    print("saved", MODELS / "emergency.pt")
    return res


if __name__ == "__main__":
    main()
