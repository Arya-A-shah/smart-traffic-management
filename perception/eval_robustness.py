"""Compare detection accuracy (mAP) in normal vs adverse conditions.

    python -m perception.eval_robustness --model models/emergency.pt \
        --normal data.yaml --adverse data_adverse.yaml

Both yaml files describe the same validation images; the adverse one
points at the copy produced by augment_weather.py. Results are saved to
results/perception_robustness.json and shown on the dashboard.
"""
import argparse
import json

from ultralytics import YOLO

from config import RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--adverse", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    model = YOLO(a.model)
    out = {}
    for name, data in (("normal", a.normal), ("adverse", a.adverse)):
        m = model.val(data=data, imgsz=a.imgsz, verbose=False, plots=False)
        out[name] = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                     "precision": float(m.box.mp), "recall": float(m.box.mr)}
        print(name, out[name])
    (RESULTS / "perception_robustness.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
