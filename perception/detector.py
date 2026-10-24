"""YOLOv8 vehicle detection and lane-wise counting.

Two models can be combined:
  * a general detector (default yolov8n.pt, COCO-pretrained) for
    car / motorcycle / bus / truck;
  * an optional fine-tuned model (perception/train_emergency.py) that
    adds ambulance / fire_truck / police classes.

Each detection is assigned to a lane by testing the bottom-centre of its
box (where the vehicle touches the road) against the lane polygons.
"""
import json

import cv2
import numpy as np
from ultralytics import YOLO

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
EMERGENCY_NAMES = {"ambulance", "fire_truck", "firetruck", "fire truck", "police",
                   "police_car", "emergency", "emergency_vehicle"}
COLORS = {"car": (0, 200, 255), "motorcycle": (255, 200, 0), "bus": (0, 180, 0),
          "truck": (40, 90, 200), "emergency": (0, 0, 255)}


class LaneCounter:
    def __init__(self, lanes_json, model="yolov8n.pt", emergency_model=None, conf=0.3,
                 imgsz=640):
        cfg = json.load(open(lanes_json))
        self.lanes = cfg["lanes"]
        self.model = YOLO(model)
        self.ev_model = YOLO(emergency_model) if emergency_model else None
        self.conf, self.imgsz = conf, imgsz
        self.n_phases = max(l["phase"] for l in self.lanes) + 1

    def _polys(self, w, h):
        return [np.array([[x * w, y * h] for x, y in l["polygon"]], dtype=np.float32)
                for l in self.lanes]

    def _lane_of(self, polys, x, y):
        for i, p in enumerate(polys):
            if cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0:
                return i
        return None

    def process(self, frame):
        """Return (detections, per-lane counts, per-phase summary)."""
        h, w = frame.shape[:2]
        polys = self._polys(w, h)
        dets = []
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                 classes=list(COCO_VEHICLES), verbose=False)[0]
        for box, c, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(),
                             res.boxes.conf.tolist()):
            dets.append({"box": box, "cls": COCO_VEHICLES[int(c)], "conf": s})
        if self.ev_model is not None:
            r2 = self.ev_model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            names = r2.names
            for box, c, s in zip(r2.boxes.xyxy.tolist(), r2.boxes.cls.tolist(),
                                 r2.boxes.conf.tolist()):
                if names[int(c)].lower() in EMERGENCY_NAMES:
                    dets.append({"box": box, "cls": "emergency", "conf": s,
                                 "subtype": names[int(c)]})

        lane_counts = [{"id": l["id"], "phase": l["phase"], "total": 0, "emergency": 0,
                        "by_class": {}} for l in self.lanes]
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            i = self._lane_of(polys, (x1 + x2) / 2, y2)
            d["lane"] = i
            if i is None:
                continue
            lc = lane_counts[i]
            if d["cls"] == "emergency":
                lc["emergency"] += 1
            else:
                lc["total"] += 1
                lc["by_class"][d["cls"]] = lc["by_class"].get(d["cls"], 0) + 1

        phases = np.zeros((self.n_phases, 2))       # vehicles, emergency
        for lc in lane_counts:
            phases[lc["phase"], 0] += lc["total"]
            phases[lc["phase"], 1] = max(phases[lc["phase"], 1], min(lc["emergency"], 1))
        return dets, lane_counts, phases

    def draw(self, frame, dets, lane_counts, banner=None):
        out = frame.copy()
        h, w = out.shape[:2]
        for poly, lc in zip(self._polys(w, h), lane_counts):
            pts = poly.astype(np.int32)
            col = (0, 0, 255) if lc["emergency"] else (255, 255, 255)
            cv2.polylines(out, [pts], True, col, 2)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(out, f"{lc['id']}: {lc['total']}", (cx - 40, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            col = COLORS.get(d["cls"], (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{d.get('subtype', d['cls'])} {d['conf']:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        if banner:
            for j, line in enumerate(banner):
                cv2.rectangle(out, (0, 22 * j), (w, 22 * (j + 1)), (0, 0, 0), -1)
                cv2.putText(out, line, (6, 22 * j + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
        return out
