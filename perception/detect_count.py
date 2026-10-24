"""Count vehicles per lane in an image, a video or a webcam stream.

    python -m perception.detect_count --source traffic.mp4 --lanes perception/lanes_example.json
    python -m perception.detect_count --source junction.jpg --emergency-model models/emergency.pt

Writes an annotated copy (results/perception/) and a CSV of per-frame,
per-lane counts that can be used to calibrate SUMO demand.
"""
import argparse
import csv
from pathlib import Path

import cv2

from config import RESULTS
from perception.detector import LaneCounter

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image, video file or webcam index")
    ap.add_argument("--lanes", default="perception/lanes_example.json")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--emergency-model", default=None)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--every", type=int, default=1, help="process every n-th frame")
    ap.add_argument("--show", action="store_true", help="display a window while running")
    a = ap.parse_args()

    out_dir = RESULTS / "perception"
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = LaneCounter(a.lanes, a.model, a.emergency_model, a.conf)
    src = Path(a.source)

    rows = []
    if src.suffix.lower() in IMG_EXT:
        frame = cv2.imread(str(src))
        dets, lanes, _ = counter.process(frame)
        cv2.imwrite(str(out_dir / f"annotated_{src.name}"), counter.draw(frame, dets, lanes))
        for lc in lanes:
            rows.append({"frame": 0, **{k: lc[k] for k in ("id", "phase", "total", "emergency")}})
        print(f"{len(dets)} vehicles detected")
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        writer, i = None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % a.every == 0:
                dets, lanes, _ = counter.process(frame)
                vis = counter.draw(frame, dets, lanes)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps / a.every, (w, h))
                writer.write(vis)
                for lc in lanes:
                    rows.append({"frame": i, **{k: lc[k] for k in ("id", "phase", "total", "emergency")}})
                if a.show:
                    cv2.imshow("lane counts", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            i += 1
        if writer:
            writer.release()
        print(f"processed {i} frames")

    with open(out_dir / "lane_counts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "id", "phase", "total", "emergency"])
        w.writeheader()
        w.writerows(rows)
    print("outputs in", out_dir)


if __name__ == "__main__":
    main()
