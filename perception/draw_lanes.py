"""Interactively draw lane polygons on the first frame of a video/image.

    python -m perception.draw_lanes --source traffic.mp4 --out my_lanes.json

Left-click to add points, 'n' to finish a lane (you will be asked for its
phase number in the terminal), 's' to save, 'q' to quit.
"""
import argparse
import json

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="my_lanes.json")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.source)
    ok, frame = cap.read()
    if not ok:
        frame = cv2.imread(a.source)
    h, w = frame.shape[:2]
    lanes, pts = [], []

    def click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([round(x / w, 4), round(y / h, 4)])

    cv2.namedWindow("draw")
    cv2.setMouseCallback("draw", click)
    while True:
        vis = frame.copy()
        for lane in lanes + [{"polygon": pts}]:
            p = [(int(x * w), int(y * h)) for x, y in lane["polygon"]]
            for i in range(len(p)):
                cv2.line(vis, p[i], p[(i + 1) % len(p)], (0, 255, 0), 2)
        cv2.imshow("draw", vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("n") and len(pts) >= 3:
            phase = int(input("phase index for this lane (0..3): "))
            name = input("lane id (e.g. north_in_1): ")
            lanes.append({"id": name, "phase": phase, "polygon": pts.copy()})
            pts.clear()
        elif k == ord("s"):
            json.dump({"lanes": lanes}, open(a.out, "w"), indent=2)
            print("saved", a.out)
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
