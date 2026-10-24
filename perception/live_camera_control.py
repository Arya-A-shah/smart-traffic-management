"""Camera -> perception -> RL policy -> signal recommendation.

This closes the loop from the proposal's architecture on a single real
intersection video: every processed frame, YOLO lane counts are turned
into the same observation vector the RL agent was trained on, the policy
recommends a phase, and the safety rule gives emergency vehicles priority.
The recommended signal and its reason are drawn on the output video.

    python -m perception.live_camera_control --source junction.mp4 \
        --lanes my_lanes.json --policy models/ppo_noforecast.pt

Limitations (state them in the report): a single camera cannot see
upstream signals or pedestrians waiting, and without tracking we treat
every vehicle inside a lane region as queued. Those features are set to
zero, so use the no-forecast policy here.
"""
import argparse

import cv2
import numpy as np
import torch

from config import FEATURES_PER_PHASE, MAX_GREEN, MAX_PHASES, MIN_GREEN, PHASE_CAPACITY, RESULTS
from perception.detector import LaneCounter
from rl.ppo import PPO


def build_obs(phases, current, green_time, n_phases):
    o = np.zeros((MAX_PHASES, FEATURES_PER_PHASE), dtype=np.float32)
    for k in range(n_phases):
        veh, ev = phases[k]
        o[k] = [veh / PHASE_CAPACITY, veh / PHASE_CAPACITY, 0, 0, 0, ev,
                1.0 if k == current else 0.0, 1.0]
    return np.concatenate([o.ravel(), [min(green_time / MAX_GREEN, 1.0)]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--lanes", default="perception/lanes_example.json")
    ap.add_argument("--policy", default="models/ppo_noforecast.pt")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--emergency-model", default=None)
    ap.add_argument("--decide-every", type=float, default=5.0, help="seconds of video")
    a = ap.parse_args()

    counter = LaneCounter(a.lanes, a.model, a.emergency_model)
    agent, _ = PPO.load(a.policy)
    n = counter.n_phases
    names = {}
    for l in counter.lanes:
        names[l["phase"]] = l.get("approach", f"phase {l['phase']}")

    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps * a.decide_every))
    out_path = RESULTS / "perception" / "camera_control.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer, i, current, green_t, reason = None, 0, 0, 0.0, "starting"
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            dets, lanes, phases = counter.process(frame)
            green_t += a.decide_every
            ev_phase = next((k for k in range(n) if phases[k, 1] > 0), None)
            if ev_phase is not None:
                current, green_t = ev_phase, 0.0
                reason = f"EMERGENCY OVERRIDE -> {names[ev_phase]}"
            elif green_t >= MIN_GREEN:
                obs = build_obs(phases, current, green_t, n)
                mask = np.zeros(MAX_PHASES, bool)
                mask[:n] = True
                if green_t >= MAX_GREEN and n > 1:
                    mask[current] = False
                act, _, _, probs = agent.act(obs[None], mask[None], deterministic=True)
                new = int(act[0])
                reason = f"RL -> {names[new]} (p={probs[0][new]:.2f})"
                if new != current:
                    current, green_t = new, 0.0
            counts = ", ".join(f"{names[k]}={int(phases[k, 0])}" for k in range(n))
            banner = [f"GREEN: {names[current]}  ({green_t:.0f}s)", reason, f"counts: {counts}"]
            vis = counter.draw(frame, dets, lanes, banner)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(1.0, fps / step), (w, h))
            writer.write(vis)
            print(f"t={i / fps:6.1f}s  {banner[0]} | {reason} | {counts}")
        i += 1
    if writer:
        writer.release()
    print("saved", out_path)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
