"""Generate a randomised but realistic one-hour demand scenario.

Each scenario (identified by a seed) has:
  * directional asymmetry  - every entry road gets its own base flow;
  * a peak-hour profile    - demand rises and falls around a random peak;
  * surge events           - short bursts on one entry (e.g. a school or
                             office letting out). These are what the
                             forecasting module learns to anticipate;
  * a vehicle mix          - cars, motorcycles, buses, trucks;
  * emergency vehicles     - a few ambulances/fire trucks per hour;
  * pedestrians            - walking trips that use the crossings.

Training and evaluation use different seeds, so the RL policy is always
evaluated on traffic it has never seen.

Usage:  python -m sim.generate_demand --seed 42 --out sim/data/demo.rou.xml
"""
import argparse
import math
from pathlib import Path

import numpy as np

from config import EPISODE_SECONDS, NET_STATIC, SIM_DATA
from sim.sumo_utils import sumolib

VTYPES = """
    <vType id="car" vClass="passenger" length="4.5" accel="2.6" decel="4.5" sigma="0.5" color="1,1,0"/>
    <vType id="motorcycle" vClass="motorcycle" length="2.2" accel="3.5" decel="5" sigma="0.5" color="0,1,1"/>
    <vType id="bus" vClass="bus" length="12" accel="1.2" decel="4" sigma="0.5" color="0,0.6,0"/>
    <vType id="truck" vClass="truck" length="9" accel="1.3" decel="4" sigma="0.5" color="0.6,0.3,0"/>
    <vType id="emergency" vClass="emergency" guiShape="emergency" length="6" accel="3" decel="5"
           speedFactor="1.3" sigma="0.2" color="1,0,0"/>
    <vType id="ped" vClass="pedestrian"/>
"""
MIX = [("car", 0.78), ("motorcycle", 0.08), ("bus", 0.04), ("truck", 0.10)]


def fringe_edges(net):
    entries, exits = [], []
    for e in net.getEdges():
        if not e.allows("passenger"):
            continue
        if e.getFromNode().getType() == "dead_end":
            entries.append(e)
        if e.getToNode().getType() == "dead_end":
            exits.append(e)
    return entries, exits


def walkable_edges(net):
    return [e for e in net.getEdges() if e.allows("pedestrian") and not e.isSpecial()]


def generate(seed, out_path, duration=EPISODE_SECONDS, scale=1.0, net_file=NET_STATIC):
    rng = np.random.default_rng(seed)
    net = sumolib.net.readNet(str(net_file))
    entries, exits = fringe_edges(net)

    # --- demand profile --------------------------------------------------
    base = {e.getID(): rng.uniform(120, 380) * scale for e in entries}  # veh/h
    peak_t = rng.uniform(0.35, 0.65) * duration
    sigma = rng.uniform(0.15, 0.3) * duration
    surges = []
    for _ in range(rng.integers(2, 5)):
        start = rng.uniform(0, duration - 600)
        surges.append((rng.choice(list(base)), start, start + rng.uniform(240, 600),
                       rng.uniform(1.8, 2.6)))

    def rate(edge_id, t):
        r = base[edge_id] * (0.55 + 0.75 * math.exp(-((t - peak_t) / sigma) ** 2))
        for e, s, f, m in surges:
            if e == edge_id and s <= t < f:
                r *= m
        return r

    items = []  # (depart, xml)
    vid = 0
    types, probs = zip(*MIX)
    for t in range(duration):
        for e in entries:
            if rng.random() < rate(e.getID(), t) / 3600.0:
                choices = [x for x in exits if x.getToNode() != e.getFromNode()]
                dest = choices[rng.integers(len(choices))]
                vtype = rng.choice(types, p=probs)
                items.append((t, f'    <trip id="v{vid}" type="{vtype}" depart="{t}" '
                                 f'from="{e.getID()}" to="{dest.getID()}" '
                                 f'departLane="best" departSpeed="max"/>'))
                vid += 1

    # --- emergency vehicles ----------------------------------------------
    for i in range(rng.integers(3, 7)):
        t = int(rng.uniform(120, duration - 300))
        e = entries[rng.integers(len(entries))]
        choices = [x for x in exits if x.getToNode() != e.getFromNode()]
        dest = choices[rng.integers(len(choices))]
        items.append((t, f'    <trip id="ev{i}" type="emergency" depart="{t}" '
                         f'from="{e.getID()}" to="{dest.getID()}" '
                         f'departLane="best" departSpeed="max"/>'))

    # --- pedestrians -----------------------------------------------------
    walk = walkable_edges(net)
    ped_rate = rng.uniform(150, 350) * scale  # persons/h
    pid = 0
    for t in range(duration):
        if rng.random() < ped_rate / 3600.0:
            a, b = rng.choice(len(walk), 2, replace=False)
            items.append((t, f'    <person id="p{pid}" type="ped" depart="{t}">\n'
                             f'        <walk from="{walk[a].getID()}" to="{walk[b].getID()}"/>\n'
                             f'    </person>'))
            pid += 1

    items.sort(key=lambda x: x[0])
    Path(out_path).write_text("<routes>" + VTYPES + "\n".join(x for _, x in items)
                              + "\n</routes>\n")
    return {"vehicles": vid, "pedestrians": pid, "surges": surges, "peak_t": peak_t}


def scenario_file(seed, scale=1.0):
    """Return the route file for a seed, generating it on first use."""
    path = SIM_DATA / "routes" / f"scenario_{seed}_{scale:g}.rou.xml"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        generate(seed, path, scale=scale)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="demand multiplier")
    ap.add_argument("--out", default=str(SIM_DATA / "demo.rou.xml"))
    a = ap.parse_args()
    info = generate(a.seed, a.out, scale=a.scale)
    print(f"wrote {a.out}: {info['vehicles']} vehicles, {info['pedestrians']} pedestrians")
