"""Rule-based safety layer that sits *above* the learned policy.

The learned policy is never trusted with safety-critical decisions. Every
second, before the policy is consulted, these rules are checked in order:

 1. Emergency-vehicle priority: if an emergency vehicle is within
    EV_DETECTION_RANGE of the stop line, the phase serving its lane is
    forced green (through the normal clearance + yellow) and held until
    the vehicle has passed.
 2. Pedestrian minimum green: while pedestrians are on or waiting at a
    crossing that the current phase serves, the phase cannot end before
    PED_MIN_GREEN seconds.
 3. Pedestrian maximum wait: no pedestrian waits longer than PED_MAX_WAIT;
    once exceeded, a phase that serves their crossing is forced next.

In the simulation, "emergency vehicle present" is read from SUMO. In a
real deployment the same flag comes from the perception module's
emergency-vehicle class (see perception/).
"""
from config import EV_DETECTION_RANGE, MIN_GREEN, PED_MAX_WAIT, PED_MIN_GREEN
from sim.sumo_utils import traci


class SafetyLayer:
    def __init__(self, enabled=True):
        self.enabled = enabled

    # -------------------------------------------------------------- rule 1
    def emergency_phase(self, inter, ev_ids):
        """Return (phase, reason) for the nearest approaching EV, else None."""
        best = None
        for v in ev_ids:
            lane = traci.vehicle.getLaneID(v)
            for k, lanes in enumerate(inter.phase_lanes):
                if lane in lanes:
                    dist = inter.lane_length[lane] - traci.vehicle.getLanePosition(v)
                    if dist <= EV_DETECTION_RANGE and (best is None or dist < best[0]):
                        best = (dist, k, v)
        if best is None:
            return None
        dist, k, v = best
        return k, (f"EMERGENCY OVERRIDE: {v} approaching from {inter.labels[k]} "
                   f"({dist:.0f} m) -> priority green")

    # -------------------------------------------------------------- rule 2
    def min_green(self, inter, ped):
        """Minimum green for the current phase given pedestrians present."""
        if not self.enabled:
            return MIN_GREEN
        for c in inter.phase_crossings[inter.current]:
            if ped["on"].get(c, 0) or ped["count"].get(c, 0):
                return PED_MIN_GREEN
        return MIN_GREEN

    # -------------------------------------------------------------- rule 3
    def pedestrian_phase(self, inter, ped, queues):
        worst = None
        for k, crossings in enumerate(inter.phase_crossings):
            for c in crossings:
                w = ped["max_wait"].get(c, 0)
                if w > PED_MAX_WAIT and (worst is None or w > worst[0]):
                    worst = (w, c)
        if worst is None:
            return None
        w, c = worst
        # of the phases that serve this crossing, pick the busiest for vehicles
        options = [k for k, cs in enumerate(inter.phase_crossings) if c in cs]
        if inter.current in options:
            return None
        k = max(options, key=lambda j: queues[j])
        return k, (f"PEDESTRIAN OVERRIDE: pedestrian waiting {w:.0f}s at crossing "
                   f"-> serve {inter.labels[k]}")

    def check(self, inter, ev_ids, ped, queues):
        """Return (forced_phase, reason) or None."""
        if not self.enabled:
            return None
        ev = self.emergency_phase(inter, ev_ids)
        if ev is not None:
            return ev
        if inter.mode == "green" and inter.green_time >= MIN_GREEN:
            return self.pedestrian_phase(inter, ped, queues)
        return None
