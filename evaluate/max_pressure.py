"""Max-pressure signal control (Varaiya, 2013) - no learning, no training.

Idea in one line: give green to the phase whose cars have the most "pressure",
    pressure(phase) = sum over its movements of
                      (cars queued on the incoming lane - cars on the outgoing lane)

Why it coordinates intersections without talking to them:
if the road *after* the light is already full (the next intersection is
congested), the pressure of that movement drops, so this junction stops
sending cars into a jam. Each agent only looks at its own in/out lanes,
yet the network behaves cooperatively.

It plugs into the same environment as the RL agent, so the same safety layer
(emergency-vehicle priority, pedestrian limits) sits on top of it.
"""
import numpy as np

from sim.sumo_utils import traci


class MaxPressureController:
    use_forecast = False

    def pressures(self, it):
        p = np.zeros(it.n_phases)
        for k, state in enumerate(it.green):
            seen = set()
            for i, frm, to in it.vehicle_links:
                if state[i] not in "Gg" or (frm, to) in seen:
                    continue
                seen.add((frm, to))
                q_in = traci.lane.getLastStepHaltingNumber(frm)
                q_out = traci.lane.getLastStepVehicleNumber(to)
                p[k] += q_in - q_out
        return p

    def decide(self, env, obs, masks, explain=False):
        actions, meta = {}, {}
        for tl in obs:
            it = env.inters[tl]
            p = self.pressures(it)
            m = masks[tl][:it.n_phases]
            p_masked = np.where(m, p, -np.inf)
            a = int(np.argmax(p_masked))
            actions[tl] = a
            verb = "keep" if a == it.current else "switch to"
            meta[tl] = {"reason": f"MAX-PRESSURE: {verb} {it.labels[a]} "
                                  f"(pressure {p[a]:.0f} vs others "
                                  f"{', '.join(f'{x:.0f}' for j, x in enumerate(p) if j != a)})"}
        return actions, meta
