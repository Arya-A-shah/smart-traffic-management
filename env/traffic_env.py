"""Multi-intersection traffic signal control environment on top of SUMO.

Each signalised intersection is one agent. The environment is
*asynchronous*: an agent is asked for an action only when its current
green has lasted at least the minimum green time and DELTA seconds have
passed since its previous decision. `reset()` and `step()` return
observations only for the agents that are ready to act.

Observation (per agent, OBS_DIM values), for each of MAX_PHASES phases:
    queue          halted vehicles on the lanes the phase serves
    vehicles       all vehicles on those lanes (incl. arriving platoons)
    ped_waiting    pedestrians waiting at crossings the phase serves
    forecast       forecast inflow for the next minute (0 if disabled)
    upstream_green share of upstream-signal links currently sending
                   traffic toward this phase (shared neighbour info)
    emergency      1 if an emergency vehicle is on the phase's lanes
    is_current     1 for the phase that is green now
    exists         0 for padding phases (used as an action mask)
plus the time the current phase has been green.

Reward (per second, summed between decisions):
    r_i = -(halted vehicles + PED_REWARD_WEIGHT * waiting pedestrians)
    combined_i = r_i + TEAM_REWARD_WEIGHT * mean(r_neighbours)
Integrating the number of halted vehicles over time equals the total
waiting time, so maximising this reward minimises waiting time, while
the neighbour term makes agents care about downstream congestion.

Modes:
    "rl"        agents choose phases (policy supplied by the caller)
    "fixed"     SUMO's fixed-time program (baseline 1)
    "actuated"  SUMO's gap-based actuated program (baseline 2)
"""
import json
from collections import deque

import numpy as np

from config import (DELTA, EPISODE_SECONDS, EV_PENALTY_WEIGHT,
                    FEATURES_PER_PHASE, FORECAST_BIN, FORECAST_HISTORY,
                    MAX_GREEN, MAX_PHASES, NET_ACTUATED, NET_STATIC, OBS_DIM,
                    PED_REWARD_WEIGHT, PHASE_CAPACITY, POTENTIAL_SHAPING_WEIGHT,
                    REWARD_SCALE, TEAM_REWARD_WEIGHT, THROUGHPUT_REWARD_WEIGHT)
from env.intersection import Intersection
from env.safety import SafetyLayer
from sim.generate_demand import scenario_file
from sim.sumo_utils import parse_tripinfo, sumo_command, sumolib, traci


class TrafficEnv:
    def __init__(self, mode="rl", forecaster=None, safety=True, gui=False,
                 log_path=None, collect_inflow=False, episode_seconds=EPISODE_SECONDS,
                 net_file=None):
        assert mode in ("rl", "fixed", "actuated")
        self.mode = mode
        self.forecaster = forecaster
        self.safety = SafetyLayer(enabled=safety and mode == "rl")
        self.gui = gui
        self.log_path = log_path
        self.collect_inflow = collect_inflow or forecaster is not None
        self.episode_seconds = episode_seconds
        self.net_file = net_file or (NET_ACTUATED if mode == "actuated" else NET_STATIC)
        self.net = sumolib.net.readNet(str(self.net_file))
        self.tl_ids = [tl.getID() for tl in self.net.getTrafficLights()]
        self.inters = {}
        self._tripinfo = None

    # ================================================================ API
    def reset(self, seed, route_file=None, tripinfo=None, scale=1.0):
        route_file = route_file or scenario_file(seed, scale)
        self._tripinfo = tripinfo
        cmd = sumo_command(self.net_file, route_file, seed=seed, gui=self.gui,
                           tripinfo=tripinfo)
        if traci.isLoaded():
            traci.close()
        traci.start(cmd)
        tl_set = set(self.tl_ids)
        self.inters = {t: Intersection(t, self.net, tl_set) for t in self.tl_ids}
        for it in self.inters.values():
            if self.mode == "rl":
                it.apply_initial()
        self.t = 0
        self.ev_ids = set()
        self.reward_acc = {t: 0.0 for t in self.tl_ids}
        self.queues = {t: np.zeros(MAX_PHASES) for t in self.tl_ids}
        self.vehicles = {t: np.zeros(MAX_PHASES) for t in self.tl_ids}
        self.ped = {"count": {}, "max_wait": {}, "on": {}}
        self.forecast = {t: np.zeros(MAX_PHASES) for t in self.tl_ids}
        self._seen = {e: set() for it in self.inters.values() for e in it.in_edges}
        self._bin_counts = {t: np.zeros(MAX_PHASES) for t in self.tl_ids}
        self._bin_upg = {t: np.zeros(MAX_PHASES) for t in self.tl_ids}
        self.history = {t: deque([np.zeros(2 * MAX_PHASES + 1)] * FORECAST_HISTORY,
                                 maxlen=FORECAST_HISTORY) for t in self.tl_ids}
        self.inflow_series = {t: [] for t in self.tl_ids}   # for forecaster training
        self.queue_samples, self.arrived = [], 0
        self.overrides = {"emergency": 0, "pedestrian": 0}
        self.prev_pot = {t: 0.0 for t in self.tl_ids}
        self._log = open(self.log_path, "w") if self.log_path else None
        if self._log:
            meta = {tl: {"labels": it.labels, "phase_edges": it.phase_edges}
                    for tl, it in self.inters.items()}
            self._log.write(json.dumps({"type": "meta", "mode": self.mode, "seed": seed,
                                        "net": str(self.net_file), "tls": meta}) + "\n")
        self.decision_meta = {}
        if self.mode != "rl":
            return {}
        return self._advance()

    def step(self, actions, meta=None):
        """actions: {tl_id: phase}. meta: optional {tl_id: dict} for logging."""
        for tl, a in actions.items():
            it = self.inters[tl]
            a = int(a)
            if a >= it.n_phases:
                a = it.current
            changed = it.request(a)
            it.next_decision = self.t + DELTA
            it.last_source = "rl"
            m = (meta or {}).get(tl, {})
            it.last_reason = m.get("reason", f"RL policy chose {it.labels[a]}"
                                   + ("" if changed else " (keep)"))
            self.decision_meta[tl] = m
            self.reward_acc[tl] = 0.0
            self._log_decision(tl, a, "rl", it.last_reason, m)
        return self._advance()

    def run_baseline(self):
        """Run a whole episode under SUMO's own controller."""
        while self.t < self.episode_seconds:
            self._sim_second()
        return self.close()

    def close(self):
        stats = {
            "avg_queue_per_lane": float(np.mean(self.queue_samples)) if self.queue_samples else 0.0,
            "throughput_veh_per_min": self.arrived / (self.t / 60.0) if self.t else 0.0,
            "emergency_overrides": self.overrides["emergency"],
            "pedestrian_overrides": self.overrides["pedestrian"],
        }
        if traci.isLoaded():
            traci.close()
        if self._log:
            self._log.close()
            self._log = None
        if self._tripinfo:
            stats.update(parse_tripinfo(self._tripinfo))
        return stats

    # ============================================================ internals
    def _advance(self):
        """Simulate until at least one agent is ready (or the episode ends)."""
        while True:
            self._sim_second()
            done = self.t >= self.episode_seconds
            ready = [] if done else [tl for tl, it in self.inters.items() if self._is_ready(it)]
            if ready or done:
                obs = {tl: self.observe(tl) for tl in (self.tl_ids if done else ready)}
                rewards = {tl: self.reward_acc[tl] for tl in obs}
                info = {"t": self.t, "masks": {tl: self.action_mask(tl) for tl in obs}}
                return obs, rewards, done, info

    def _is_ready(self, it):
        if it.mode != "green":
            return False
        if it.last_source == "emergency" and self._ev_on_current(it):
            return False                      # hold green while the EV is served
        return (it.green_time >= self.safety.min_green(it, self.ped)
                and self.t >= it.next_decision)

    def _ev_on_current(self, it):
        lanes = set(it.phase_lanes[it.current])
        return any(traci.vehicle.getLaneID(v) in lanes for v in self.ev_ids)

    def _sim_second(self):
        traci.simulationStep()
        self.t += 1
        for v in traci.simulation.getDepartedIDList():
            if traci.vehicle.getTypeID(v) == "emergency":
                self.ev_ids.add(v)
        for v in traci.simulation.getArrivedIDList():
            self.ev_ids.discard(v)
            self.arrived += 1

        self._update_pedestrians()
        inst = {}
        total_halt, n_lanes = 0.0, 0
        for tl, it in self.inters.items():
            if self.mode != "rl":
                it.sync_from_program()
            q = np.zeros(MAX_PHASES)
            n = np.zeros(MAX_PHASES)
            for k, lanes in enumerate(it.phase_lanes):
                for l in lanes:
                    q[k] += traci.lane.getLastStepHaltingNumber(l)
                    n[k] += traci.lane.getLastStepVehicleNumber(l)
                n_lanes += len(lanes)
            self.queues[tl], self.vehicles[tl] = q, n
            total_halt += q.sum()
            ped_wait = sum(self.ped["count"].get(c, 0)
                           for c in {c for cs in it.phase_crossings for c in cs})

            # Emergency vehicle penalty if stopped on approaching lanes
            ev_halted = 0
            for v in self.ev_ids:
                try:
                    if traci.vehicle.getSpeed(v) < 1.0:
                        v_lane = traci.vehicle.getLaneID(v)
                        for k, lanes in enumerate(it.phase_lanes):
                            if v_lane in lanes and k != it.current:
                                ev_halted += 1
                except Exception:
                    pass

            # Core step reward
            r_step = -(q.sum() + PED_REWARD_WEIGHT * ped_wait + EV_PENALTY_WEIGHT * ev_halted)
            # Throughput bonus: reward clearing vehicles during green
            if it.mode == "green" and q[it.current] < n[it.current]:
                r_step += THROUGHPUT_REWARD_WEIGHT * (n[it.current] - q[it.current]) * 0.1

            # Potential-based reward shaping: gamma * Phi(s') - Phi(s) where Phi(s) = -sum(q^2)
            cur_pot = -float(np.sum(q ** 2))
            pot_diff = 0.95 * cur_pot - self.prev_pot.get(tl, cur_pot)
            self.prev_pot[tl] = cur_pot
            r_step += POTENTIAL_SHAPING_WEIGHT * pot_diff * 0.01

            inst[tl] = r_step * REWARD_SCALE
        self.queue_samples.append(total_halt / max(n_lanes, 1))

        for tl, it in self.inters.items():
            nb = [inst[n] for n in it.neighbors]
            self.reward_acc[tl] += inst[tl] + (TEAM_REWARD_WEIGHT * np.mean(nb) if nb else 0.0)

        if self.collect_inflow:
            self._count_inflow()

        if self.mode == "rl":
            for tl, it in self.inters.items():
                forced = self.safety.check(it, self.ev_ids, self.ped, self.queues[tl])
                if forced is not None:
                    k, reason = forced
                    target = it.target if it.mode != "green" else it.current
                    if k != target:
                        it.request(k)
                        kind = "emergency" if reason.startswith("EMERGENCY") else "pedestrian"
                        self.overrides[kind] += 1
                        it.last_source, it.last_reason = kind, reason
                        self._log_decision(tl, k, kind, reason, {})
                    elif reason.startswith("EMERGENCY"):
                        it.last_source, it.last_reason = "emergency", reason
                        it.next_decision = self.t + 1        # hold while EV passes
                it.tick()

        if self._log and self.t % 5 == 0:
            self._log_snapshot()

    def _update_pedestrians(self):
        count, max_wait, on = {}, {}, {}
        for p in traci.person.getIDList():
            road = traci.person.getRoadID(p)
            if road.startswith(":") and "_c" in road:
                on[road] = on.get(road, 0) + 1
                continue
            w = traci.person.getWaitingTime(p)
            if w > 0:
                c = traci.person.getNextEdge(p)
                if c:
                    count[c] = count.get(c, 0) + 1
                    max_wait[c] = max(max_wait.get(c, 0), w)
        self.ped = {"count": count, "max_wait": max_wait, "on": on}

    def _upstream_green(self, it):
        out = np.zeros(MAX_PHASES)
        for k, ups in enumerate(it.upstream):
            fr = []
            for up, idx in ups:
                s = traci.trafficlight.getRedYellowGreenState(up)
                if idx:
                    fr.append(sum(s[i] in "Gg" for i in idx) / len(idx))
            out[k] = np.mean(fr) if fr else 0.0
        return out

    def _count_inflow(self):
        for tl, it in self.inters.items():
            for k, edges in enumerate(it.phase_edges):
                for e in edges:
                    ids = set(traci.edge.getLastStepVehicleIDs(e))
                    self._bin_counts[tl][k] += len(ids - self._seen[e])
                    self._seen[e] = ids
            self._bin_upg[tl] += self._upstream_green(it)
        if self.t % FORECAST_BIN == 0:
            for tl in self.tl_ids:
                feat = np.concatenate([self._bin_counts[tl], self._bin_upg[tl] / FORECAST_BIN,
                                       [self.t / self.episode_seconds]])
                self.history[tl].append(feat)
                self.inflow_series[tl].append(feat)
                self._bin_counts[tl] = np.zeros(MAX_PHASES)
                self._bin_upg[tl] = np.zeros(MAX_PHASES)
            if self.forecaster is not None:
                batch = np.stack([np.stack(self.history[tl]) for tl in self.tl_ids])
                pred = self.forecaster.predict(batch)          # (n_tl, horizon, P)
                for i, tl in enumerate(self.tl_ids):
                    self.forecast[tl] = pred[i].sum(axis=0)

    # ------------------------------------------------------------ observation
    def observe(self, tl):
        it = self.inters[tl]
        up = self._upstream_green(it)
        ev = np.zeros(MAX_PHASES)
        for v in self.ev_ids:
            lane = traci.vehicle.getLaneID(v)
            for k, lanes in enumerate(it.phase_lanes):
                if lane in lanes:
                    ev[k] = 1.0
        o = np.zeros((MAX_PHASES, FEATURES_PER_PHASE), dtype=np.float32)
        for k in range(it.n_phases):
            peds = sum(self.ped["count"].get(c, 0) for c in it.phase_crossings[k])
            o[k] = [self.queues[tl][k] / PHASE_CAPACITY,
                    self.vehicles[tl][k] / PHASE_CAPACITY,
                    min(peds / 10.0, 1.0),
                    self.forecast[tl][k] / 20.0,
                    up[k], ev[k],
                    1.0 if k == it.current else 0.0, 1.0]
        return np.concatenate([o.ravel(), [min(it.green_time / MAX_GREEN, 1.0)]]).astype(np.float32)

    def action_mask(self, tl):
        it = self.inters[tl]
        m = np.zeros(MAX_PHASES, dtype=bool)
        m[:it.n_phases] = True
        if it.green_time >= MAX_GREEN and it.n_phases > 1:
            m[it.current] = False           # fairness: force a change after MAX_GREEN
        return m

    # ------------------------------------------------------------ logging
    def _log_decision(self, tl, phase, source, reason, meta):
        if not self._log:
            return
        rec = {"type": "decision", "t": self.t, "tl": tl, "phase": int(phase),
               "label": self.inters[tl].labels[phase], "source": source, "reason": reason}
        for k in ("probs", "attribution", "value"):
            if k in meta:
                rec[k] = meta[k]
        self._log.write(json.dumps(rec) + "\n")

    def _log_snapshot(self):
        tls = {}
        for tl, it in self.inters.items():
            tls[tl] = {"phase": it.current, "label": it.labels[it.current], "mode": it.mode,
                       "green_time": it.green_time,
                       "queues": self.queues[tl][:it.n_phases].tolist(),
                       "forecast": self.forecast[tl][:it.n_phases].round(2).tolist(),
                       "labels": it.labels,
                       "reason": it.last_reason, "source": it.last_source}
        edges = {e: traci.edge.getLastStepHaltingNumber(e)
                 for it in self.inters.values() for e in it.in_edges}
        rec = {"type": "snapshot", "t": self.t, "tls": tls, "edges": edges,
               "halting": float(sum(q.sum() for q in self.queues.values())),
               "running": traci.vehicle.getIDCount(), "arrived": self.arrived,
               "emergency": [[v, *map(float, traci.vehicle.getPosition(v))]
                             for v in sorted(self.ev_ids)]}
        self._log.write(json.dumps(rec) + "\n")
