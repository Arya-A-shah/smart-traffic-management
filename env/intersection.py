"""One signalised intersection: its geometry, its phases and its signal
state machine (green -> pedestrian clearance -> yellow -> next green).

The RL agent never writes raw signal strings. It only chooses *which
phase* should be green next; this class inserts the clearance and yellow
intervals so every transition is safe regardless of what the policy
outputs.
"""
from config import CLEARANCE, MAX_PHASES, YELLOW
from sim.sumo_utils import traci


def _direction(net, edge_id):
    """Human-readable compass direction an approach comes *from*."""
    e = net.getEdge(edge_id)
    (x1, y1), (x2, y2) = e.getFromNode().getCoord(), e.getToNode().getCoord()
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) >= abs(dy):
        return "West" if dx > 0 else "East"
    return "South" if dy > 0 else "North"


class Intersection:
    def __init__(self, tl_id, net, tl_ids):
        self.id = tl_id
        links = traci.trafficlight.getControlledLinks(tl_id)
        logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
        states = [p.state for p in logic.phases]

        # ---- classify controlled links -----------------------------------
        self.vehicle_links, self.crossing_of_link = [], {}
        for i, link in enumerate(links):
            if not link:
                continue
            frm, to, _via = link[0]
            if to.startswith(":"):                     # walkingarea -> crossing
                self.crossing_of_link[i] = to.rsplit("_", 1)[0]
            else:
                self.vehicle_links.append((i, frm, to))

        # ---- split the program into [green, (clearance), yellow] blocks ----
        blocks, cur = [], []
        for s in states:
            cur.append(s)
            if "y" in s:
                blocks.append(cur)
                cur = []
        if not blocks:
            raise ValueError(f"{tl_id}: program has no yellow phases")
        if len(blocks) > MAX_PHASES:
            raise ValueError(f"{tl_id} has {len(blocks)} phases > MAX_PHASES={MAX_PHASES}")
        self.green = [b[0] for b in blocks]
        self.clear = [self._crossings_red(b[0]) for b in blocks]
        self.yellow = [b[-1] for b in blocks]
        self.n_phases = len(self.green)

        # map SUMO program phase index -> our phase index (for baselines)
        self.program_phase_to_block = []
        for k, b in enumerate(blocks):
            self.program_phase_to_block += [k] * len(b)

        # ---- what each phase serves ------------------------------------
        self.phase_lanes, self.phase_edges, self.phase_crossings = [], [], []
        for g in self.green:
            lanes = sorted({frm for i, frm, _ in self.vehicle_links if g[i] in "Gg"})
            self.phase_lanes.append(lanes)
            self.phase_edges.append(sorted({l.rsplit("_", 1)[0] for l in lanes}))
            self.phase_crossings.append(sorted({c for i, c in self.crossing_of_link.items()
                                                if g[i] in "Gg"}))
        self.lane_length = {l: traci.lane.getLength(l) for ls in self.phase_lanes for l in ls}
        self.in_edges = sorted({e for es in self.phase_edges for e in es})
        self.labels = [" + ".join(_direction(net, e) for e in es) or f"phase {k}"
                       for k, es in enumerate(self.phase_edges)]

        # ---- neighbours and upstream links (for coordination) ------------
        self.neighbors = set()
        for e in self.in_edges:
            n = net.getEdge(e).getFromNode().getID()
            if n in tl_ids and n != tl_id:
                self.neighbors.add(n)
        for e in net.getNode(tl_id).getOutgoing():
            n = e.getToNode().getID()
            if n in tl_ids and n != tl_id:
                self.neighbors.add(n)
        self.neighbors = sorted(self.neighbors)
        # upstream[k] = list of (upstream_tl, [link indices feeding phase k's edges])
        self.upstream = [[] for _ in range(self.n_phases)]
        for k, edges in enumerate(self.phase_edges):
            for e in edges:
                up = net.getEdge(e).getFromNode().getID()
                if up in tl_ids:
                    idx = [i for i, l in enumerate(traci.trafficlight.getControlledLinks(up))
                           if l and l[0][1].rsplit("_", 1)[0] == e]
                    self.upstream[k].append((up, idx))

        self.reset_state()

    def _crossings_red(self, state):
        s = list(state)
        for i in self.crossing_of_link:
            s[i] = "r"
        return "".join(s)

    # ------------------------------------------------------------------
    def reset_state(self):
        self.current = 0
        self.mode = "green"          # green | clear | yellow
        self.mode_time = 0
        self.green_time = 0
        self.target = None
        self.next_decision = 0
        self.last_reason = "start"
        self.last_source = "init"

    def apply_initial(self):
        traci.trafficlight.setRedYellowGreenState(self.id, self.green[0])

    def request(self, k):
        """Ask for phase k. Starts a safe transition if needed."""
        if self.mode == "green":
            if k == self.current:
                return False
            self.target = k
            self.mode, self.mode_time = "clear", 0
            traci.trafficlight.setRedYellowGreenState(self.id, self.clear[self.current])
            return True
        if self.mode == "clear" and k == self.current:    # abort: back to green
            self.mode, self.target = "green", None
            traci.trafficlight.setRedYellowGreenState(self.id, self.green[self.current])
            return True
        self.target = k
        return True

    def tick(self):
        """Advance the signal state machine by one second."""
        self.mode_time += 1
        if self.mode == "green":
            self.green_time += 1
        elif self.mode == "clear" and self.mode_time >= CLEARANCE:
            self.mode, self.mode_time = "yellow", 0
            traci.trafficlight.setRedYellowGreenState(self.id, self.yellow[self.current])
        elif self.mode == "yellow" and self.mode_time >= YELLOW:
            self.current, self.target = self.target, None
            self.mode, self.mode_time, self.green_time = "green", 0, 0
            traci.trafficlight.setRedYellowGreenState(self.id, self.green[self.current])

    def sync_from_program(self):
        """Baselines: read the phase SUMO's own controller is showing."""
        p = traci.trafficlight.getPhase(self.id)
        self.current = self.program_phase_to_block[p]
        s = traci.trafficlight.getRedYellowGreenState(self.id)
        self.mode = "yellow" if "y" in s else "green"
