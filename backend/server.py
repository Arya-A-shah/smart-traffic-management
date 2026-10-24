"""Backend for the Godot traffic game.

    python backend/server.py            (then open http://127.0.0.1:8765)

What it does
  * The game asks it every simulated second: "which approach gets green?"
    -> it runs the chosen algorithm + the ambulance rule (algorithms.py)
  * It records everything the game reports -> backend/runs/<run>/ (CSV + JSON)
  * It draws the graphs (plots.py) and serves a live dashboard page
  * "Compare" replays the same traffic under every algorithm (minisim.py)

Only needs: Python 3.9+, matplotlib. (torch is optional - for the RL agent.)
"""
import csv
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plots  # noqa: E402
from algorithms import ALGORITHMS, NAMES, decide, rl_available  # noqa: E402
from minisim import compare  # noqa: E402

PORT = 8765
RUNS_DIR = HERE / "runs"
LOCK = threading.Lock()


class Run:
    counter = 0

    def __init__(self, algo, scenario, ev_priority):
        Run.counter += 1
        self.n = Run.counter
        self.algo, self.scenario, self.ev_priority = algo, scenario, ev_priority
        self.started = time.strftime("%Y%m%d_%H%M%S")
        self.dir = RUNS_DIR / f"{self.started}_{self.n:02d}_{algo}"
        self.timeseries, self.vehicles, self.decisions = [], [], []
        self.last_phase = None
        self.last_t = 0

    def summary(self):
        cars = [v for v in self.vehicles if v["type"] != "ambulance"]
        amb = [v for v in self.vehicles if v["type"] == "ambulance"]
        return {
            "run": self.n, "algo": self.algo, "algo_name": ALGORITHMS[self.algo]["name"],
            "ev_priority": self.ev_priority, "sim_time": self.last_t,
            "vehicles_passed": len(self.vehicles),
            "avg_wait": sum(v["wait"] for v in cars) / len(cars) if cars else 0.0,
            "max_wait": max([v["wait"] for v in cars] or [0.0]),
            "ambulances_passed": len(amb),
            "amb_wait": sum(v["wait"] for v in amb) / len(amb) if amb else None,
            "max_queue": max([sum(r["queue"]) for r in self.timeseries] or [0]),
            "switches": sum(1 for d in self.decisions if d["changed"]),
            "emergency_overrides": sum(1 for d in self.decisions if d["source"] == "emergency" and d["changed"]),
            "folder": str(self.dir),
        }

    def save(self, with_plots=True):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "timeseries.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "green", "green_time"] + [f"queue_{n}" for n in NAMES] + [f"approaching_{n}" for n in NAMES])
            for r in self.timeseries:
                w.writerow([r["t"], NAMES[r["current"]], r["green_time"], *r["queue"], *r["approaching"]])
        with open(self.dir / "vehicles.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "type", "from", "turn", "spawn_t", "exit_t", "wait_s", "travel_s"])
            for v in self.vehicles:
                w.writerow([v.get("id"), v["type"], NAMES[v["dir"]], v.get("turn", ""), v.get("spawn_t"),
                            v.get("exit_t"), round(v["wait"], 2), round(v.get("travel", 0), 2)])
        with open(self.dir / "decisions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "green", "changed", "source", "reason"])
            for d in self.decisions:
                w.writerow([d["t"], NAMES[d["phase"]], d["changed"], d["source"], d["reason"]])
        (self.dir / "summary.json").write_text(json.dumps(
            {**self.summary(), "scenario": self.scenario}, indent=2))
        if with_plots:
            (self.dir / "queues.png").write_bytes(plots.live_queues(self))
            (self.dir / "waits_by_type.png").write_bytes(plots.live_waits(self))


STATE = {"run": None, "runs": [], "last_scenario": None, "last_compare": None}


def current_run():
    return STATE["run"]


def handle_start(body):
    algo = body.get("algo", "fixed")
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algo {algo}")
    with LOCK:
        if STATE["run"] and STATE["run"].timeseries:
            STATE["run"].save()
        run = Run(algo, body.get("scenario", {}), bool(body.get("ev_priority", True)))
        STATE["run"] = run
        STATE["runs"].append(run)
        STATE["last_scenario"] = body.get("scenario")
        STATE["last_compare"] = None
    return {"run": run.n, "algo": algo}


def handle_decide(body):
    s = body["state"]
    run = current_run()
    algo = body.get("algo") or (run.algo if run else "fixed")
    if run is None or algo != run.algo:   # no run yet / algorithm changed -> open a new run
        handle_start({"algo": algo, "ev_priority": body.get("ev_priority", True),
                      "scenario": STATE["last_scenario"] or {}})
        run = current_run()
    ev_priority = body.get("ev_priority", run.ev_priority if run else True)
    phase, reason, source = decide(algo, s, ev_priority)
    if run:
        with LOCK:
            run.ev_priority = ev_priority
            run.last_t = s["t"]
            run.timeseries.append({k: s[k] for k in ("t", "current", "green_time", "queue", "approaching")})
            run.vehicles.extend(body.get("completed", []))
            changed = phase != s["current"]
            if changed or source == "emergency" and run.last_phase != ("ev", phase):
                run.decisions.append({"t": s["t"], "phase": phase, "changed": changed,
                                      "source": source, "reason": reason})
            run.last_phase = ("ev", phase) if source == "emergency" else phase
            if len(run.timeseries) % 30 == 0:       # autosave the CSVs every 30 s
                run.save(with_plots=False)
    return {"phase": phase, "reason": reason, "source": source,
            "summary": run.summary() if run else None}


def handle_compare(body):
    scenario = body.get("scenario") or STATE["last_scenario"] or {
        "initial": [6, 3, 6, 3], "flows": [6, 3, 6, 3], "random_ambulances": True, "seed": 7}
    scenario = {**scenario, "duration": int(body.get("duration", 300))}
    algos = [a for a in ALGORITHMS if a != "rl" or rl_available()]
    ev = bool(body.get("ev_priority", True))
    res = compare(scenario, algos, ev)
    png = plots.compare_chart(res, "" if ev else "  (ambulance rule OFF)")
    table = [{k: r[k] for k in ("algo", "avg_wait", "max_wait", "amb_wait", "max_queue", "vehicles")} for r in res]
    STATE["last_compare"] = (png, table)
    RUNS_DIR.mkdir(exist_ok=True)
    (RUNS_DIR / "last_compare.png").write_bytes(png)
    (RUNS_DIR / "last_compare.json").write_text(json.dumps({"scenario": scenario, "results": table}, indent=2))
    return png, table


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):          # keep the console quiet
        pass

    def _send(self, code, data, ctype="application/json"):
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        elif isinstance(data, str):
            data = data.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        try:
            if p == "/":
                self._send(200, (HERE / "dashboard.html").read_text(), "text/html; charset=utf-8")
            elif p == "/api/status":
                self._send(200, {"ok": True, "rl_available": rl_available(),
                                 "algorithms": {k: v for k, v in ALGORITHMS.items()}})
            elif p == "/api/summary":
                run = current_run()
                self._send(200, {"current": run.summary() if run else None,
                                 "runs": [r.summary() for r in STATE["runs"] if r.vehicles],
                                 "decisions": run.decisions[-12:][::-1] if run else [],
                                 "compare": STATE["last_compare"][1] if STATE["last_compare"] else None})
            elif p == "/plot/queues.png":
                run = current_run()
                self._send(200, plots.live_queues(run) if run else plots._empty("Start the game to see live data"), "image/png")
            elif p == "/plot/waits.png":
                run = current_run()
                self._send(200, plots.live_waits(run) if run else plots._empty("No data yet"), "image/png")
            elif p == "/plot/runs.png":
                self._send(200, plots.runs_compare(STATE["runs"]), "image/png")
            elif p == "/plot/compare.png":
                q = parse_qs(u.query)
                if STATE["last_compare"] and "refresh" not in q:
                    png = STATE["last_compare"][0]
                else:
                    png, _ = handle_compare({"ev_priority": q.get("ev", ["1"])[0] == "1"})
                self._send(200, png, "image/png")
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            body = self._body()
            if p == "/api/start":
                self._send(200, handle_start(body))
            elif p == "/api/decide":
                self._send(200, handle_decide(body))
            elif p == "/api/stop":
                run = current_run()
                if run and run.timeseries:
                    with LOCK:
                        run.save()
                self._send(200, {"saved": str(run.dir) if run else None,
                                 "summary": run.summary() if run else None})
            elif p == "/api/compare":
                png, _ = handle_compare(body)
                self._send(200, png, "image/png")
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    RUNS_DIR.mkdir(exist_ok=True)
    print("Loading RL agent ...", "ok" if rl_available() else "not available (install torch) - other algorithms work")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Traffic backend running:  http://127.0.0.1:{PORT}   (Ctrl+C to stop)")
    print(f"Run data is saved in:     {RUNS_DIR}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        run = current_run()
        if run and run.timeseries:
            run.save()
        print("\nstopped")


if __name__ == "__main__":
    main()
