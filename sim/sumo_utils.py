"""Helpers for locating SUMO, starting simulations and reading outputs."""
import os
import xml.etree.ElementTree as ET

# Works both with `pip install eclipse-sumo` and a system-wide SUMO install.
if "SUMO_HOME" not in os.environ:
    try:
        import sumo  # provided by the eclipse-sumo pip package
        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
    except ImportError:
        pass

import sumolib  # noqa: E402
import traci    # noqa: E402


def sumo_binary(gui: bool = False) -> str:
    return sumolib.checkBinary("sumo-gui" if gui else "sumo")


def sumo_command(net_file, route_file, seed=0, gui=False, tripinfo=None,
                 end=None):
    cmd = [
        sumo_binary(gui),
        "-n", str(net_file),
        "-r", str(route_file),
        "--seed", str(seed),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--time-to-teleport", "300",
        "--waiting-time-memory", "10000",
        "--device.emissions.probability", "1",
        "--tripinfo-output.write-unfinished", "true",
    ]
    if end is not None:
        cmd += ["--end", str(end)]
    if tripinfo:
        cmd += ["--tripinfo-output", str(tripinfo)]
    if gui:
        cmd += ["--start", "true", "--quit-on-end", "true", "--delay", "50"]
    return cmd


def parse_tripinfo(path):
    """Aggregate per-vehicle / per-pedestrian statistics from tripinfo XML."""
    root = ET.parse(path).getroot()
    wait, loss, fuel, co2 = [], [], [], []
    ev_loss, ev_wait = [], []
    ped_loss = []
    for trip in root.iter("tripinfo"):
        w = float(trip.get("waitingTime"))
        tl = float(trip.get("timeLoss"))
        em = trip.find("emissions")
        if trip.get("vType") == "emergency":
            ev_loss.append(tl)
            ev_wait.append(w)
            continue
        wait.append(w)
        loss.append(tl)
        if em is not None:
            fuel.append(float(em.get("fuel_abs")) / 1000.0)   # mg -> g
            co2.append(float(em.get("CO2_abs")) / 1000.0)
    for person in root.iter("personinfo"):
        ped_loss.append(sum(float(w.get("timeLoss", 0))
                            for w in person.iter("walk")))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "vehicles": len(wait),
        "avg_waiting_time_s": mean(wait),
        "avg_time_loss_s": mean(loss),
        "avg_fuel_g": mean(fuel),
        "total_co2_kg": sum(co2) / 1000.0,
        "emergency_vehicles": len(ev_loss),
        "emergency_time_loss_s": mean(ev_loss),
        "emergency_waiting_s": mean(ev_wait),
        "pedestrians": len(ped_loss),
        "ped_time_loss_s": mean(ped_loss),
    }
