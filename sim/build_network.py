"""Build the road network used by all experiments.

Default: a 2x2 grid of four signalised four-arm intersections with
two lanes per direction, sidewalks and pedestrian crossings.

Supports:
- --grid-size 2 or 3: 2x2 (4 junctions) or 3x3 (9 junctions) grid networks
- --osm map.osm: imports a real area exported from OpenStreetMap

Both variants are written twice: once with fixed-time signal programs
and once with SUMO's actuated programs (the sensor-based baseline).
Signals use the "incoming" layout: each phase gives green to one
approach, so the RL agent chooses *which approach* is served next.

Usage:
    python -m sim.build_network
    python -m sim.build_network --grid-size 3
    python -m sim.build_network --osm path/to/map.osm
"""
import argparse
import subprocess

from config import NET_ACTUATED, NET_STATIC, NET_ACTUATED_3X3, NET_STATIC_3X3
from sim.sumo_utils import sumolib

COMMON_TLS = [
    "--tls.layout", "incoming",
    "--sidewalks.guess", "true",
    "--crossings.guess", "true",
    "--no-turnarounds", "true"
]


def build_grid(out, tls_type, grid_number=2, length=200):
    cmd = [
        sumolib.checkBinary("netgenerate"),
        "--grid",
        "--grid.number", str(grid_number),
        "--grid.length", str(length),
        "--grid.attach-length", str(length),
        "--default.lanenumber", "2",
        "--default.speed", "13.89",
        "--tls.guess", "true",
        "--tls.default-type", tls_type,
        "-o", str(out)
    ] + COMMON_TLS
    subprocess.run(cmd, check=True)


def build_osm(osm_file, out, tls_type):
    cmd = [
        sumolib.checkBinary("netconvert"),
        "--osm-files", str(osm_file),
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join",
        "--remove-edges.isolated",
        "--keep-edges.by-vclass", "passenger,pedestrian",
        "--tls.default-type", tls_type,
        "-o", str(out)
    ] + COMMON_TLS
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", help="optional .osm file to import instead of the grid")
    ap.add_argument("--grid-size", type=int, choices=[2, 3], default=2,
                    help="Grid dimensions: 2 (2x2 grid, 4 junctions) or 3 (3x3 grid, 9 junctions)")
    args = ap.parse_args()

    targets = (
        (NET_STATIC_3X3, NET_ACTUATED_3X3) if args.grid_size == 3
        else (NET_STATIC, NET_ACTUATED)
    )

    for path, tls_type in ((targets[0], "static"), (targets[1], "actuated")):
        if args.osm:
            build_osm(args.osm, path, tls_type)
        else:
            build_grid(path, tls_type, grid_number=args.grid_size)
        print(f"Successfully generated network: {path}")

    net = sumolib.net.readNet(str(targets[0]))
    print(f"Network built. Configured traffic light controllers: {[tl.getID() for tl in net.getTrafficLights()]}")


if __name__ == "__main__":
    main()
