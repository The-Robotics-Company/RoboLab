# SPDX-License-Identifier: Apache-2.0
"""Generate the shared DR episode list for food_packing.

Both the pi05 eval and the demo generation read THIS file, so train and test share a
distribution by construction rather than by two samplers agreeing.

Only the XY of the two manipulated objects varies. Draws are rejected unless the moved
object clears EVERY other object on the table: the nominal gaps are tight (mustard ->
container is 9.4 cm, spam -> sugar_box 12.1 cm), so a +/-4 cm draw can overlap a
neighbour. Two interpenetrating 128-hull colliders do not resolve gently -- PhysX
depenetrates them explosively and the scene flings objects across the table (seen as
one env in four blowing up). Collision-free sampling is half the fix; the other half is
capping max_depenetration_velocity on the bodies, in gen_demos.py.

    python make_dr_episodes.py --n 200 --seed 0
"""
import argparse, json, os
import numpy as np

NOMINAL = {"mustard_bottle": (0.41397, 0.14843), "spam_can": (0.65275, 0.01087)}

# Fixed objects: centre XY and footprint radius, from measured vertices.
STATIC = {
    "cheezit_box":    ((0.4958, -0.1002), 0.096),
    "soup_can":       ((0.4492,  0.0405), 0.052),
    "sugar_box":      ((0.5899,  0.1136), 0.057),
    "container":      ((0.3271,  0.1124), 0.076),
    "small_box":      ((0.4459,  0.0392), 0.075),
    "grey_bin_right": ((0.6509, -0.4084), 0.190),
    "grey_bin_left":  ((0.7336,  0.3772), 0.205),
}
RADIUS = {"mustard_bottle": 0.062, "spam_can": 0.059}

# The delivered layout is close-packed and demonstrably stable (everything rests without
# interpenetrating), so the criterion is NOT an absolute clearance -- that rejects the
# nominal scene itself. It is "never get closer to a neighbour than the delivery already
# is": each moved object must keep at least its nominal centre distance, less a small
# tolerance. Guaranteed satisfiable, since the zero-offset draw passes by definition.
TOL = 0.005      # metres a draw may close in on a neighbour vs nominal


def _nominal_gap(name, other):
    return float(np.linalg.norm(np.array(NOMINAL[name]) - np.array(STATIC[other][0])))


GAPS = {n: {o: _nominal_gap(n, o) for o in STATIC} for n in NOMINAL}


def _clear(name, xy):
    for other, (oxy, _) in STATIC.items():
        if np.linalg.norm(np.array(xy) - np.array(oxy)) < GAPS[name][other] - TOL:
            return False
    return True


def sample(rng, rad):
    pair0 = float(np.linalg.norm(np.array(NOMINAL["mustard_bottle"]) - np.array(NOMINAL["spam_can"])))
    for _ in range(20000):
        o = {k: [float(rng.uniform(-rad, rad)), float(rng.uniform(-rad, rad))] for k in NOMINAL}
        p1 = np.array(NOMINAL["mustard_bottle"]) + o["mustard_bottle"]
        p2 = np.array(NOMINAL["spam_can"]) + o["spam_can"]
        if np.linalg.norm(p1 - p2) < min(pair0, RADIUS["mustard_bottle"] + RADIUS["spam_can"]) - TOL:
            continue
        if _clear("mustard_bottle", p1) and _clear("spam_can", p2):
            return o
    raise RuntimeError("no collision-free draw found; reduce --radius")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--radius", type=float, default=0.04)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dr_episodes.json"))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    eps = [sample(rng, a.radius) for _ in range(a.n)]
    json.dump({"seed": a.seed, "radius_m": a.radius, "tolerance_m": TOL,
               "nominal_xy": {k: list(v) for k, v in NOMINAL.items()},
               "collision_checked_against": sorted(STATIC), "episodes": eps},
              open(a.out, "w"), indent=2)
    arr = np.array([[e["mustard_bottle"][0], e["mustard_bottle"][1],
                     e["spam_can"][0], e["spam_can"][1]] for e in eps])
    print(f"wrote {a.n} collision-free episodes -> {a.out}")
    print(f"  radius +/-{a.radius} m, tolerance {TOL} m, checked against {len(STATIC)} objects")
    print(f"  realised |offset| mean {np.abs(arr).mean()*100:.1f} cm, max {np.abs(arr).max()*100:.1f} cm")


main()
