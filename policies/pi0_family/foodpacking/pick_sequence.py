# SPDX-License-Identifier: Apache-2.0
"""Waypoints for the vectorised food_packing expert -- hot-reloadable.

Split out of gen_demos.py so `--dev` can reload it on save without rebooting
Isaac (~2 min a time). Same pattern as trc-rollout's pick_sequence.py and
pointbody's scripts/dev.py.

Everything here is iteration surface: waypoint order, offsets, step budgets.
Anything structural -- scene, robot, cameras, colliders -- still needs a restart.

``phases(ctx)`` returns [(label, target_fn, grip, budget)] where target_fn maps
(grasp_xyz, hand_quat, over_xyz) -> the TCP goal for that waypoint.
"""
import numpy as np

# Budgets are CAPS; gen_demos exits a waypoint early once every env has stopped
# moving. Sized from measured single-env convergence, with generous headroom.
ALIGN_BACKOFF = 0.15      # m along -approach for the pre-grasp standoff
LIFT_HEIGHT = 0.22        # m straight up after the grasp closes
RETRACT_HEIGHT = 0.12     # m straight up after releasing
DWELLS = {"close", "release"}   # run their full budget; the 4-bar gripper is slow


def phases():
    return [
        # label,       target,                                          grip,   budget
        ("align",      lambda g, q, o, ap: g - ap * ALIGN_BACKOFF,      "open",  150),
        ("descend",    lambda g, q, o, ap: g,                            "open",  140),
        ("close",      lambda g, q, o, ap: g,                            "close",  70),
        ("lift",       lambda g, q, o, ap: g + np.array([0, 0, LIFT_HEIGHT]), "close", 140),
        ("transport",  lambda g, q, o, ap: o,                            "close", 320),
        ("release",    lambda g, q, o, ap: o,                            "open",   20),
        ("retract",    lambda g, q, o, ap: o + np.array([0, 0, RETRACT_HEIGHT]), "open", 40),
    ]
