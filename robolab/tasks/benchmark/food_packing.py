# SPDX-License-Identifier: Apache-2.0

"""Pack the mustard bottle and the spam can into the two grey bins.

Scene: ``food_packing_opus_gt.usda``, imported from a Robolab delivery-adapter drop
by ``scripts/import_robolab_delivery.py``. Objects keep their delivered ground-truth
poses; only a global z shift put the table's support surface at the canonical z = 0.

Left/right are unambiguous here: the exo camera sits at y = +0.57 looking toward -y,
so screen-left and robot-left agree. ``grey_bin_left`` is at y = +0.39,
``grey_bin_right`` at y = -0.41.

The instruction says "bin", never "box": the scene contains three actual boxes
(cheezit_box, sugar_box, small_box), so "box" would be ambiguous to a language-
conditioned policy. It is lowercase plain English to match DROID, because
``pi05_robolab_franka`` sets ``prompt_from_task=True`` -- this string IS the training
prompt, and the zero-shot eval must send the same one.
"""

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_groups_in_containers, pick_and_place
from robolab.core.task.task import Task


@configclass
class FoodPackingTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_groups_in_containers,
        params={
            "groups": [
                {"object": ["mustard_bottle"], "container": "grey_bin_left", "logical": "all",
                 "require_contact_with": False, "require_gripper_detached": True},
                {"object": ["spam_can"], "container": "grey_bin_right", "logical": "all",
                 "require_contact_with": False, "require_gripper_detached": True},
            ]
        },
    )


@dataclass
class FoodPackingTask(Task):
    contact_object_list = [
        "mustard_bottle",
        "spam_can",
        "cheezit_box",
        "soup_can",
        "sugar_box",
        "small_box",
        "container",
        "grey_bin_left",
        "grey_bin_right",
        "table",
    ]
    scene = import_scene("food_packing_opus_gt.usda", contact_object_list)
    terminations = FoodPackingTerminations
    instruction = {
        "default": "put the mustard bottle in the left bin and the spam can in the right bin",
        "vague": "pack the groceries into the bins",
        "specific": (
            "pick up the yellow French's mustard bottle and place it inside the grey bin on the left, "
            "then pick up the blue Spam can and place it inside the grey bin on the right"
        ),
    }
    episode_length_s: int = 120
    attributes = ["spatial", "sorting"]
    subtasks = [
        pick_and_place(object=["mustard_bottle"], container="grey_bin_left", logical="all", score=0.5),
        pick_and_place(object=["spam_can"], container="grey_bin_right", logical="all", score=0.5),
    ]
