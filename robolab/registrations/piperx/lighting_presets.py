# SPDX-License-Identifier: Apache-2.0

"""Lighting presets for the PiPER-X registrations.

The Piper-X default look is the shared inspection-stage rig: HDR background as the only light (selected with
``run_rollout.py --background``, default ``home_office``) plus a visible ground plane. That rig is robot-agnostic and
lives in ``robolab.registrations.stage_lighting.StageLightingCfg``; ``PiperXRenderLightingCfg`` is kept as an alias
so existing imports and the shared cfg bundle keep working.
"""

from robolab.registrations.stage_lighting import StageLightingCfg

PiperXRenderLightingCfg = StageLightingCfg

__all__ = ["PiperXRenderLightingCfg", "StageLightingCfg"]
