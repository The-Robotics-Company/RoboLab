# SPDX-License-Identifier: Apache-2.0

"""Lighting presets for the PiPER-X registrations.

The Piper-X default look is the shared rig USD ``stages/rigs/home_office.usda`` (HDR dome as the only light + visible
ground), spawned once at /World by ``robolab.registrations.rig.rig_cfg``. ``PiperXRenderLightingCfg`` is kept as an
alias of that default rig cfg so existing imports and the shared cfg bundle keep working.
"""

from robolab.registrations.rig import DEFAULT_RIG, rig_cfg

PiperXRenderLightingCfg = rig_cfg(DEFAULT_RIG)

__all__ = ["PiperXRenderLightingCfg"]
