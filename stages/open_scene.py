"""Startup script for isaac-sim.sh --exec: opens the USD given in $ROBOLAB_SCENE, then frames the viewport."""
import asyncio, os, omni.usd, omni.kit.app
SCENE = os.environ["ROBOLAB_SCENE"]
async def _open():
    app = omni.kit.app.get_app()
    for _ in range(60):
        await app.next_update_async()
    ok = omni.usd.get_context().open_stage(SCENE)
    print(f"[open_scene] open_stage({SCENE}) -> {ok}")
    for _ in range(120):                      # let materials/meshes load before framing
        await app.next_update_async()
    try:
        from omni.kit.viewport.utility import get_active_viewport, frame_viewport_selection
        omni.usd.get_context().get_selection().clear_selected_prim_paths()
        frame_viewport_selection(get_active_viewport())   # no selection -> frame all
        print("[open_scene] framed viewport")
    except Exception as e:  # noqa: BLE001
        print(f"[open_scene] frame failed: {e}")
asyncio.ensure_future(_open())
