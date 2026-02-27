from __future__ import annotations

from typing import List, Tuple
import numpy as np

from ant_sim.sim.world import AntColonyWorld


def downsample_mean(mat: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Simple mean downsample (no external deps)."""
    in_h, in_w = mat.shape
    if out_h >= in_h and out_w >= in_w:
        return mat.copy()

    y_scale = in_h / out_h
    x_scale = in_w / out_w
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for oy in range(out_h):
        y0 = int(oy * y_scale)
        y1 = int((oy + 1) * y_scale)
        y1 = max(y1, y0 + 1)
        y1 = min(y1, in_h)
        for ox in range(out_w):
            x0 = int(ox * x_scale)
            x1 = int((ox + 1) * x_scale)
            x1 = max(x1, x0 + 1)
            x1 = min(x1, in_w)
            block = mat[y0:y1, x0:x1]
            out[oy, ox] = float(block.mean()) if block.size > 0 else 0.0
    return out


def to_uint8_heatmap(mat: np.ndarray, clip: float) -> np.ndarray:
    m = np.clip(mat, 0.0, clip) / max(clip, 1e-6)
    u8 = (m * 255.0).astype(np.uint8)
    return u8


def pack_world_state_json(world: AntColonyWorld, heatmap_size: Tuple[int, int] = (64, 48)) -> dict:
    out_w, out_h = heatmap_size

    # combine colonies pheromone as max for visualization
    pher_max = world.pheromone.max(axis=0)
    pher_ds = downsample_mean(pher_max, out_h=out_h, out_w=out_w)
    pher_u8 = to_uint8_heatmap(pher_ds, clip=world.cfg.pheromone_clip)
    pher_list: List[int] = pher_u8.reshape(-1).tolist()

    nests = [{"x": c.nest[0], "y": c.nest[1], "colony": c.colony_id} for c in world.colonies]
    ants = [{"x": a.x, "y": a.y, "colony": a.colony_id, "carrying": int(a.carrying)} for a in world.ants]
    foods = [{"x": x, "y": y, "amount": int(amt)} for (x, y), amt in world.foods.items()]
    delivered = [c.delivered for c in world.colonies]

    return {
        "type": "state",
        "tick": int(world.tick),
        "grid_w": int(world.cfg.grid_w),
        "grid_h": int(world.cfg.grid_h),
        "nests": nests,
        "ants": ants,
        "foods": foods,
        "delivered": delivered,
        "pheromone": {"w": int(out_w), "h": int(out_h), "data": pher_list},
    }