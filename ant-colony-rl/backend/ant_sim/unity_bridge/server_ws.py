from __future__ import annotations

import argparse
import asyncio
import json
from typing import Set

import websockets

from ant_sim.sim.entities import WorldConfig
from ant_sim.sim.world import AntColonyWorld
from ant_sim.unity_bridge.serializers import pack_world_state_json


async def handle_client(ws, clients: Set):
    clients.add(ws)
    try:
        await ws.send(json.dumps({"type": "hello", "msg": "connected"}))
        async for msg in ws:
            # Minimal command handling (optional for Unity)
            # Example commands:
            # {"type":"reset","seed":123}
            # {"type":"pause","value":true}
            try:
                data = json.loads(msg)
            except Exception:
                continue
            ws._last_cmd = data  # store last cmd, read by loop
    finally:
        clients.discard(ws)


async def main_async(host: str, port: int, hz: float):
    cfg = WorldConfig()
    world = AntColonyWorld(cfg, seed=42)

    clients: Set = set()
    server = await websockets.serve(lambda ws: handle_client(ws, clients), host, port)
    print(f"WebSocket server listening on ws://{host}:{port}")

    paused = False
    step_dt = 1.0 / max(hz, 1e-6)

    try:
        while True:
            # read last commands
            for ws in list(clients):
                cmd = getattr(ws, "_last_cmd", None)
                if cmd and isinstance(cmd, dict):
                    t = cmd.get("type")
                    if t == "reset":
                        seed = cmd.get("seed", None)
                        world.reset(seed=seed)
                    elif t == "pause":
                        paused = bool(cmd.get("value", True))
                    ws._last_cmd = None

            if not paused:
                world.step_all_heuristic()

            state = pack_world_state_json(world, heatmap_size=(64, 48))
            payload = json.dumps(state)

            # broadcast
            dead = []
            for ws in clients:
                try:
                    await ws.send(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                clients.discard(ws)

            await asyncio.sleep(step_dt)
    finally:
        server.close()
        await server.wait_closed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--hz", type=float, default=20.0)
    args = parser.parse_args()
    asyncio.run(main_async(args.host, args.port, args.hz))


if __name__ == "__main__":
    main()