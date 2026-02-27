from __future__ import annotations

import sys
import os
import pygame

# Allow running from repo root: python backend/demos/demo_pygame.py
THIS_DIR = os.path.dirname(__file__)
BACKEND_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from ant_sim.sim.entities import WorldConfig
from ant_sim.sim.world import AntColonyWorld


def main():
    cfg = WorldConfig(grid_w=80, grid_h=60, num_colonies=2, ants_per_colony=120)
    world = AntColonyWorld(cfg, seed=1)

    CELL = 12
    W, H = cfg.grid_w * CELL, cfg.grid_h * CELL
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Ant Colony Demo (Rule-based, Multi-colony)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    paused = False
    speed = 1

    running = True
    while running:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_1:
                    speed = 1
                elif event.key == pygame.K_2:
                    speed = 3
                elif event.key == pygame.K_3:
                    speed = 8
                elif event.key == pygame.K_r:
                    world.reset(seed=1)

        if not paused:
            for _ in range(speed):
                world.step_all_heuristic()

        screen.fill((10, 10, 14))

        # pheromone heatmap (max over colonies)
        pher = world.pheromone.max(axis=0)
        k = 2
        for y in range(0, cfg.grid_h, k):
            for x in range(0, cfg.grid_w, k):
                if world.obstacles[y, x] == 1:
                    continue
                v = pher[y, x]
                if v > 0.05:
                    intensity = int(min(160, v * 10.0))
                    col = (intensity, intensity, 0)
                    pygame.draw.rect(screen, col, (x * CELL, y * CELL, CELL * k, CELL * k))

        # obstacles
        for y in range(cfg.grid_h):
            for x in range(cfg.grid_w):
                if world.obstacles[y, x] == 1:
                    pygame.draw.rect(screen, (40, 40, 48), (x * CELL, y * CELL, CELL, CELL))

        # foods
        for (x, y), amt in world.foods.items():
            r = 2 + int(min(6, amt / 60))
            pygame.draw.circle(screen, (60, 200, 90), (x * CELL + CELL // 2, y * CELL + CELL // 2), r)

        # nests
        for c in world.colonies:
            col = (90, 160, 255) if c.colony_id == 0 else (255, 140, 120)
            nx, ny = c.nest
            pygame.draw.rect(screen, col, (nx * CELL, ny * CELL, CELL, CELL))

        # ants
        for ant in world.ants:
            if ant.carrying:
                col = (255, 255, 255)
            else:
                col = (120, 200, 255) if ant.colony_id == 0 else (255, 160, 140)
            pygame.draw.circle(screen, col, (ant.x * CELL + CELL // 2, ant.y * CELL + CELL // 2), 2)

        txt = f"tick={world.tick}  speed={speed}x  pause=[SPACE]  speed=[1/2/3]  reset=[R]  delivered={[c.delivered for c in world.colonies]}  foods={len(world.foods)}"
        screen.blit(font.render(txt, True, (220, 220, 220)), (8, 8))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()