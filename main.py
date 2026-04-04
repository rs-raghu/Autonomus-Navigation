"""
main.py  —  Phase 1: Simulation Environment
Autonomous Agricultural Tractor Simulation

Run:  python main.py

Controls
--------
  W / ↑     Throttle forward
  S / ↓     Reverse / brake
  A / ←     Steer left
  D / →     Steer right
  ESC       Quit
"""

import sys
import pygame

from src.tractor  import Tractor
from src.path     import Path
from src.renderer import Renderer

WINDOW_W, WINDOW_H = 900, 580
FPS                = 60
WINDOW_TITLE       = "Autonomous Tractor — Phase 1: Simulation Environment"


def main() -> None:
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock   = pygame.time.Clock()

    path     = Path()
    tractor  = Tractor(x=80.0, y=220.0, heading=0.0)   # start at HOME, heading east
    renderer = Renderer(screen)

    while True:
        # ── Delta-time (capped to avoid physics explosion on lag spikes) ─────
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        # ── Input ─────────────────────────────────────────────────────────────
        keys     = pygame.key.get_pressed()
        throttle = float(
            (keys[pygame.K_UP]   or keys[pygame.K_w]) -
            (keys[pygame.K_DOWN] or keys[pygame.K_s]) * 0.6
        )
        steer = float(
            (keys[pygame.K_RIGHT] or keys[pygame.K_d]) -
            (keys[pygame.K_LEFT]  or keys[pygame.K_a])
        )

        # ── Physics update ────────────────────────────────────────────────────
        tractor.update(steer, throttle, dt)

        # ── Cross-track error ─────────────────────────────────────────────────
        cte = path.cross_track_error(tractor.x, tractor.y)

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.draw_frame(path, tractor, cte)
        pygame.display.flip()


if __name__ == "__main__":
    main()
