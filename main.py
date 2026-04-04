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
  R         Restart (after arrival)
  ESC       Quit
"""

import sys
import math
import time
import csv
import os
import pygame

from src.tractor  import Tractor
from src.path     import Path
from src.renderer import Renderer

WINDOW_W, WINDOW_H = 900, 580
FPS                = 60
WINDOW_TITLE       = "Autonomous Tractor — Phase 1"

# ── Home position & initial heading ──────────────────────────────────────────
HOME_X, HOME_Y = 640.0, 530.0
HOME_HEADING   = -math.pi / 2   # −90° = pointing north (up in screen)


# ── Data logger ───────────────────────────────────────────────────────────────

LOG_MIN_DIST = 3.0   # px — minimum position change before a new row is written

class DataLogger:
    """
    Position-change-based logger.

    A new row is written only when the tractor has moved at least
    LOG_MIN_DIST pixels since the last logged position.  Each row
    also carries dt_s — the elapsed seconds since the previous entry —
    so time information is never lost, just compressed.

    Result: uniform spatial coverage with far fewer rows than frame-
    based logging, and no redundant stationary entries.
    """

    def __init__(self):
        self.records:    list[dict]        = []
        self.start_time: float             = time.time()
        self._last_x:    float             = float('nan')
        self._last_y:    float             = float('nan')
        self._last_t:    float             = 0.0          # time of last logged row

    def maybe_log(self, tractor: Tractor, cte: float,
                  steer_input: float, throttle_input: float,
                  wp_idx: int) -> bool:
        """
        Log a row only if the tractor has moved >= LOG_MIN_DIST px.
        Returns True when a row was actually written.
        """
        now  = time.time() - self.start_time
        dist = math.hypot(tractor.x - self._last_x,
                          tractor.y - self._last_y)

        # Always log the very first call (NaN guard) or when moved enough
        if math.isnan(dist) or dist >= LOG_MIN_DIST:
            dt_since_last = round(now - self._last_t, 4)
            self.records.append({
                "t_s":             round(now, 3),
                "dt_s":            dt_since_last,        # ← time since last row
                "x":               round(tractor.x, 2),
                "y":               round(tractor.y, 2),
                "dist_moved_px":   round(dist if not math.isnan(dist) else 0.0, 2),
                "heading_deg":     round(math.degrees(tractor.heading), 2),
                "speed_px_s":      round(tractor.speed, 2),
                "steer_angle_deg": round(math.degrees(tractor.steer_angle), 2),
                "steer_input":     round(steer_input, 3),
                "throttle_input":  round(throttle_input, 3),
                "cte_px":          round(cte, 2),
                "waypoint_idx":    wp_idx,
            })
            self._last_x = tractor.x
            self._last_y = tractor.y
            self._last_t = now
            return True
        return False

    def save(self) -> str:
        """Write CSV to reports/trial_N.csv and return the file path."""
        if not self.records:
            return ""
        os.makedirs("reports", exist_ok=True)

        n = 1
        while os.path.exists(f"reports/trial_{n}.csv"):
            n += 1
        filepath = f"reports/trial_{n}.csv"

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.records[0].keys())
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[DataLogger] {len(self.records)} position-change rows → {filepath}")
        return filepath

    def reset(self) -> None:
        self.records    = []
        self.start_time = time.time()
        self._last_x    = float('nan')
        self._last_y    = float('nan')
        self._last_t    = 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tractor() -> Tractor:
    return Tractor(x=HOME_X, y=HOME_Y, heading=HOME_HEADING)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock    = pygame.time.Clock()

    path     = Path()
    tractor  = make_tractor()
    renderer = Renderer(screen)
    logger   = DataLogger()

    arrived     = False
    report_path = ""

    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_r and arrived:
                    # Restart
                    tractor     = make_tractor()
                    logger.reset()
                    arrived     = False
                    report_path = ""

        # ── Input (frozen after arrival) ──────────────────────────────────────
        if not arrived:
            keys     = pygame.key.get_pressed()
            throttle = float(
                (keys[pygame.K_UP]   or keys[pygame.K_w]) -
                (keys[pygame.K_DOWN] or keys[pygame.K_s]) * 0.6
            )
            steer = float(
                (keys[pygame.K_RIGHT] or keys[pygame.K_d]) -
                (keys[pygame.K_LEFT]  or keys[pygame.K_a])
            )
        else:
            throttle = steer = 0.0

        # ── Physics update ────────────────────────────────────────────────────
        if not arrived:
            tractor.update(steer, throttle, dt)

        # ── Metrics ───────────────────────────────────────────────────────────
        cte    = path.cross_track_error(tractor.x, tractor.y)
        wp_idx = path.nearest_waypoint_index(tractor.x, tractor.y)

        # ── Logging (position-change-based) ──────────────────────────────────
        if not arrived:
            logger.maybe_log(tractor, cte, steer, throttle, wp_idx)

        # ── Arrival detection ─────────────────────────────────────────────────
        if not arrived and path.reached_destination(tractor.x, tractor.y):
            arrived     = True
            report_path = logger.save()

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.draw_frame(path, tractor, cte, arrived, report_path)
        pygame.display.flip()


if __name__ == "__main__":
    main()