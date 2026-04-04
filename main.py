"""
main.py  —  Phase 1: Manual Driving + Data Collection
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

CSV schema
----------
The DataLogger now computes Pure Pursuit features at every log point so that
manually-driven CSVs can be concatenated with phase2_expert.py CSVs for Phase 3
training.  Both files share the same core column set.

Core columns (Phase 3 training features):
  t_s                 wall-clock seconds since run start
  lap                 always 1 for manual runs
  x, y                tractor position (px)
  heading_deg         tractor heading (degrees)
  speed_px_s          tractor speed (px / s)
  cte_px              signed CTE from centre-line
  lateral_offset_px   always 0 for manual runs
  heading_error_deg   heading vs path tangent (degrees)
  lookahead_angle_deg Pure Pursuit α at current pose (degrees)
  steer_input         your keyboard steer input  [-1, 1]  ← label for Phase 3
  throttle_input      your keyboard throttle input
  noise_injected      always 0 for manual runs

Extra columns (manual-drive diagnostics, not used in Phase 3):
  dt_s                seconds since previous logged position
  dist_moved_px       distance moved since last log point (px)
  steer_angle_deg     actual front-wheel angle (degrees)
  waypoint_idx        index of nearest waypoint
"""

import sys
import math
import time
import csv
import os
import pygame

from src.tractor      import Tractor
from src.path         import Path
from src.renderer     import Renderer
from src.pure_pursuit import PurePursuit

WINDOW_W, WINDOW_H = 900, 580
FPS                = 60
WINDOW_TITLE       = "Autonomous Tractor — Phase 1"

HOME_X, HOME_Y = 640.0, 530.0
HOME_HEADING   = -math.pi / 2   # pointing north (up)

LOOKAHEAD_PX   = 80.0           # must match phase2_expert.py for comparable α


# ── Data logger ───────────────────────────────────────────────────────────────

LOG_MIN_DIST = 3.0   # px — log a new row only when tractor moved this far

class DataLogger:
    """
    Position-change-based logger.  A row is written only when the tractor has
    moved ≥ LOG_MIN_DIST px since the last logged position.

    Computes heading_error_deg and lookahead_angle_deg via Pure Pursuit so the
    CSV is compatible with phase2_expert.py for pd.concat in Phase 3.
    """

    # Columns produced — identical set to ExpertLogger.FIELDS in phase2_expert.py
    # (extra manual-drive columns appended at the end).
    CORE_FIELDS = [
        "t_s", "lap", "x", "y", "heading_deg", "speed_px_s",
        "cte_px", "lateral_offset_px",
        "heading_error_deg", "lookahead_angle_deg",
        "steer_input", "throttle_input", "noise_injected",
    ]
    EXTRA_FIELDS = ["dt_s", "dist_moved_px", "steer_angle_deg", "waypoint_idx"]

    def __init__(self, path: Path, pp: PurePursuit):
        self._path        = path
        self._pp          = pp
        self.records:     list[dict] = []
        self.start_time:  float      = time.time()
        self._last_x:     float      = float('nan')
        self._last_y:     float      = float('nan')
        self._last_t:     float      = 0.0

    def maybe_log(self,
                  tractor:        Tractor,
                  cte:            float,
                  steer_input:    float,
                  throttle_input: float,
                  wp_idx:         int) -> bool:
        """
        Log a row only if the tractor has moved ≥ LOG_MIN_DIST px.
        Returns True when a row was written.
        """
        now  = time.time() - self.start_time
        dist = math.hypot(tractor.x - self._last_x,
                          tractor.y - self._last_y)

        if math.isnan(dist) or dist >= LOG_MIN_DIST:
            # Pure Pursuit features — computed so the CSV matches expert schema
            pp_result     = self._pp.compute(
                tractor.x, tractor.y, tractor.heading, self._path)
            h_err_rad     = self._pp.heading_error(
                tractor.x, tractor.y, tractor.heading, self._path)

            dt_since_last = round(now - self._last_t, 4)

            self.records.append({
                # ── Core Phase-3 columns ──────────────────────────────────────
                "t_s":                  round(now, 3),
                "lap":                  1,           # manual runs are single-lap
                "x":                    round(tractor.x, 2),
                "y":                    round(tractor.y, 2),
                "heading_deg":          round(math.degrees(tractor.heading), 2),
                "speed_px_s":           round(tractor.speed, 2),
                "cte_px":               round(cte, 3),
                "lateral_offset_px":    0.0,         # no offset in manual drives
                "heading_error_deg":    round(math.degrees(h_err_rad), 3),
                "lookahead_angle_deg":  round(math.degrees(pp_result.alpha_rad), 3),
                "steer_input":          round(steer_input, 3),
                "throttle_input":       round(throttle_input, 3),
                "noise_injected":       0,           # never in manual drives
                # ── Extra manual-drive diagnostics ────────────────────────────
                "dt_s":                 dt_since_last,
                "dist_moved_px":        round(dist if not math.isnan(dist) else 0.0, 2),
                "steer_angle_deg":      round(math.degrees(tractor.steer_angle), 2),
                "waypoint_idx":         wp_idx,
            })
            self._last_x = tractor.x
            self._last_y = tractor.y
            self._last_t = now
            return True
        return False

    def save(self) -> str:
        """Write CSV to reports/trial_N.csv; return the file path."""
        if not self.records:
            return ""
        os.makedirs("reports", exist_ok=True)
        n = 1
        while os.path.exists(f"reports/trial_{n}.csv"):
            n += 1
        filepath = f"reports/trial_{n}.csv"
        all_fields = self.CORE_FIELDS + self.EXTRA_FIELDS
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(self.records)
        print(f"[DataLogger] {len(self.records)} rows → {filepath}")
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
    pp       = PurePursuit(lookahead=LOOKAHEAD_PX)   # shared with DataLogger
    tractor  = make_tractor()
    renderer = Renderer(screen)
    logger   = DataLogger(path, pp)

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

        # ── Logging ───────────────────────────────────────────────────────────
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