"""
phase5_vision.py  —  Phase 5: Vision Cone & Safety Stop

Runs the trained SAC policy (models/sac_best.zip) with:
  • A triangular vision cone projected ahead of the tractor
  • Randomly spawned obstacle sprites (logs and rocks) on the path
  • Automatic hard stop when any obstacle enters the cone
  • R key to clear the obstacle and resume (pre-Flutter manual override)
  • All stop events logged to reports/phase5_stops.csv

State machine
-------------
  RUNNING  → SAC drives, cone is green, obstacles spawn on a timer
  STOPPED  → obstacle in cone, tractor frozen, red banner + red cone
  CLEARED  → R pressed, obstacle removed, brief delay then RUNNING

Run:
    python phase5_vision.py                     # uses models/sac_best.zip
    python phase5_vision.py --model models/sac_final.zip
    python phase5_vision.py --spawn-interval 4  # obstacle every 4 s (default 6)
    python phase5_vision.py --no-rl             # Pure Pursuit only (useful debug)

Controls
--------
  R         Clear obstacle + resume (manual override hook)
  SPACE     Pause / resume
  O         Spawn an obstacle immediately
  ESC       Quit and save log
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import pygame

from src.tractor   import Tractor
from src.path      import Path
from src.renderer  import Renderer
from src.vision    import VisionCone
from src.obstacles import ObstacleManager, SPAWN_MARGIN
from src.gym_env   import TractorEnv, HOME_X, HOME_Y, HOME_HEADING, THROTTLE_RL, LOOKAHEAD_PX
from src.pure_pursuit import PurePursuit

WINDOW_W, WINDOW_H = 900, 580
FPS                = 60

# ── Config ────────────────────────────────────────────────────────────────────
CONE_HALF_ANGLE_DEG = 30.0    # vision cone half-width
CONE_DEPTH_PX       = 120.0   # vision cone reach
SPAWN_INTERVAL_S    = 3.0     # seconds between auto-spawned obstacles
                               # (lap ~9.6 s, so expect 2–3 spawns per lap)
RESUME_DELAY_FRAMES = 45      # brief pause after clearing before SAC resumes

# Obstacle spawning — positional guards
# Math: at t=3s tractor is at wp≈49/165 (30%). With SPAWN_AHEAD_WPS=12 that
# gives min_frac≈0.37, max_frac=0.80 → valid 70-waypoint (~327px) window.
SPAWN_MAX_PROGRESS  = 0.68    # stop spawning once tractor passes this fraction
SPAWN_AHEAD_WPS     = 12      # minimum waypoints ahead of tractor to spawn
                               # (≈48 px; cone depth is 120 px, so tractor sees it)

# ── Colours ───────────────────────────────────────────────────────────────────
C_WHITE  = (220, 220, 220)
C_GOLD   = (255, 215,   0)
C_OK     = ( 72, 220,  88)
C_WARN   = (255, 162,  40)
C_DANGER = (255,  52,  52)
C_PANEL  = (  0,   0,   0, 155)


# ── Stop event logger ─────────────────────────────────────────────────────────

class StopLogger:
    """Records every hard-stop event for Phase 6 fine-tuning integration."""

    FIELDS = [
        "event_no", "t_s",
        "x", "y", "heading_deg",
        "obstacle_kind", "obstacle_x", "obstacle_y",
        "resume_command",          # always "CLEAR_R" for Phase 5 (manual)
        "stopped_duration_s",      # seconds the tractor was frozen
    ]

    def __init__(self, report_dir: str = "reports"):
        self._dir      = report_dir
        self._events:  list[dict] = []
        self._t0       = time.time()
        self._event_no = 0
        self._stop_t   = None          # wall-clock time of last stop
        self._pending  = None          # partial event waiting for resume

    def on_stop(self, tractor: Tractor, obstacle) -> None:
        self._stop_t      = time.time()
        self._event_no   += 1
        self._pending     = {
            "event_no":       self._event_no,
            "t_s":            round(self._stop_t - self._t0, 3),
            "x":              round(tractor.x, 2),
            "y":              round(tractor.y, 2),
            "heading_deg":    round(math.degrees(tractor.heading), 2),
            "obstacle_kind":  obstacle.kind,
            "obstacle_x":     round(obstacle.x, 2),
            "obstacle_y":     round(obstacle.y, 2),
            "resume_command": "CLEAR_R",
            "stopped_duration_s": 0.0,
        }

    def on_resume(self) -> None:
        if self._pending and self._stop_t:
            self._pending["stopped_duration_s"] = round(
                time.time() - self._stop_t, 3)
            self._events.append(self._pending)
            self._pending = None
            self._stop_t  = None

    def save(self) -> str:
        if not self._events:
            return ""
        os.makedirs(self._dir, exist_ok=True)
        n = 1
        while os.path.exists(f"{self._dir}/phase5_stops_{n}.csv"):
            n += 1
        fp = f"{self._dir}/phase5_stops_{n}.csv"
        with open(fp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self._events)
        print(f"[StopLogger] {len(self._events)} stop event(s) → {fp}")
        return fp


# ── HUD overlay ───────────────────────────────────────────────────────────────

def _panel(surf, lines, pos, width, font):
    h     = len(lines) * 18 + 10
    panel = pygame.Surface((width, h), pygame.SRCALPHA)
    panel.fill(C_PANEL)
    pygame.draw.rect(panel, (110, 110, 110, 90),
                     panel.get_rect(), 1, border_radius=6)
    for i, (text, col) in enumerate(lines):
        panel.blit(font.render(text, True, col), (4, 5 + i * 18))
    surf.blit(panel, pos)


def draw_phase5_hud(surf, font, tractor, cte, state,
                    n_obstacles, n_stops, lap):
    W, H = surf.get_size()
    abs_cte   = abs(cte)
    cte_col   = C_OK if abs_cte < 10 else (C_WARN if abs_cte < 25 else C_DANGER)
    state_col = C_DANGER if state == "STOPPED" else (C_WARN if state == "CLEARED" else C_OK)

    lines = [
        (f"  Lap        {lap:>6}",            C_WHITE),
        (f"  State   {state:<10}",             state_col),
        (f"  CTE     {cte:>+8.1f} px",         cte_col),
        (f"  Speed   {tractor.speed:>7.1f} px/s", C_WHITE),
        (f"  Obstacles  {n_obstacles:>4}",     C_WHITE),
        (f"  Stops       {n_stops:>3}",        C_WARN if n_stops else C_WHITE),
    ]
    _panel(surf, lines, (W - 210, 10), 205, font)

    hint = font.render(
        "R clear  |  O spawn  |  SPACE pause  |  ESC quit",
        True, (110, 110, 110))
    surf.blit(hint, (10, H - 18))


# ── Main loop ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="models/sac_best.zip")
    p.add_argument("--spawn-interval", type=float, default=SPAWN_INTERVAL_S)
    p.add_argument("--no-rl",          action="store_true",
                   help="Use Pure Pursuit instead of SAC (debug mode)")
    return p.parse_args()


def make_tractor():
    return Tractor(x=HOME_X, y=HOME_Y, heading=HOME_HEADING)


def main():
    args = parse_args()

    # ── Load SAC model ────────────────────────────────────────────────────────
    policy = None
    if not args.no_rl:
        model_path = args.model
        if not os.path.exists(model_path):
            for candidate in ["models/sac_best.zip",
                              "models/sac_final.zip",
                              "models/sac_latest.zip"]:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            else:
                print("[Phase 5] No SAC model found — falling back to Pure Pursuit.")
                args.no_rl = True

        if not args.no_rl:
            from stable_baselines3 import SAC
            policy = SAC.load(model_path)
            print(f"[Phase 5] Loaded SAC model: {model_path}")

    # ── Pygame init ───────────────────────────────────────────────────────────
    pygame.init()
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Autonomous Tractor — Phase 5 (Vision & Safety)")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("monospace", 13)

    # ── Simulation objects ────────────────────────────────────────────────────
    path     = Path()
    cone     = VisionCone(half_angle_deg=CONE_HALF_ANGLE_DEG,
                          depth_px=CONE_DEPTH_PX)
    obs_mgr  = ObstacleManager()
    renderer = Renderer(screen)
    pp       = PurePursuit(lookahead=LOOKAHEAD_PX)
    stop_log = StopLogger()

    # Gym env for observations (headless)
    env      = TractorEnv(render_mode=None)
    obs, _   = env.reset(seed=0)
    tractor  = env._tractor

    # ── State ─────────────────────────────────────────────────────────────────
    state          = "RUNNING"    # RUNNING | STOPPED | CLEARED
    resume_counter = 0            # countdown frames after CLEARED
    current_stop_obs = None       # the obstacle that triggered the stop

    lap            = 1
    n_stops        = 0
    paused         = False
    # Start the timer half an interval in the past so the first obstacle
    # appears at t ≈ SPAWN_INTERVAL_S/2 rather than waiting a full interval.
    last_spawn_t   = time.time() - args.spawn_interval * 0.5

    print("[Phase 5] Running — R clear obstacle, O spawn, SPACE pause, ESC quit\n")

    while True:
        dt = clock.tick(FPS) / 1000.0

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop_log.save()
                env.close()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    stop_log.save()
                    env.close()
                    pygame.quit()
                    sys.exit()

                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"[Phase 5] {'Paused' if paused else 'Resumed'}")

                elif event.key == pygame.K_o:
                    _n   = len(path.waypoints)
                    _idx = path.nearest_waypoint_index(tractor.x, tractor.y)
                    _mf  = (_idx + SPAWN_AHEAD_WPS) / max(_n - 1, 1)
                    _xf  = 1.0 - SPAWN_MARGIN
                    if _mf < _xf:
                        new_obs = obs_mgr.spawn_random(path,
                                                       min_frac=_mf,
                                                       max_frac=_xf)
                        print(f"[Phase 5] Manual spawn: {new_obs.kind} "
                              f"at ({new_obs.x:.0f}, {new_obs.y:.0f})")
                    else:
                        print("[Phase 5] Spawn skipped — tractor too close to destination")

                elif event.key == pygame.K_r and state == "STOPPED":
                    # ── Manual override: clear obstacle and resume ─────────────
                    if current_stop_obs:
                        obs_mgr.remove(current_stop_obs)
                        current_stop_obs = None
                    stop_log.on_resume()
                    state          = "CLEARED"
                    resume_counter = RESUME_DELAY_FRAMES
                    print(f"[Phase 5] Override: obstacle cleared — resuming in "
                          f"{RESUME_DELAY_FRAMES} frames")

        if paused:
            # Keep rendering but freeze physics
            cte      = path.cross_track_error(tractor.x, tractor.y)
            geom     = cone.geometry(tractor)
            detected = cone.detect(tractor, obs_mgr.obstacles)
            renderer.draw_frame(path, tractor, cte,
                                cone_geom=geom,
                                obstacle_mgr=obs_mgr,
                                detected=detected,
                                stopped=(state == "STOPPED"))
            draw_phase5_hud(screen, font, tractor, cte, "PAUSED",
                            len(obs_mgr.obstacles), n_stops, lap)
            pygame.display.flip()
            continue

        # ── Auto-spawn timer (progress-aware) ────────────────────────────────
        n_wps        = len(path.waypoints)
        wp_idx_now   = path.nearest_waypoint_index(tractor.x, tractor.y)
        progress_now = wp_idx_now / max(n_wps - 1, 1)

        if (state == "RUNNING" and
                time.time() - last_spawn_t > args.spawn_interval and
                progress_now < SPAWN_MAX_PROGRESS):

            # Always spawn ahead of the tractor.
            # min_frac: at least SPAWN_AHEAD_WPS waypoints forward.
            # max_frac: no closer than SPAWN_MARGIN (20%) from the destination.
            min_frac = (wp_idx_now + SPAWN_AHEAD_WPS) / max(n_wps - 1, 1)
            max_frac = 1.0 - SPAWN_MARGIN

            if min_frac < max_frac:
                new_obs = obs_mgr.spawn_random(path,
                                               min_frac=min_frac,
                                               max_frac=max_frac)
                last_spawn_t = time.time()
                print(f"[Phase 5] Auto-spawn: {new_obs.kind} "
                      f"at ({new_obs.x:.0f}, {new_obs.y:.0f})  "
                      f"[progress {progress_now*100:.0f}%]")

        # ── CLEARED countdown ─────────────────────────────────────────────────
        if state == "CLEARED":
            resume_counter -= 1
            if resume_counter <= 0:
                state = "RUNNING"
                print("[Phase 5] Resumed RUNNING")

        # ── Physics (frozen when STOPPED) ─────────────────────────────────────
        if state in ("RUNNING", "CLEARED"):
            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
            else:
                # Pure Pursuit fallback
                result = pp.compute(tractor.x, tractor.y,
                                    tractor.heading, path)
                tractor.update(result.steer_input, THROTTLE_RL, 1/FPS)
                terminated = path.reached_destination(tractor.x, tractor.y)
                truncated  = False
                obs        = np.zeros(6, dtype=np.float32)  # unused

            tractor = env._tractor if policy else tractor

            if terminated or truncated:
                print(f"[Phase 5] Lap {lap} complete "
                      f"({'arrived' if (policy and info.get('arrived')) else 'off-path/timeout'})")
                lap  += 1
                obs, _ = env.reset()
                tractor = env._tractor
                obs_mgr.clear()
                # Half-interval offset: first spawn of the new lap fires at
                # t ≈ SPAWN_INTERVAL_S/2 rather than after a full wait.
                last_spawn_t = time.time() - args.spawn_interval * 0.5
                state = "RUNNING"

        # ── Vision cone detection ─────────────────────────────────────────────
        cte      = path.cross_track_error(tractor.x, tractor.y)
        geom     = cone.geometry(tractor)
        detected = cone.detect(tractor, obs_mgr.obstacles)

        if state == "RUNNING" and detected:
            # First detection: hard stop
            state            = "STOPPED"
            current_stop_obs = detected[0]
            n_stops         += 1
            stop_log.on_stop(tractor, current_stop_obs)
            print(f"[Phase 5] ⚠ HARD STOP  obstacle={current_stop_obs.kind}  "
                  f"pos=({current_stop_obs.x:.0f},{current_stop_obs.y:.0f})  "
                  f"tractor=({tractor.x:.0f},{tractor.y:.0f})")

        # ── Render ────────────────────────────────────────────────────────────
        renderer.draw_frame(path, tractor, cte,
                            cone_geom=geom,
                            obstacle_mgr=obs_mgr,
                            detected=detected,
                            stopped=(state == "STOPPED"))
        draw_phase5_hud(screen, font, tractor, cte, state,
                        len(obs_mgr.obstacles), n_stops, lap)
        pygame.display.flip()


if __name__ == "__main__":
    main()