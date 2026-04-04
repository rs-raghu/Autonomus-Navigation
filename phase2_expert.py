"""
phase2_expert.py  —  Phase 2: Pure Pursuit Expert Driver + Dataset Collection

Drives the tractor autonomously and logs every frame to CSV for imitation
learning.  Three behaviours are deliberately mixed into the dataset:

  Normal laps      — Pure Pursuit on the centre-line, constant speed.
  Noise events     — Random lateral displacement injected at a random time
                     between 1 and 5 seconds; produces sharp-recovery examples.
  Offset laps      — Every 5th lap the tractor drives the full path with a
                     constant random lateral bias (8–16 px left or right of
                     centre, staying well within the dirt strip).  No noise
                     is injected during offset laps.  This teaches the model
                     to maintain a consistent non-zero CTE when required.

Run:   python phase2_expert.py

Controls
--------
  SPACE     Pause / resume collection
  ESC       Save immediately and quit

CSV schema (identical to main.py for pd.concat compatibility)
-------------------------------------------------------------
  t_s                 wall-clock seconds since session start
  lap                 lap counter (auto-increments on each arrival)
  x, y                tractor screen position (px)
  heading_deg         tractor heading (degrees)
  speed_px_s          tractor speed (px / s)
  cte_px              signed CTE from centre-line  (+right, −left)
  lateral_offset_px   target offset this lap (0 on normal / noise laps)
  heading_error_deg   heading vs path tangent (degrees)
  lookahead_angle_deg α: heading-to-lookahead angle (degrees)
  steer_input         Pure Pursuit output  [-1, 1]  ← imitation target
  throttle_input      throttle used  [-1, 1]
  noise_injected      1 on displacement frames, 0 otherwise
"""

import sys
import math
import time
import csv
import os
import random

import pygame

from src.tractor      import Tractor
from src.path         import Path
from src.renderer     import Renderer
from src.pure_pursuit import PurePursuit

# ── Simulation config ─────────────────────────────────────────────────────────
WINDOW_W, WINDOW_H = 900, 580
FPS                = 60
WINDOW_TITLE       = "Autonomous Tractor — Phase 2  (Expert Data Collection)"

HOME_X, HOME_Y = 640.0, 530.0
HOME_HEADING   = -math.pi / 2

LOOKAHEAD_PX     = 80.0    # Pure Pursuit lookahead distance (px)
THROTTLE_NORMAL  = 0.78    # throttle during normal driving
THROTTLE_RECOV   = 0.45    # reduced throttle when far off target line
CTE_RECOV_THRESH = 35.0    # |cte| above which recovery throttle activates

TARGET_ROWS      = 50_000  # stop and save once this many rows are collected

# Noise injection — fired at a RANDOM time between 1 s and 5 s (60–300 frames)
NOISE_MIN_FRAMES = 60      # 1 s  at 60 fps
NOISE_MAX_FRAMES = 300     # 5 s  at 60 fps
NOISE_MIN_PX     = 20.0    # minimum lateral displacement
NOISE_MAX_PX     = 60.0    # maximum lateral displacement
NOISE_HEADING_DEG = 5.0    # ±heading jitter on injection
NOISE_FLASH_DUR  = 14      # frames to display yellow flash

# Offset laps — every N_LAPS_PER_OFFSET-th lap drives at a fixed bias
N_LAPS_PER_OFFSET = 5      # every 5th lap is an offset lap
OFFSET_MIN_PX    = 8.0     # minimum absolute offset from centre-line
OFFSET_MAX_PX    = 16.0    # maximum absolute offset  (DIRT_HALF_W=22, so safe)

# ── Visual palette ────────────────────────────────────────────────────────────
C_LOOKAHEAD  = ( 80, 220, 255)   # cyan lookahead circle
C_OFFSET_LAP = (255, 140,  40)   # orange tint for offset laps
C_WHITE      = (220, 220, 220)
C_OK         = ( 72, 220,  88)
C_WARN       = (255, 162,  40)
C_DANGER     = (255,  52,  52)
C_GOLD       = (255, 215,   0)
C_PANEL_BG   = (  0,   0,   0, 152)
C_PANEL_BD   = (110, 110, 110,  90)


# ── Data logger ───────────────────────────────────────────────────────────────

class ExpertLogger:
    """One row per frame — dense enough for a 50 k-row IL corpus."""

    # Column order matches main.py's DataLogger for pd.concat compatibility.
    FIELDS = [
        "t_s", "lap", "x", "y", "heading_deg", "speed_px_s",
        "cte_px", "lateral_offset_px",
        "heading_error_deg", "lookahead_angle_deg",
        "steer_input", "throttle_input", "noise_injected",
    ]

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._t0 = time.time()

    def log(self,
            tractor:           Tractor,
            cte:               float,
            lateral_offset_px: float,
            heading_err_deg:   float,
            alpha_deg:         float,
            steer_input:       float,
            throttle_input:    float,
            lap:               int,
            noise_injected:    bool) -> None:
        self.rows.append({
            "t_s":                round(time.time() - self._t0, 3),
            "lap":                lap,
            "x":                  round(tractor.x, 2),
            "y":                  round(tractor.y, 2),
            "heading_deg":        round(math.degrees(tractor.heading), 2),
            "speed_px_s":         round(tractor.speed, 2),
            "cte_px":             round(cte, 3),
            "lateral_offset_px":  round(lateral_offset_px, 2),
            "heading_error_deg":  round(heading_err_deg, 3),
            "lookahead_angle_deg": round(alpha_deg, 3),
            "steer_input":        round(steer_input, 4),
            "throttle_input":     round(throttle_input, 3),
            "noise_injected":     int(noise_injected),
        })

    def save(self) -> str:
        if not self.rows:
            print("[ExpertLogger] No data to save.")
            return ""
        os.makedirs("reports", exist_ok=True)
        n = 1
        while os.path.exists(f"reports/expert_{n}.csv"):
            n += 1
        fp = f"reports/expert_{n}.csv"
        with open(fp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        noise_n  = sum(r["noise_injected"] for r in self.rows)
        offset_n = sum(1 for r in self.rows if r["lateral_offset_px"] != 0)
        print(f"[ExpertLogger] {len(self.rows):,} rows  →  {fp}")
        print(f"               normal: {len(self.rows)-noise_n-offset_n:,}  "
              f"noise-recovery: {noise_n:,}  offset: {offset_n:,}")
        return fp

    @property
    def count(self) -> int:
        return len(self.rows)


# ── Noise injection ───────────────────────────────────────────────────────────

def inject_noise(tractor: Tractor, path: Path) -> None:
    """Displace the tractor laterally 20–60 px off the path centre-line."""
    wps = path.waypoints
    # Project onto nearest segment to find the local perpendicular
    best_i, best_d = 0, float('inf')
    for i in range(len(wps) - 1):
        ax, ay = wps[i];  bx, by = wps[i + 1]
        dx, dy = bx - ax, by - ay
        sq = dx*dx + dy*dy
        if sq < 1e-6:
            continue
        t = max(0.0, min(1.0, ((tractor.x-ax)*dx + (tractor.y-ay)*dy) / sq))
        d = math.hypot(tractor.x-(ax+t*dx), tractor.y-(ay+t*dy))
        if d < best_d:
            best_d, best_i = d, i

    ax, ay = wps[best_i];  bx, by = wps[best_i + 1]
    seg = math.hypot(bx-ax, by-ay)
    if seg < 1e-6:
        return

    nx = -(by-ay) / seg    # perpendicular unit vector
    ny =  (bx-ax) / seg

    side = random.choice([-1.0, 1.0])
    mag  = random.uniform(NOISE_MIN_PX, NOISE_MAX_PX)
    tractor.x       += nx * mag * side
    tractor.y       += ny * mag * side
    tractor.heading += math.radians(random.uniform(-NOISE_HEADING_DEG,
                                                    NOISE_HEADING_DEG))


# ── Offset selection ──────────────────────────────────────────────────────────

def pick_lap_offset(lap: int) -> float:
    """
    Return the lateral offset (px) for this lap.
    Every N_LAPS_PER_OFFSET-th lap → random signed offset in [8, 16] px.
    All other laps → 0 (centre-line tracking).
    """
    if lap % N_LAPS_PER_OFFSET == 0:
        mag  = random.uniform(OFFSET_MIN_PX, OFFSET_MAX_PX)
        side = random.choice([-1.0, 1.0])
        return mag * side
    return 0.0


# ── Overlay drawing ───────────────────────────────────────────────────────────

def _panel(surf: pygame.Surface,
           lines: list[tuple[str, tuple]],
           pos: tuple[int, int],
           width: int,
           font: pygame.font.Font) -> None:
    h = 10 + len(lines) * 18
    p = pygame.Surface((width, h), pygame.SRCALPHA)
    p.fill(C_PANEL_BG)
    pygame.draw.rect(p, C_PANEL_BD, p.get_rect(), 1, border_radius=6)
    for i, (txt, col) in enumerate(lines):
        p.blit(font.render(txt, True, col), (6, 5 + i * 18))
    surf.blit(p, pos)


def draw_overlay(surf: pygame.Surface,
                 result,                  # PPResult
                 tractor: Tractor,
                 cte: float,
                 heading_err_deg: float,
                 lap: int,
                 lap_offset: float,
                 rows: int,
                 paused: bool,
                 noise_flash: int,
                 save_path: str,
                 font: pygame.font.Font) -> None:
    W, H = surf.get_size()

    # Lookahead line + circle
    lx, ly = int(result.lookahead_pt[0]), int(result.lookahead_pt[1])
    lp_col = C_OFFSET_LAP if lap_offset != 0.0 else C_LOOKAHEAD
    pygame.draw.line(surf, lp_col, (int(tractor.x), int(tractor.y)), (lx, ly), 1)
    pygame.draw.circle(surf, lp_col, (lx, ly), 8, 2)

    # Noise flash
    if noise_flash > 0:
        a = int(170 * noise_flash / NOISE_FLASH_DUR)
        t = pygame.Surface((W, H), pygame.SRCALPHA)
        t.fill((255, 230, 40, a))
        surf.blit(t, (0, 0))

    # Offset lap tint (subtle orange vignette)
    if lap_offset != 0.0 and noise_flash == 0:
        v = pygame.Surface((W, H), pygame.SRCALPHA)
        v.fill((255, 140, 40, 18))
        surf.blit(v, (0, 0))

    # Right-side status panel
    pct      = min(rows / TARGET_ROWS, 1.0)
    m_col    = C_WARN if paused else C_OK
    abs_cte  = abs(cte)
    cte_col  = C_OK if abs_cte < 10 else (C_WARN if abs_cte < 30 else C_DANGER)

    if lap_offset != 0.0:
        lap_lbl = f"OFFSET {lap_offset:+.0f} px"
        lap_col = C_OFFSET_LAP
    else:
        lap_lbl = "NORMAL"
        lap_col = C_OK

    lines = [
        (f"  Lap          {lap:>5}",             C_WHITE),
        (f"  Type  {lap_lbl:<14}",               lap_col),
        (f"  Rows    {rows:>9,}",                C_WHITE),
        (f"  Target  {TARGET_ROWS:>9,}",         C_WHITE),
        (f"  Progress {pct*100:>7.1f} %",        m_col),
        (f"  CTE       {cte:>+8.1f} px",         cte_col),
        (f"  Hdg err  {heading_err_deg:>+7.1f}°", C_WHITE),
        (f"  α        {math.degrees(result.alpha_rad):>+7.1f}°", C_WHITE),
        (f"  {'PAUSED' if paused else 'COLLECTING'}",  m_col),
    ]
    _panel(surf, lines, (W - 220, 10), 215, font)

    # Progress bar
    bx_, by_ = 10, H - 38
    bw_, bh_ = W - 20, 8
    pygame.draw.rect(surf, (35, 35, 35), (bx_, by_, bw_, bh_), border_radius=4)
    if pct > 0:
        pygame.draw.rect(surf, C_OK, (bx_, by_, int(bw_*pct), bh_), border_radius=4)
    hint = font.render(
        "SPACE pause  |  ESC save & quit", True, (130, 130, 130))
    surf.blit(hint, (bx_, by_ + 11))

    # Save banner
    if save_path:
        ov = pygame.Surface((W, 70), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 200))
        surf.blit(ov, (0, H//2 - 35))
        msg = font.render(
            f"DATASET SAVED  →  {save_path}    ESC to quit", True, C_GOLD)
        surf.blit(msg, (W//2 - msg.get_width()//2, H//2 - 10))


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tractor() -> Tractor:
    return Tractor(x=HOME_X, y=HOME_Y, heading=HOME_HEADING)


def next_noise_frame(current: int) -> int:
    """Schedule next noise injection at a RANDOM time 1–5 s from now."""
    return current + random.randint(NOISE_MIN_FRAMES, NOISE_MAX_FRAMES)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("monospace", 13)

    path     = Path()
    tractor  = make_tractor()
    renderer = Renderer(screen)
    pp       = PurePursuit(lookahead=LOOKAHEAD_PX)
    logger   = ExpertLogger()

    lap        : int   = 1
    lap_offset : float = pick_lap_offset(lap)   # 0.0 for lap 1
    frame_no   : int   = 0
    next_noise : int   = next_noise_frame(0)
    noise_flash: int   = 0
    paused     : bool  = False
    done       : bool  = False
    save_path  : str   = ""

    print(f"[Phase 2] Target {TARGET_ROWS:,} rows — SPACE pause — ESC save+quit")
    print(f"[Phase 2] Lap 1  offset={lap_offset:+.1f} px  "
          f"first_noise @ frame {next_noise}")

    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if not save_path:
                    save_path = logger.save()
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if not save_path:
                        save_path = logger.save()
                    done = True
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"[Phase 2] {'Paused' if paused else 'Resumed'} "
                          f"— {logger.count:,} rows")

        # ── Done state: hold save banner ──────────────────────────────────────
        if done:
            cte    = path.cross_track_error(tractor.x, tractor.y)
            result = pp.compute(tractor.x, tractor.y, tractor.heading, path)
            renderer.draw_frame(path, tractor, cte)
            draw_overlay(screen, result, tractor, cte, 0.0,
                         lap, lap_offset, logger.count,
                         False, 0, save_path, font)
            pygame.display.flip()
            continue

        # ── Paused: freeze physics ────────────────────────────────────────────
        if paused:
            cte    = path.cross_track_error(tractor.x, tractor.y)
            h_err  = pp.heading_error(tractor.x, tractor.y, tractor.heading, path)
            result = pp.compute(tractor.x, tractor.y, tractor.heading, path,
                                lateral_offset=lap_offset)
            renderer.draw_frame(path, tractor, cte)
            draw_overlay(screen, result, tractor, cte,
                         math.degrees(h_err),
                         lap, lap_offset, logger.count,
                         True, noise_flash, save_path, font)
            pygame.display.flip()
            continue

        frame_no += 1

        # ── Noise injection (skip entirely on offset laps) ────────────────────
        noise_this_frame = False
        if lap_offset == 0.0 and frame_no >= next_noise and logger.count > 30:
            inject_noise(tractor, path)
            noise_this_frame = True
            noise_flash      = NOISE_FLASH_DUR
            next_noise       = next_noise_frame(frame_no)
            print(f"[Phase 2]  ↯ noise  lap={lap}  row={logger.count:,}  "
                  f"next_noise in {next_noise - frame_no} frames")

        if noise_flash > 0:
            noise_flash -= 1

        # ── Pure Pursuit (with offset for offset laps) ────────────────────────
        result = pp.compute(tractor.x, tractor.y, tractor.heading, path,
                            lateral_offset=lap_offset)

        # Adaptive throttle: back off when far from target line
        cte_raw  = path.cross_track_error(tractor.x, tractor.y)
        cte_from_target = cte_raw - lap_offset          # distance from intended line
        throttle = (THROTTLE_NORMAL
                    if abs(cte_from_target) < CTE_RECOV_THRESH
                    else THROTTLE_RECOV)

        # ── Physics ───────────────────────────────────────────────────────────
        tractor.update(result.steer_input, throttle, dt)

        # ── Post-step metrics ─────────────────────────────────────────────────
        cte       = path.cross_track_error(tractor.x, tractor.y)
        h_err_rad = pp.heading_error(tractor.x, tractor.y, tractor.heading, path)

        # ── Log ───────────────────────────────────────────────────────────────
        logger.log(
            tractor           = tractor,
            cte               = cte,
            lateral_offset_px = lap_offset,
            heading_err_deg   = math.degrees(h_err_rad),
            alpha_deg         = math.degrees(result.alpha_rad),
            steer_input       = result.steer_input,
            throttle_input    = throttle,
            lap               = lap,
            noise_injected    = noise_this_frame,
        )

        # ── Arrival → auto-restart or finish ──────────────────────────────────
        if path.reached_destination(tractor.x, tractor.y):
            print(f"[Phase 2]  ✓ lap {lap} done  "
                  f"(offset={lap_offset:+.1f})  rows={logger.count:,}")
            if logger.count >= TARGET_ROWS:
                save_path = logger.save()
                done      = True
            else:
                lap       += 1
                lap_offset = pick_lap_offset(lap)
                tractor    = make_tractor()
                frame_no   = 0
                next_noise = next_noise_frame(0)
                print(f"[Phase 2]  → lap {lap}  offset={lap_offset:+.1f} px")

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.draw_frame(path, tractor, cte)
        draw_overlay(screen, result, tractor, cte,
                     math.degrees(h_err_rad),
                     lap, lap_offset, logger.count,
                     False, noise_flash, save_path, font)
        pygame.display.flip()


if __name__ == "__main__":
    main()