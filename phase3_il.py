"""
phase3_il.py  —  Phase 3: Imitation Learning Policy — Live Simulation

Loads model_il.pt (produced by train_il.py) and runs the trained MLP as the
tractor's steering policy inside the Pygame simulation.

The Pure Pursuit expert is rendered as a ghost (thin cyan line to its
lookahead point) so you can visually compare the IL policy against the oracle.

Run:
    python phase3_il.py

Controls
--------
  SPACE     Pause / resume
  G         Toggle ghost (Pure Pursuit oracle overlay)
  ESC       Quit and print session summary

Live HUD (right panel)
-----------------------
  CTE           signed cross-track error (px)
  heading err   heading vs path tangent  (degrees)
  speed         tractor speed (px / s)
  α             lookahead angle fed to model (degrees)
  steer pred    model's steering output [-1, 1]
  lap CTE avg   running mean |CTE| for the current lap

Performance is logged to reports/il_session.csv on quit.
"""

import sys
import math
import time
import csv
import os

import torch
import torch.nn as nn
import pygame

from src.tractor      import Tractor
from src.path         import Path
from src.renderer     import Renderer
from src.pure_pursuit import PurePursuit

WINDOW_W, WINDOW_H = 900, 580
FPS                = 60
WINDOW_TITLE       = "Autonomous Tractor — Phase 3 (Imitation Learning)"

HOME_X, HOME_Y  = 640.0, 530.0
HOME_HEADING    = -math.pi / 2

LOOKAHEAD_PX    = 80.0        # must match train_il.py / phase2_expert.py
THROTTLE_IL     = 0.78        # fixed throttle — same as expert
MODEL_PATH      = "model_il.pt"

FEATURE_COLS    = ["cte_px", "heading_error_deg", "speed_px_s", "lookahead_angle_deg"]

# ── Colours ────────────────────────────────────────────────────────────────────
C_WHITE        = (220, 220, 220)
C_PANEL_BG     = (  0,   0,   0, 155)
C_PANEL_BD     = (110, 110, 110,  90)
C_OK           = ( 72, 220,  88)
C_WARN         = (255, 162,  40)
C_DANGER       = (255,  52,  52)
C_GHOST        = ( 80, 220, 255)    # Pure Pursuit oracle line
C_IL           = (200, 130, 255)    # IL lookahead line (purple)
C_GOLD         = (255, 215,   0)


# ── MLP definition (must match train_il.py exactly) ───────────────────────────

class SteeringMLP(nn.Module):
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, 32),     nn.Tanh(),
            nn.Linear(32, 1),      nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Policy wrapper ─────────────────────────────────────────────────────────────

class ILPolicy:
    """
    Wraps the trained SteeringMLP checkpoint for single-step inference.
    Loads normalisation stats from the checkpoint so features are scaled
    identically to training.
    """

    def __init__(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found: '{checkpoint_path}'\n"
                f"Run  python train_il.py  first."
            )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.model = SteeringMLP(in_dim=len(FEATURE_COLS))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self._mean = torch.tensor(ckpt["norm_stats"]["mean"], dtype=torch.float32)
        self._std  = torch.tensor(ckpt["norm_stats"]["std"],  dtype=torch.float32)

        best_rmse = ckpt.get("best_val_rmse", float("nan"))
        print(f"[ILPolicy] Loaded '{checkpoint_path}'  "
              f"(val RMSE = {best_rmse:.4f})")

    @torch.no_grad()
    def predict(self,
                cte_px:              float,
                heading_error_deg:   float,
                speed_px_s:          float,
                lookahead_angle_deg: float) -> float:
        """Return normalised steer_input in [-1, 1]."""
        raw = torch.tensor(
            [cte_px, heading_error_deg, speed_px_s, lookahead_angle_deg],
            dtype=torch.float32)
        x = (raw - self._mean) / self._std
        return float(self.model(x.unsqueeze(0)))


# ── Session logger ─────────────────────────────────────────────────────────────

class SessionLogger:
    """Logs one row per frame for post-session analysis."""

    FIELDS = [
        "t_s", "lap", "x", "y", "heading_deg", "speed_px_s",
        "cte_px", "heading_error_deg", "lookahead_angle_deg",
        "steer_il", "steer_pp",
    ]

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._t0 = time.time()

    def log(self,
            tractor:          Tractor,
            cte:              float,
            heading_err_deg:  float,
            alpha_deg:        float,
            steer_il:         float,
            steer_pp:         float,
            lap:              int) -> None:
        self.rows.append({
            "t_s":               round(time.time() - self._t0, 3),
            "lap":               lap,
            "x":                 round(tractor.x, 2),
            "y":                 round(tractor.y, 2),
            "heading_deg":       round(math.degrees(tractor.heading), 2),
            "speed_px_s":        round(tractor.speed, 2),
            "cte_px":            round(cte, 3),
            "heading_error_deg": round(heading_err_deg, 3),
            "lookahead_angle_deg": round(alpha_deg, 3),
            "steer_il":          round(steer_il, 4),
            "steer_pp":          round(steer_pp, 4),
        })

    def save(self) -> str:
        if not self.rows:
            return ""
        os.makedirs("reports", exist_ok=True)
        n = 1
        while os.path.exists(f"reports/il_session_{n}.csv"):
            n += 1
        fp = f"reports/il_session_{n}.csv"
        with open(fp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"[SessionLogger] {len(self.rows):,} rows  →  {fp}")
        return fp


# ── HUD helpers ────────────────────────────────────────────────────────────────

def _cte_colour(abs_cte: float):
    return C_OK if abs_cte < 10 else (C_WARN if abs_cte < 25 else C_DANGER)


def draw_il_overlay(surf:            pygame.Surface,
                    font:            pygame.font.Font,
                    tractor:         Tractor,
                    cte:             float,
                    heading_err_deg: float,
                    alpha_deg:       float,
                    steer_il:        float,
                    steer_pp:        float,
                    lap:             int,
                    lap_cte_sum:     float,
                    lap_cte_count:   int,
                    show_ghost:      bool,
                    pp_result,                   # PPResult
                    il_lookahead_pt: tuple,
                    paused:          bool) -> None:

    W, H = surf.get_size()

    # ── IL lookahead line (purple) ────────────────────────────────────────────
    lx, ly = int(il_lookahead_pt[0]), int(il_lookahead_pt[1])
    pygame.draw.line(surf, C_IL,
                     (int(tractor.x), int(tractor.y)), (lx, ly), 1)
    pygame.draw.circle(surf, C_IL, (lx, ly), 8, 2)

    # ── Ghost: Pure Pursuit oracle line (cyan) ────────────────────────────────
    if show_ghost:
        gx, gy = int(pp_result.lookahead_pt[0]), int(pp_result.lookahead_pt[1])
        pygame.draw.line(surf, C_GHOST,
                         (int(tractor.x), int(tractor.y)), (gx, gy), 1)
        pygame.draw.circle(surf, C_GHOST, (gx, gy), 8, 2)

    # ── Right-side status panel ───────────────────────────────────────────────
    abs_cte      = abs(cte)
    avg_cte      = (lap_cte_sum / lap_cte_count) if lap_cte_count else 0.0
    steer_diff   = steer_il - steer_pp
    diff_col     = (C_OK if abs(steer_diff) < 0.05
                    else C_WARN if abs(steer_diff) < 0.15
                    else C_DANGER)

    lines = [
        (f"  Lap            {lap:>4}",                       C_WHITE),
        (f"  CTE       {cte:>+8.1f} px",                     _cte_colour(abs_cte)),
        (f"  Lap |CTE| {avg_cte:>7.2f} px",                  C_WHITE),
        (f"  Hdg err  {heading_err_deg:>+7.1f} °",           C_WHITE),
        (f"  α        {alpha_deg:>+7.1f} °",                 C_WHITE),
        (f"  Steer IL {steer_il:>+8.3f}",                    C_IL),
        (f"  Steer PP {steer_pp:>+8.3f}",                    C_GHOST),
        (f"  Δsteer   {steer_diff:>+8.3f}",                  diff_col),
        (f"  {'PAUSED' if paused else 'RUNNING':>15}",        C_WARN if paused else C_OK),
    ]

    h      = 10 + len(lines) * 18
    panel  = pygame.Surface((225, h), pygame.SRCALPHA)
    panel.fill(C_PANEL_BG)
    pygame.draw.rect(panel, C_PANEL_BD, panel.get_rect(), 1, border_radius=6)
    for i, (text, col) in enumerate(lines):
        panel.blit(font.render(text, True, col), (4, 5 + i * 18))
    surf.blit(panel, (W - 235, 10))

    # ── Legend (bottom left) ──────────────────────────────────────────────────
    legend_y = H - 52
    pygame.draw.line(surf, C_IL,    (10, legend_y),      (34, legend_y),      2)
    pygame.draw.circle(surf, C_IL,  (34, legend_y), 4,   2)
    surf.blit(font.render("IL policy", True, C_IL), (40, legend_y - 6))

    if show_ghost:
        pygame.draw.line(surf,   C_GHOST, (10, legend_y + 18), (34, legend_y + 18), 2)
        pygame.draw.circle(surf, C_GHOST, (34, legend_y + 18), 4, 2)
        surf.blit(font.render("PP oracle (G to hide)", True, C_GHOST),
                  (40, legend_y + 12))
    else:
        surf.blit(font.render("G — show PP oracle", True, (120, 120, 120)),
                  (10, legend_y + 12))

    # ── Controls hint ─────────────────────────────────────────────────────────
    hint = font.render("SPACE pause  |  G ghost  |  ESC quit", True, (110, 110, 110))
    surf.blit(hint, (10, H - 18))


def draw_arrival_banner(surf: pygame.Surface, font: pygame.font.Font,
                        lap: int, avg_cte: float) -> None:
    W, H = surf.get_size()
    overlay = pygame.Surface((W, 120), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 190))
    surf.blit(overlay, (0, H // 2 - 60))

    big_font = pygame.font.SysFont("monospace", 22, bold=True)
    title = big_font.render(f"Lap {lap} complete!", True, C_GOLD)
    surf.blit(title, (W // 2 - title.get_width() // 2, H // 2 - 48))
    sub = font.render(
        f"Mean |CTE| = {avg_cte:.2f} px   —   restarting …",
        True, C_WHITE)
    surf.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 + 4))


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_tractor() -> Tractor:
    return Tractor(x=HOME_X, y=HOME_Y, heading=HOME_HEADING)


def print_summary(logger: SessionLogger, total_laps: int) -> None:
    if not logger.rows:
        return
    ctes    = [abs(r["cte_px"]) for r in logger.rows]
    steers  = [r["steer_il"]    for r in logger.rows]
    diffs   = [abs(r["steer_il"] - r["steer_pp"]) for r in logger.rows]
    print("\n" + "=" * 52)
    print(f"  Phase 3 session summary  ({total_laps} lap(s) completed)")
    print("=" * 52)
    print(f"  Total frames logged : {len(logger.rows):>8,}")
    print(f"  Mean  |CTE|         : {sum(ctes)/len(ctes):>8.2f} px")
    print(f"  Max   |CTE|         : {max(ctes):>8.2f} px")
    print(f"  Mean  |Δsteer|      : {sum(diffs)/len(diffs):>8.4f}")
    print(f"  Max   |Δsteer|      : {max(diffs):>8.4f}")
    print("=" * 52 + "\n")


# ── Main loop ──────────────────────────────────────────────────────────────────

def main() -> None:
    policy   = ILPolicy(MODEL_PATH)

    pygame.init()
    pygame.display.set_caption(WINDOW_TITLE)
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("monospace", 13)

    path     = Path()
    pp       = PurePursuit(lookahead=LOOKAHEAD_PX)
    tractor  = make_tractor()
    renderer = Renderer(screen)
    logger   = SessionLogger()

    lap          : int   = 1
    lap_cte_sum  : float = 0.0
    lap_cte_count: int   = 0
    paused       : bool  = False
    show_ghost   : bool  = True
    total_laps   : int   = 0

    # Arrival flash
    arrival_flash_frames : int   = 0
    ARRIVAL_FLASH_DUR    : int   = 90   # 1.5 s at 60 fps

    print("[Phase 3] Running IL policy — SPACE pause, G ghost, ESC quit")

    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logger.save()
                print_summary(logger, total_laps)
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    logger.save()
                    print_summary(logger, total_laps)
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"[Phase 3] {'Paused' if paused else 'Resumed'}")
                elif event.key == pygame.K_g:
                    show_ghost = not show_ghost

        if paused:
            # Freeze physics but keep drawing
            cte          = path.cross_track_error(tractor.x, tractor.y)
            h_err_rad    = pp.heading_error(tractor.x, tractor.y,
                                            tractor.heading, path)
            pp_result    = pp.compute(tractor.x, tractor.y,
                                      tractor.heading, path)
            alpha_deg    = math.degrees(pp_result.alpha_rad)
            steer_il     = policy.predict(cte,
                                          math.degrees(h_err_rad),
                                          tractor.speed,
                                          alpha_deg)
            renderer.draw_frame(path, tractor, cte)
            draw_il_overlay(
                screen, font, tractor, cte, math.degrees(h_err_rad),
                alpha_deg, steer_il, pp_result.steer_input,
                lap, lap_cte_sum, lap_cte_count, show_ghost,
                pp_result, pp_result.lookahead_pt, paused=True)
            pygame.display.flip()
            continue

        # ── Pure Pursuit oracle (for ghost + feature extraction) ──────────────
        pp_result = pp.compute(tractor.x, tractor.y, tractor.heading, path)

        # ── CTE + heading error ────────────────────────────────────────────────
        cte       = path.cross_track_error(tractor.x, tractor.y)
        h_err_rad = pp.heading_error(tractor.x, tractor.y, tractor.heading, path)
        alpha_deg = math.degrees(pp_result.alpha_rad)

        # ── IL steering prediction ─────────────────────────────────────────────
        steer_il  = policy.predict(
            cte_px              = cte,
            heading_error_deg   = math.degrees(h_err_rad),
            speed_px_s          = tractor.speed,
            lookahead_angle_deg = alpha_deg,
        )

        # ── Physics (IL steers, fixed throttle) ───────────────────────────────
        tractor.update(steer_il, THROTTLE_IL, dt)

        # ── Post-step metrics ─────────────────────────────────────────────────
        cte            = path.cross_track_error(tractor.x, tractor.y)
        lap_cte_sum   += abs(cte)
        lap_cte_count += 1

        # ── Log ───────────────────────────────────────────────────────────────
        logger.log(
            tractor          = tractor,
            cte              = cte,
            heading_err_deg  = math.degrees(h_err_rad),
            alpha_deg        = alpha_deg,
            steer_il         = steer_il,
            steer_pp         = pp_result.steer_input,
            lap              = lap,
        )

        # ── Arrival ───────────────────────────────────────────────────────────
        if path.reached_destination(tractor.x, tractor.y):
            avg_cte = lap_cte_sum / max(lap_cte_count, 1)
            print(f"[Phase 3]  ✓ lap {lap}  mean|CTE|={avg_cte:.2f} px")
            total_laps     += 1
            arrival_flash_frames = ARRIVAL_FLASH_DUR
            tractor        = make_tractor()
            lap           += 1
            lap_cte_sum    = 0.0
            lap_cte_count  = 0

        # ── Draw ──────────────────────────────────────────────────────────────
        renderer.draw_frame(path, tractor, cte)

        # Arrival flash: show banner for a few seconds then auto-clear
        if arrival_flash_frames > 0:
            avg_cte_display = (lap_cte_sum / max(lap_cte_count, 1)
                               if lap_cte_count else 0.0)
            draw_arrival_banner(screen, font, lap - 1, avg_cte_display)
            arrival_flash_frames -= 1

        draw_il_overlay(
            screen, font, tractor, cte, math.degrees(h_err_rad),
            alpha_deg, steer_il, pp_result.steer_input,
            lap, lap_cte_sum, lap_cte_count, show_ghost,
            pp_result, pp_result.lookahead_pt, paused=False)

        pygame.display.flip()


if __name__ == "__main__":
    main()