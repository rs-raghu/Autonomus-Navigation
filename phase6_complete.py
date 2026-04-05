"""
phase6_complete.py  —  Phase 6: Full Autonomous Pipeline

Five-stage pipeline entirely within one Pygame window:

  DRAW      Click to place path waypoints.  Live Catmull-Rom spline preview.
  PREVIEW   Review smoothed path, curvature heatmap, validation warnings.
  COLLECT   Pure Pursuit expert drives N laps with noise injection.
            Logs state/action pairs to CSV (same schema as Phase 2).
  TRAIN     MLP trains on collected data in a background thread.
            Live loss curve redraws each epoch.
  DEMO      Trained IL model drives the custom path.
            Expert ghost (Pure Pursuit, cyan) shown alongside for comparison.

Controls
--------
  DRAW:     Left-click  add waypoint
            Backspace   undo last point
            S           save path to JSON
            L           load path from JSON
            Enter       confirm → PREVIEW  (need ≥ 3 points)

  PREVIEW:  Space       start COLLECT
            Backspace   back to DRAW

  COLLECT:  Esc         abort → back to PREVIEW
            (runs automatically until N_COLLECT_TARGET rows collected)

  TRAIN:    (automatic — cannot interrupt)

  DEMO:     R           restart lap
            Backspace   draw a new path
            Esc         quit
"""

import csv
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np
import pygame
import torch
import torch.nn as nn

from src.tractor      import Tractor
from src.pure_pursuit import PurePursuit
from src.renderer     import Renderer
from src.custom_path  import CustomPath, DIRT_HALF_W, catmull_rom_chain, resample_uniform

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_W, WINDOW_H   = 900, 580
FPS                  = 60

# Collection
THROTTLE_COLLECT     = 0.78
LOOKAHEAD_PX         = 80.0
N_COLLECT_TARGET     = 2_500   # rows before training starts
N_COLLECT_LAPS_MAX   = 20      # safety cap on laps
NOISE_INTERVAL       = 100     # frames between noise injections
NOISE_MAGNITUDE      = 22.0    # px lateral displacement
NOISE_HEADING_DEG    = 5.0

# Training
N_EPOCHS             = 70
BATCH_SIZE           = 256
LR                   = 1e-3
WEIGHT_DECAY         = 1e-4
PATIENCE             = 12
FEATURE_COLS         = ["cte_px", "heading_error_deg", "speed_px_s", "lookahead_angle_deg"]
LABEL_COL            = "steer_input"

# Files
COLLECT_CSV          = "reports/custom_expert.csv"
MODEL_OUT            = "models/custom_il.pt"
PATH_JSON            = "reports/custom_path.json"

# Colours
C_GRASS  = ( 55, 108,  42)
C_DIRT   = (128,  90,  48)
C_EDGE   = (100,  70,  36)
C_CTRL   = (100, 220, 100)
C_SPLINE = (210, 190,  80)
C_START  = ( 45, 210,  75)
C_END    = (210,  52,  52)
C_EXPERT = ( 80, 220, 255)   # cyan  — PP ghost
C_MODEL  = (255, 200,  60)   # amber — IL model
C_OK     = ( 72, 220,  88)
C_WARN   = (255, 162,  40)
C_DANGER = (255,  52,  52)
C_WHITE  = (220, 220, 220)
C_GOLD   = (255, 215,   0)
C_PANEL  = (  0,   0,   0, 155)
C_GRID   = ( 65,  90,  55)


# ── MLP (identical to Phase 3) ────────────────────────────────────────────────

class SteeringMLP(nn.Module):
    def __init__(self, in_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, 32),     nn.Tanh(),
            nn.Linear(32,  1),     nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def panel(surf, lines, pos, w, font, alpha=155):
    h  = len(lines) * 19 + 10
    s  = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill((0, 0, 0, alpha))
    pygame.draw.rect(s, (110, 110, 110, 90), s.get_rect(), 1, border_radius=6)
    for i, (text, col) in enumerate(lines):
        s.blit(font.render(text, True, col), (5, 5 + i*19))
    surf.blit(s, pos)


def draw_path_strip(surf, waypoints, curv_colors=None):
    """Draw dirt strip along waypoints, optionally colour-coded by curvature."""
    for i in range(len(waypoints)-1):
        col = curv_colors[i] if curv_colors else C_DIRT
        p1  = (int(waypoints[i][0]),   int(waypoints[i][1]))
        p2  = (int(waypoints[i+1][0]), int(waypoints[i+1][1]))
        pygame.draw.line(surf, col, p1, p2, DIRT_HALF_W * 2)
    for i, wp in enumerate(waypoints):
        col = curv_colors[i] if curv_colors else C_DIRT
        pygame.draw.circle(surf, col, (int(wp[0]), int(wp[1])), DIRT_HALF_W)


def draw_path_edges(surf, waypoints):
    for side in (-1, 1):
        for i in range(len(waypoints)-1):
            ax, ay = waypoints[i];  bx, by = waypoints[i+1]
            dx, dy = bx-ax, by-ay
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                continue
            nx = -dy/seg * side;  ny = dx/seg * side
            off = DIRT_HALF_W - 1
            pygame.draw.line(surf, C_EDGE,
                             (int(ax+nx*off), int(ay+ny*off)),
                             (int(bx+nx*off), int(by+ny*off)), 1)


def draw_centre_dashes(surf, waypoints):
    for i in range(0, len(waypoints)-1, 4):
        pygame.draw.line(surf, (162, 122, 70),
                         (int(waypoints[i][0]),   int(waypoints[i][1])),
                         (int(waypoints[i+1][0]), int(waypoints[i+1][1])), 1)


def draw_markers(surf, waypoints, font):
    sx, sy = int(waypoints[0][0]),  int(waypoints[0][1])
    ex, ey = int(waypoints[-1][0]), int(waypoints[-1][1])
    pygame.draw.circle(surf, C_START, (sx, sy), 14)
    pygame.draw.circle(surf, (255,255,255), (sx, sy), 14, 2)
    lbl = font.render("START", True, (0,0,0))
    surf.blit(lbl, (sx - lbl.get_width()//2, sy + 16))
    pygame.draw.circle(surf, C_END, (ex, ey), 14)
    pygame.draw.circle(surf, (255,255,255), (ex, ey), 14, 2)
    lbl = font.render("END", True, (0,0,0))
    surf.blit(lbl, (ex - lbl.get_width()//2, ey - 30))


def draw_loss_chart(surf, losses, rect, font):
    """Draw a live MSE loss curve inside rect = (x, y, w, h)."""
    x, y, w, h = rect
    pygame.draw.rect(surf, (20, 20, 20), rect, border_radius=6)
    pygame.draw.rect(surf, (80, 80, 80), rect, 1, border_radius=6)

    if len(losses) < 2:
        return

    max_l = max(losses) or 1.0
    min_l = min(losses)
    rng   = max(max_l - min_l, 1e-6)

    pts = []
    for i, lv in enumerate(losses):
        px_ = x + int(i / (len(losses)-1) * (w-2)) + 1
        py_ = y + h - 2 - int((lv - min_l) / rng * (h-4))
        pts.append((px_, py_))

    if len(pts) >= 2:
        pygame.draw.lines(surf, C_MODEL, False, pts, 2)

    # Latest value label
    lbl = font.render(f"loss {losses[-1]:.5f}", True, C_WHITE)
    surf.blit(lbl, (x + w - lbl.get_width() - 4, y + 4))


def draw_grid(surf):
    for gx in range(0, WINDOW_W, 50):
        pygame.draw.line(surf, C_GRID, (gx, 0), (gx, WINDOW_H), 1)
    for gy in range(0, WINDOW_H, 50):
        pygame.draw.line(surf, C_GRID, (0, gy), (WINDOW_W, gy), 1)


# ── Collection helpers ────────────────────────────────────────────────────────

def inject_noise(tractor: Tractor, path: CustomPath):
    wps = path.waypoints
    best_i, best_d = 0, float('inf')
    for i in range(len(wps)-1):
        ax, ay = wps[i];  bx, by = wps[i+1]
        dx, dy = bx-ax, by-ay
        sq = dx*dx + dy*dy
        if sq < 1e-6:
            continue
        t = max(0.0, min(1.0, ((tractor.x-ax)*dx+(tractor.y-ay)*dy)/sq))
        d = math.hypot(tractor.x-(ax+t*dx), tractor.y-(ay+t*dy))
        if d < best_d:
            best_d, best_i = d, i

    ax, ay = wps[best_i];  bx, by = wps[best_i+1]
    seg = math.hypot(bx-ax, by-ay)
    if seg < 1e-6:
        return

    import random
    nx = -(by-ay)/seg;  ny = (bx-ax)/seg
    side = random.choice([-1.0, 1.0])
    mag  = random.uniform(NOISE_MAGNITUDE*0.4, NOISE_MAGNITUDE)
    tractor.x       += nx * mag * side
    tractor.y       += ny * mag * side
    tractor.heading += math.radians(random.uniform(-NOISE_HEADING_DEG,
                                                    NOISE_HEADING_DEG))


def make_tractor(path: CustomPath) -> Tractor:
    sx, sy = path.waypoints[0]
    return Tractor(x=sx, y=sy, heading=path.start_heading())


# ── Training thread ───────────────────────────────────────────────────────────

def run_training(csv_path: str, model_out: str,
                 q: queue.Queue) -> None:
    """Runs in a daemon thread.  Puts ('epoch', ep, train_loss) and ('done',) into q."""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path, usecols=FEATURE_COLS + [LABEL_COL])
        df = df.dropna()

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.float32)

        # Normalise on full dataset (single-split is fine for demo)
        mean  = X.mean(axis=0)
        std   = np.maximum(X.std(axis=0), 1e-6)
        X_norm = (X - mean) / std

        X_t = torch.tensor(X_norm)
        y_t = torch.tensor(y)

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model   = SteeringMLP().to(device)
        opt     = torch.optim.Adam(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
        sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      opt, factor=0.5, patience=5)
        crit    = nn.MSELoss()

        best_loss   = float('inf')
        patience_ct = 0

        for ep in range(1, N_EPOCHS + 1):
            # Mini-batch loop
            idx  = torch.randperm(len(X_t))
            total = 0.0
            model.train()
            for start in range(0, len(X_t), BATCH_SIZE):
                batch_idx = idx[start:start+BATCH_SIZE]
                xb = X_t[batch_idx].to(device)
                yb = y_t[batch_idx].to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = crit(pred, yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(yb)

            train_loss = total / len(X_t)
            sched.step(train_loss)
            q.put(('epoch', ep, train_loss))

            if train_loss < best_loss - 1e-6:
                best_loss   = train_loss
                patience_ct = 0
                torch.save({'model_state': model.state_dict(),
                            'norm_stats':  {'mean': mean.tolist(),
                                            'std':  std.tolist()},
                            'feature_cols': FEATURE_COLS,
                            'label_col':    LABEL_COL,
                            'best_val_rmse': math.sqrt(best_loss)},
                           model_out)
            else:
                patience_ct += 1
                if patience_ct >= PATIENCE:
                    break

        q.put(('done',))

    except Exception as e:
        q.put(('error', str(e)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Autonomous Tractor — Phase 6")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("monospace", 13)
    font_big = pygame.font.SysFont("monospace", 20, bold=True)
    font_sm  = pygame.font.SysFont("monospace", 11)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models",  exist_ok=True)

    # ── Top-level state ───────────────────────────────────────────────────────
    stage          = "DRAW"     # DRAW | PREVIEW | COLLECT | TRAIN | DEMO
    control_pts:   list[tuple] = []
    custom_path:   CustomPath | None = None
    renderer:      Renderer | None   = None

    # COLLECT state
    collect_rows:  list[dict]  = []
    collect_lap    = 0
    collect_frame  = 0
    pp             = PurePursuit(lookahead=LOOKAHEAD_PX)
    tractor: Tractor | None = None

    # TRAIN state
    train_q:       queue.Queue = queue.Queue()
    train_losses:  list[float] = []
    train_epoch    = 0
    train_done     = False

    # DEMO state
    il_policy      = None       # loaded after training
    demo_tractor:  Tractor | None = None
    demo_lap       = 1
    demo_cte_sum   = 0.0
    demo_cte_n     = 0

    def reset_collect():
        nonlocal collect_rows, collect_lap, collect_frame, tractor
        collect_rows  = []
        collect_lap   = 0
        collect_frame = 0
        tractor       = make_tractor(custom_path)

    def start_training():
        nonlocal train_losses, train_epoch, train_done
        train_losses = []
        train_epoch  = 0
        train_done   = False
        # Save CSV
        if collect_rows:
            with open(COLLECT_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=collect_rows[0].keys())
                writer.writeheader()
                writer.writerows(collect_rows)
            print(f"[Phase 6] Saved {len(collect_rows)} rows → {COLLECT_CSV}")
        t = threading.Thread(target=run_training,
                             args=(COLLECT_CSV, MODEL_OUT, train_q),
                             daemon=True)
        t.start()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        # ═════════════════════════════════════════════════════════════════════
        # DRAW stage
        # ═════════════════════════════════════════════════════════════════════
        if stage == "DRAW":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_BACKSPACE and control_pts:
                        control_pts.pop()
                    elif event.key == pygame.K_RETURN and len(control_pts) >= 3:
                        custom_path = CustomPath(control_pts)
                        renderer    = Renderer(screen)
                        stage       = "PREVIEW"
                    elif event.key == pygame.K_s and len(control_pts) >= 2:
                        with open(PATH_JSON, "w") as f:
                            json.dump(control_pts, f)
                        print(f"[Phase 6] Path saved → {PATH_JSON}")
                    elif event.key == pygame.K_l:
                        if os.path.exists(PATH_JSON):
                            with open(PATH_JSON) as f:
                                control_pts = [tuple(p) for p in json.load(f)]
                            print(f"[Phase 6] Path loaded ({len(control_pts)} points)")
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    control_pts.append(event.pos)

            # Render DRAW
            screen.fill(C_GRASS)
            draw_grid(screen)

            # Live spline preview
            if len(control_pts) >= 2:
                preview_spline = catmull_rom_chain(control_pts, 15)
                preview_pts    = [(int(p[0]), int(p[1])) for p in preview_spline]
                if len(preview_pts) >= 2:
                    # Draw strip preview
                    for i in range(len(preview_pts)-1):
                        pygame.draw.line(screen, C_DIRT,
                                         preview_pts[i], preview_pts[i+1],
                                         DIRT_HALF_W * 2)
                    # Edges
                    for side in (-1, 1):
                        for i in range(len(preview_pts)-1):
                            ax, ay = preview_pts[i]
                            bx, by = preview_pts[i+1]
                            dx, dy = bx-ax, by-ay
                            seg = math.hypot(dx, dy)
                            if seg < 1e-6: continue
                            nx = -dy/seg*side*(DIRT_HALF_W-1)
                            ny =  dx/seg*side*(DIRT_HALF_W-1)
                            pygame.draw.line(screen, C_EDGE,
                                             (int(ax+nx), int(ay+ny)),
                                             (int(bx+nx), int(by+ny)), 1)

            # Control points
            for i, cp in enumerate(control_pts):
                pygame.draw.circle(screen, C_CTRL, (int(cp[0]), int(cp[1])), 7)
                pygame.draw.circle(screen, (255,255,255), (int(cp[0]),int(cp[1])), 7, 1)
                lbl = font_sm.render(str(i+1), True, (0,0,0))
                screen.blit(lbl, (int(cp[0])-4, int(cp[1])-6))
                if i > 0:
                    pygame.draw.line(screen, C_SPLINE,
                                     (int(control_pts[i-1][0]), int(control_pts[i-1][1])),
                                     (int(cp[0]), int(cp[1])), 1)

            # Mouse preview dot
            mx, my = pygame.mouse.get_pos()
            pygame.draw.circle(screen, (180, 255, 180), (mx, my), 5, 1)

            # Instructions panel
            n = len(control_pts)
            lines = [
                (f"  Points placed: {n}",                       C_OK if n>=3 else C_WARN),
                ("  Left-click    add waypoint",                 C_WHITE),
                ("  Backspace     undo last point",              C_WHITE),
                ("  S             save path",                    C_WHITE),
                ("  L             load saved path",              C_WHITE),
                ("  Enter         confirm (need ≥ 3 points)",   C_OK if n>=3 else C_WARN),
            ]
            panel(screen, lines, (10, 10), 290, font)

            if n < 3:
                msg = font_big.render("Draw your path  →  click to place points",
                                      True, C_GOLD)
                screen.blit(msg, (WINDOW_W//2 - msg.get_width()//2, WINDOW_H - 40))

            pygame.display.flip()
            continue

        # ═════════════════════════════════════════════════════════════════════
        # PREVIEW stage
        # ═════════════════════════════════════════════════════════════════════
        if stage == "PREVIEW":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_BACKSPACE:
                        stage = "DRAW"
                    elif event.key == pygame.K_SPACE:
                        reset_collect()
                        stage = "COLLECT"

            v   = custom_path.validation()
            cols = custom_path.curvature_colors()

            screen.fill(C_GRASS)
            draw_path_strip(screen, custom_path.waypoints, cols)
            draw_path_edges(screen, custom_path.waypoints)
            draw_centre_dashes(screen, custom_path.waypoints)
            draw_markers(screen, custom_path.waypoints, font)

            # Info panel
            warn_col = C_DANGER if not v.ok else C_OK
            lines = [
                (f"  Arc length   {v.arc_length:.0f} px",        C_WHITE),
                (f"  Waypoints    {len(custom_path.waypoints)}", C_WHITE),
                (f"  Min radius   {v.min_radius:.0f} px",        warn_col),
                (f"  Sharp turns  {v.sharp_turns}",              C_DANGER if v.sharp_turns else C_OK),
            ]
            for w_str in v.warnings:
                lines.append((f"  ⚠ {w_str}", C_DANGER))
            if v.ok:
                lines.append(("  Path looks good!", C_OK))
            panel(screen, lines, (10, 10), 310, font)

            # Legend
            legend = [
                ("  Green  = gentle curve",  C_OK),
                ("  Amber  = moderate curve", C_WARN),
                ("  Red    = tight curve",    C_DANGER),
            ]
            panel(screen, legend, (10, WINDOW_H-80), 210, font)

            hint = font.render(
                "Space → start collection   |   Backspace → re-draw",
                True, (140, 140, 140))
            screen.blit(hint, (WINDOW_W//2 - hint.get_width()//2, WINDOW_H-18))

            pygame.display.flip()
            continue

        # ═════════════════════════════════════════════════════════════════════
        # COLLECT stage — Pure Pursuit drives, logging data
        # ═════════════════════════════════════════════════════════════════════
        if stage == "COLLECT":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        stage = "PREVIEW"

            # ── Physics step ──────────────────────────────────────────────────
            cte      = custom_path.cross_track_error(tractor.x, tractor.y)
            pp_res   = pp.compute(tractor.x, tractor.y, tractor.heading,
                                  custom_path)
            h_err    = pp.heading_error(tractor.x, tractor.y,
                                        tractor.heading, custom_path)
            alpha_d  = math.degrees(pp_res.alpha_rad)

            # Adaptive throttle: slow down on sharp CTE
            throttle = THROTTLE_COLLECT if abs(cte) < 35 else 0.45
            tractor.update(pp_res.steer_input, throttle, 1/FPS)

            # Log row
            collect_rows.append({
                "cte_px":              round(cte, 3),
                "heading_error_deg":   round(math.degrees(h_err), 3),
                "speed_px_s":          round(tractor.speed, 2),
                "lookahead_angle_deg": round(alpha_d, 3),
                "steer_input":         round(pp_res.steer_input, 4),
            })

            # Noise injection
            collect_frame += 1
            if collect_frame % NOISE_INTERVAL == 0:
                inject_noise(tractor, custom_path)

            # Lap end?
            if custom_path.reached_destination(tractor.x, tractor.y):
                collect_lap += 1
                tractor = make_tractor(custom_path)
                collect_frame = 0

            # Done collecting?
            rows_done = len(collect_rows)
            if rows_done >= N_COLLECT_TARGET or collect_lap >= N_COLLECT_LAPS_MAX:
                start_training()
                stage = "TRAIN"

            # ── Render COLLECT ────────────────────────────────────────────────
            screen.fill(C_GRASS)
            draw_path_strip(screen, custom_path.waypoints)
            draw_path_edges(screen, custom_path.waypoints)
            draw_centre_dashes(screen, custom_path.waypoints)
            draw_markers(screen, custom_path.waypoints, font)

            # Lookahead line
            lx, ly = pp_res.lookahead_pt
            pygame.draw.line(screen, C_EXPERT,
                             (int(tractor.x), int(tractor.y)),
                             (int(lx), int(ly)), 1)
            pygame.draw.circle(screen, C_EXPERT, (int(lx), int(ly)), 5, 2)

            tractor.draw(screen)

            # Progress bar
            pct = min(rows_done / N_COLLECT_TARGET, 1.0)
            bx, by, bw, bh = 10, WINDOW_H-38, WINDOW_W-20, 10
            pygame.draw.rect(screen, (30, 30, 30), (bx, by, bw, bh), border_radius=5)
            pygame.draw.rect(screen, C_OK, (bx, by, int(bw*pct), bh), border_radius=5)

            lines = [
                (f"  Rows collected  {rows_done:>5,} / {N_COLLECT_TARGET:,}", C_WHITE),
                (f"  Lap             {collect_lap:>5}",                        C_WHITE),
                (f"  CTE             {cte:>+8.1f} px",
                 C_OK if abs(cte)<10 else C_WARN),
                (f"  Steer           {pp_res.steer_input:>+7.3f}",             C_WHITE),
            ]
            panel(screen, lines, (10, 10), 290, font)

            msg = font.render(f"Collecting…  {pct*100:.0f}%   Esc = abort",
                              True, (140, 140, 140))
            screen.blit(msg, (WINDOW_W//2 - msg.get_width()//2, WINDOW_H-18))

            pygame.display.flip()
            continue

        # ═════════════════════════════════════════════════════════════════════
        # TRAIN stage
        # ═════════════════════════════════════════════════════════════════════
        if stage == "TRAIN":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

            # Drain the training queue
            while True:
                try:
                    msg = train_q.get_nowait()
                    if msg[0] == 'epoch':
                        _, ep, loss = msg
                        train_epoch = ep
                        train_losses.append(loss)
                    elif msg[0] == 'done':
                        train_done = True
                    elif msg[0] == 'error':
                        print(f"[Train error] {msg[1]}")
                        train_done = True
                except queue.Empty:
                    break

            if train_done and os.path.exists(MODEL_OUT):
                # Load policy for demo
                ckpt = torch.load(MODEL_OUT, map_location="cpu",
                                  weights_only=False)
                _model = SteeringMLP()
                _model.load_state_dict(ckpt["model_state"])
                _model.eval()
                _mean  = torch.tensor(ckpt["norm_stats"]["mean"],
                                      dtype=torch.float32)
                _std   = torch.tensor(ckpt["norm_stats"]["std"],
                                      dtype=torch.float32)

                class _Policy:
                    def __init__(self, m, mn, ms):
                        self.m, self.mn, self.ms = m, mn, ms
                    @torch.no_grad()
                    def predict(self, cte, h_err, speed, alpha):
                        raw = torch.tensor([cte, h_err, speed, alpha],
                                           dtype=torch.float32)
                        x   = (raw - self.mn) / self.ms
                        return float(self.m(x.unsqueeze(0)))

                il_policy   = _Policy(_model, _mean, _std)
                demo_tractor = make_tractor(custom_path)
                stage        = "DEMO"
                print(f"[Phase 6] Training done  RMSE={ckpt['best_val_rmse']:.4f}  → DEMO")

            # ── Render TRAIN ──────────────────────────────────────────────────
            screen.fill((12, 12, 20))

            title = font_big.render("Training neural network…", True, C_GOLD)
            screen.blit(title, (WINDOW_W//2 - title.get_width()//2, 28))

            # Loss chart
            chart_rect = (80, 100, WINDOW_W-160, 220)
            draw_loss_chart(screen, train_losses, chart_rect, font)

            ep_lbl = font.render(
                f"Epoch {train_epoch} / {N_EPOCHS}   "
                f"{'Done!' if train_done else 'training…'}",
                True, C_WHITE)
            screen.blit(ep_lbl, (WINDOW_W//2 - ep_lbl.get_width()//2, 340))

            # Epoch progress bar
            if train_epoch > 0:
                pct = train_epoch / N_EPOCHS
                bx, by = 80, 370
                bw, bh = WINDOW_W-160, 8
                pygame.draw.rect(screen, (40,40,40), (bx, by, bw, bh), border_radius=4)
                pygame.draw.rect(screen, C_MODEL,
                                 (bx, by, int(bw*pct), bh), border_radius=4)

            rows_lbl = font.render(
                f"Training on {len(collect_rows):,} rows  |  "
                f"model → {MODEL_OUT}", True, (120,120,120))
            screen.blit(rows_lbl, (WINDOW_W//2 - rows_lbl.get_width()//2, 400))

            pygame.display.flip()
            continue

        # ═════════════════════════════════════════════════════════════════════
        # DEMO stage
        # ═════════════════════════════════════════════════════════════════════
        if stage == "DEMO":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_r:
                        demo_tractor = make_tractor(custom_path)
                        demo_lap     = 1
                        demo_cte_sum = 0.0
                        demo_cte_n   = 0
                    elif event.key == pygame.K_BACKSPACE:
                        control_pts  = list(custom_path.control_pts)
                        stage        = "DRAW"

            # ── IL inference ──────────────────────────────────────────────────
            cte     = custom_path.cross_track_error(demo_tractor.x,
                                                    demo_tractor.y)
            pp_res  = pp.compute(demo_tractor.x, demo_tractor.y,
                                 demo_tractor.heading, custom_path)
            h_err   = pp.heading_error(demo_tractor.x, demo_tractor.y,
                                       demo_tractor.heading, custom_path)

            steer_il = il_policy.predict(cte,
                                         math.degrees(h_err),
                                         demo_tractor.speed,
                                         math.degrees(pp_res.alpha_rad))
            steer_pp = pp_res.steer_input

            demo_tractor.update(steer_il, THROTTLE_COLLECT, dt)
            demo_cte_sum += abs(cte);  demo_cte_n += 1

            if custom_path.reached_destination(demo_tractor.x, demo_tractor.y):
                mean_cte = demo_cte_sum / max(demo_cte_n, 1)
                print(f"[Demo] Lap {demo_lap} done  mean|CTE|={mean_cte:.2f}px")
                demo_lap    += 1
                demo_tractor = make_tractor(custom_path)
                demo_cte_sum = 0.0
                demo_cte_n   = 0

            # ── Render DEMO ───────────────────────────────────────────────────
            screen.fill(C_GRASS)
            draw_path_strip(screen, custom_path.waypoints)
            draw_path_edges(screen, custom_path.waypoints)
            draw_centre_dashes(screen, custom_path.waypoints)
            draw_markers(screen, custom_path.waypoints, font)

            # Expert ghost (cyan line to PP lookahead)
            lx, ly = pp_res.lookahead_pt
            pygame.draw.line(screen, C_EXPERT,
                             (int(demo_tractor.x), int(demo_tractor.y)),
                             (int(lx), int(ly)), 1)
            pygame.draw.circle(screen, C_EXPERT, (int(lx), int(ly)), 5, 2)

            # IL model line (amber to same lookahead point for reference)
            pygame.draw.line(screen, C_MODEL,
                             (int(demo_tractor.x), int(demo_tractor.y)),
                             (int(lx), int(ly)), 1)

            demo_tractor.draw(screen)

            # HUD
            abs_cte  = abs(cte)
            cte_col  = C_OK if abs_cte < 10 else (C_WARN if abs_cte < 25 else C_DANGER)
            diff_col = C_OK if abs(steer_il-steer_pp) < 0.15 else C_WARN
            mean_cte = demo_cte_sum / max(demo_cte_n, 1)
            m_col    = C_OK if mean_cte < 12 else (C_WARN if mean_cte < 25 else C_DANGER)

            lines = [
                (f"  Lap         {demo_lap:>5}",                      C_WHITE),
                (f"  CTE      {cte:>+8.1f} px",                       cte_col),
                (f"  IL steer {steer_il:>+7.3f}",                     C_MODEL),
                (f"  PP steer {steer_pp:>+7.3f}",                     C_EXPERT),
                (f"  Δsteer   {abs(steer_il-steer_pp):>7.3f}",        diff_col),
                (f"  Mean|CTE|{mean_cte:>7.2f} px",                   m_col),
            ]
            panel(screen, lines, (WINDOW_W-220, 10), 215, font)

            # Legend
            leg_y = WINDOW_H - 50
            pygame.draw.line(screen, C_EXPERT, (10,leg_y),(34,leg_y), 2)
            pygame.draw.circle(screen, C_EXPERT, (34,leg_y), 4, 2)
            screen.blit(font.render("Expert (PP oracle)", True, C_EXPERT), (40, leg_y-6))
            pygame.draw.line(screen, C_MODEL, (10,leg_y+18),(34,leg_y+18), 2)
            screen.blit(font.render("IL model (trained)", True, C_MODEL), (40, leg_y+12))

            hint = font.render(
                "R restart  |  Backspace draw new path  |  Esc quit",
                True, (110,110,110))
            screen.blit(hint, (WINDOW_W//2 - hint.get_width()//2, WINDOW_H-18))

            pygame.display.flip()


if __name__ == "__main__":
    main()