"""
phase4_eval.py  —  Phase 4: Evaluate the trained SAC policy

Loads models/sac_best.zip (or models/sac_final.zip) and runs the tractor
in the Pygame simulation with a live stats overlay showing:
  • Per-step reward breakdown
  • Running mean |CTE|
  • Comparison: SAC steer vs Pure Pursuit steer
  • Lap count + arrival detection

Run:
    python phase4_eval.py                        # uses sac_best.zip
    python phase4_eval.py --model models/sac_final.zip
    python phase4_eval.py --laps 5               # run N laps then quit

Controls
--------
  SPACE     Pause / resume
  G         Toggle Pure Pursuit ghost overlay
  ESC       Quit and print summary
"""

import argparse
import math
import os
import sys
import csv
import time

import numpy as np
import pygame
from stable_baselines3 import SAC

from src.tractor      import Tractor
from src.path         import Path
from src.renderer     import Renderer
from src.pure_pursuit import PurePursuit
from src.gym_env      import TractorEnv, HOME_X, HOME_Y, HOME_HEADING, THROTTLE_RL, SIM_DT, LOOKAHEAD_PX

WINDOW_W, WINDOW_H = 900, 580

C_WHITE   = (220, 220, 220)
C_OK      = ( 72, 220,  88)
C_WARN    = (255, 162,  40)
C_DANGER  = (255,  52,  52)
C_SAC     = (200, 130, 255)   # purple — SAC lookahead line
C_GHOST   = ( 80, 220, 255)   # cyan   — Pure Pursuit oracle
C_GOLD    = (255, 215,   0)
C_PANEL   = (  0,   0,   0, 155)


def make_tractor() -> Tractor:
    return Tractor(x=HOME_X, y=HOME_Y, heading=HOME_HEADING)


def _panel(surf, lines, pos, w, font):
    h     = len(lines) * 18 + 10
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill(C_PANEL)
    pygame.draw.rect(panel, (110, 110, 110, 90), panel.get_rect(), 1, border_radius=6)
    for i, (text, col) in enumerate(lines):
        panel.blit(font.render(text, True, col), (4, 5 + i * 18))
    surf.blit(panel, pos)


def draw_eval_overlay(surf, font, tractor, cte, h_err_deg,
                      steer_sac, steer_pp, lap, lap_cte_sum,
                      lap_cte_count, show_ghost, pp_result,
                      paused, episode_reward):
    W, H = surf.get_size()

    # SAC lookahead line (to Pure Pursuit lookahead — best proxy for "target")
    lx, ly = pp_result.lookahead_pt
    pygame.draw.line(surf, C_SAC,
                     (int(tractor.x), int(tractor.y)),
                     (int(lx), int(ly)), 1)
    pygame.draw.circle(surf, C_SAC, (int(lx), int(ly)), 5, 2)

    if show_ghost:
        pygame.draw.line(surf, C_GHOST,
                         (int(tractor.x), int(tractor.y)),
                         (int(lx), int(ly)), 1)

    abs_cte   = abs(cte)
    cte_col   = C_OK if abs_cte < 10 else (C_WARN if abs_cte < 25 else C_DANGER)
    diff_col  = C_OK if abs(steer_sac - steer_pp) < 0.15 else C_WARN
    mean_cte  = lap_cte_sum / max(lap_cte_count, 1)
    m_col     = C_OK if mean_cte < 12 else (C_WARN if mean_cte < 25 else C_DANGER)

    lines = [
        (f"  Lap           {lap:>5}",                  C_WHITE),
        (f"  CTE        {cte:>+8.1f} px",              cte_col),
        (f"  Hdg err   {h_err_deg:>+7.1f} °",          C_WHITE),
        (f"  SAC steer {steer_sac:>+7.3f}",            C_SAC),
        (f"  PP  steer {steer_pp:>+7.3f}",             C_GHOST),
        (f"  Δsteer    {abs(steer_sac-steer_pp):>7.3f}", diff_col),
        (f"  Mean|CTE| {mean_cte:>7.2f} px",           m_col),
        (f"  Ep reward {episode_reward:>+8.1f}",        C_WHITE),
        (f"  {'PAUSED' if paused else 'RUNNING'}",      C_WARN if paused else C_OK),
    ]
    _panel(surf, lines, (W - 220, 10), 215, font)

    # Legend
    ly_ = H - 52
    pygame.draw.line(surf,   C_SAC,   (10, ly_),      (34, ly_),      2)
    pygame.draw.circle(surf, C_SAC,   (34, ly_), 4,   2)
    surf.blit(font.render("SAC policy", True, C_SAC), (40, ly_ - 6))

    if show_ghost:
        pygame.draw.line(surf,   C_GHOST, (10, ly_ + 18), (34, ly_ + 18), 2)
        pygame.draw.circle(surf, C_GHOST, (34, ly_ + 18), 4, 2)
        surf.blit(font.render("PP oracle (G hide)", True, C_GHOST), (40, ly_ + 12))
    else:
        surf.blit(font.render("G — show PP oracle", True, (120, 120, 120)),
                  (10, ly_ + 12))

    hint = font.render("SPACE pause  |  G ghost  |  ESC quit", True, (110, 110, 110))
    surf.blit(hint, (10, H - 18))


def draw_arrival(surf, font, lap, mean_cte):
    W, H = surf.get_size()
    ov   = pygame.Surface((W, 120), pygame.SRCALPHA)
    ov.fill((0, 0, 0, 190))
    surf.blit(ov, (0, H // 2 - 60))
    big = pygame.font.SysFont("monospace", 22, bold=True)
    title = big.render(f"Lap {lap} complete!", True, C_GOLD)
    surf.blit(title, (W//2 - title.get_width()//2, H//2 - 48))
    sub = font.render(
        f"Mean |CTE| = {mean_cte:.2f} px  —  restarting …", True, C_WHITE)
    surf.blit(sub, (W//2 - sub.get_width()//2, H//2 + 4))


def print_summary(rows, total_laps):
    if not rows:
        return
    ctes   = [abs(r["cte_px"]) for r in rows]
    steers = [r["steer_sac"]   for r in rows]
    diffs  = [abs(r["steer_sac"] - r["steer_pp"]) for r in rows]
    print("\n" + "=" * 52)
    print(f"  Phase 4 evaluation  ({total_laps} lap(s))")
    print("=" * 52)
    print(f"  Frames logged   : {len(rows):>8,}")
    print(f"  Mean  |CTE|     : {sum(ctes)/len(ctes):>8.2f} px")
    print(f"  Max   |CTE|     : {max(ctes):>8.2f} px")
    print(f"  Mean  |Δsteer|  : {sum(diffs)/len(diffs):>8.4f}")
    print("=" * 52 + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/sac_best.zip")
    p.add_argument("--laps",  type=int, default=0,
                   help="Number of laps to run (0 = run forever)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        # Fallback order
        for candidate in ["models/sac_best.zip", "models/sac_final.zip",
                          "models/sac_latest.zip"]:
            if os.path.exists(candidate):
                model_path = candidate
                break
        else:
            print(f"[Eval] No model found. Run  python phase4_rl.py  first.")
            sys.exit(1)

    print(f"[Eval] Loading {model_path} …")
    model = SAC.load(model_path)

    pygame.init()
    screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Phase 4 Eval — {os.path.basename(model_path)}")
    clock    = pygame.time.Clock()
    font     = pygame.font.SysFont("monospace", 13)

    path     = Path()
    pp       = PurePursuit(lookahead=LOOKAHEAD_PX)
    renderer = Renderer(screen)

    # Build a TractorEnv just to get observations (headless)
    env      = TractorEnv(render_mode=None)

    obs, _   = env.reset(seed=0)
    tractor  = env._tractor      # expose for rendering

    lap          = 1
    lap_cte_sum  = 0.0
    lap_cte_count= 0
    total_laps   = 0
    ep_reward    = 0.0
    paused       = False
    show_ghost   = True
    arrival_frames = 0
    ARRIVAL_DUR  = 90

    rows: list[dict] = []
    t0 = time.time()

    print("[Eval] Running SAC policy — SPACE pause, G ghost, ESC quit\n")

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    break
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_g:
                    show_ghost = not show_ghost
        else:
            if paused:
                cte       = path.cross_track_error(tractor.x, tractor.y)
                h_err_rad = pp.heading_error(tractor.x, tractor.y, tractor.heading, path)
                pp_res    = pp.compute(tractor.x, tractor.y, tractor.heading, path)
                steer_sac, _ = model.predict(obs, deterministic=True)
                renderer.draw_frame(path, tractor, cte)
                draw_eval_overlay(screen, font, tractor, cte,
                                  math.degrees(h_err_rad),
                                  float(steer_sac[0]), pp_res.steer_input,
                                  lap, lap_cte_sum, lap_cte_count,
                                  show_ghost, pp_res, True, ep_reward)
                pygame.display.flip()
                continue

            # ── SAC inference ────────────────────────────────────────────────
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            tractor = env._tractor

            cte       = path.cross_track_error(tractor.x, tractor.y)
            h_err_rad = pp.heading_error(tractor.x, tractor.y, tractor.heading, path)
            pp_res    = pp.compute(tractor.x, tractor.y, tractor.heading, path)

            lap_cte_sum   += abs(cte)
            lap_cte_count += 1

            rows.append({
                "t_s":      round(time.time() - t0, 3),
                "lap":      lap,
                "x":        round(tractor.x, 2),
                "y":        round(tractor.y, 2),
                "cte_px":   round(cte, 3),
                "steer_sac":round(float(action[0]), 4),
                "steer_pp": round(pp_res.steer_input, 4),
            })

            # ── Arrival / episode end ────────────────────────────────────────
            if terminated or truncated:
                mean_cte = lap_cte_sum / max(lap_cte_count, 1)
                arrived  = info.get("arrived", False)
                print(f"  Lap {lap:>3}  mean|CTE|={mean_cte:.2f}px  "
                      f"reward={ep_reward:+.1f}  "
                      f"{'ARRIVED' if arrived else 'OFF-PATH'}")

                if arrived:
                    total_laps     += 1
                    arrival_frames  = ARRIVAL_DUR

                if args.laps and total_laps >= args.laps:
                    break

                lap           += 1
                lap_cte_sum    = 0.0
                lap_cte_count  = 0
                ep_reward      = 0.0
                obs, _         = env.reset()
                tractor        = env._tractor

            # ── Render ──────────────────────────────────────────────────────
            renderer.draw_frame(path, tractor, cte)

            if arrival_frames > 0:
                draw_arrival(screen, font, lap - 1,
                             lap_cte_sum / max(lap_cte_count, 1))
                arrival_frames -= 1

            draw_eval_overlay(screen, font, tractor, cte,
                              math.degrees(h_err_rad),
                              float(action[0]), pp_res.steer_input,
                              lap, lap_cte_sum, lap_cte_count,
                              show_ghost, pp_res, False, ep_reward)
            pygame.display.flip()
            continue
        break

    # ── Save session CSV ──────────────────────────────────────────────────────
    env.close()
    if rows:
        os.makedirs("reports", exist_ok=True)
        n = 1
        while os.path.exists(f"reports/rl_eval_{n}.csv"):
            n += 1
        fp = f"reports/rl_eval_{n}.csv"
        with open(fp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Eval] Session log → {fp}")

    print_summary(rows, total_laps)
    pygame.quit()


if __name__ == "__main__":
    main()