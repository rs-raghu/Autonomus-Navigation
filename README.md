# Autonomous Agricultural Tractor Simulation

University project — Artificial Intelligence & Machine Learning applied to autonomous vehicle navigation.

---

## Project overview

A 2D top-down simulation of an autonomous tractor that learns to navigate a path from HOME to FIELD using a full AI/ML pipeline:

- **Ackermann steering physics** — realistic front-wheel kinematics, not teleportation
- **Pure Pursuit** — algorithmic expert generates a clean training dataset
- **Imitation Learning** — MLP warm-starts from expert demonstrations
- **Reinforcement Learning (SAC)** — fine-tunes the policy via continuous action-space RL
- **Vision cone + obstacle detection** — triangular forward scan, hard stop on detection
- **Custom path pipeline** — draw any path, auto-collect data, train, and demo in one window

---

## Setup

```bash
pip install -r requirements.txt
```

---

## File map

| File | Phase | Purpose |
|------|-------|---------|
| `main.py` | 1 | Manual keyboard driving + position-based CSV logger |
| `phase2_expert.py` | 2 | Pure Pursuit expert driver, noise injection, CSV collection |
| `train_il.py` | 3 | Train steering MLP on expert CSV (outputs `model_il.pt`) |
| `phase3_il.py` | 3 | Run trained IL model with PP ghost overlay |
| `phase4_rl.py` | 4 | SAC training with IL warm-start (outputs `models/sac_best.zip`) |
| `phase4_eval.py` | 4 | Evaluate SAC policy with live HUD |
| `phase5_vision.py` | 5 | SAC + vision cone + obstacle spawning + hard stop |
| `phase6_complete.py` | 6 | Full pipeline: Draw path → Preview → Collect → Train → Demo |

### `src/` modules

| Module | Role |
|--------|------|
| `tractor.py` | Ackermann kinematics + Pygame sprite |
| `path.py` | Fixed GPS waypoint path (5 sections, S-curve) |
| `custom_path.py` | User-drawn Catmull-Rom spline path with curvature validation |
| `pure_pursuit.py` | Pure Pursuit controller with lateral offset support |
| `renderer.py` | All Pygame drawing (path, tractor, HUD, banners, vision cone) |
| `gym_env.py` | Gymnasium wrapper for SAC training |
| `vision.py` | Triangular vision cone geometry + point-in-triangle detection |
| `obstacles.py` | Log and Rock obstacle types + spawning manager |

---

## Run order

### Phases 1–5 (fixed path)

```bash
# Phase 1 — manual drive and collect
python main.py

# Phase 2 — expert data collection (~50k rows, runs automatically)
python phase2_expert.py

# Phase 3 — train IL model, then run demo
python train_il.py
python phase3_il.py

# Phase 4 — SAC training (~15–30 min), then evaluate
python phase4_rl.py
python phase4_eval.py

# Phase 5 — vision cone + obstacle detection
python phase5_vision.py
python phase5_vision.py --no-rl       # Pure Pursuit only (debug)
```

### Phase 6 — full autonomous pipeline (custom path)

```bash
python phase6_complete.py
```

Five stages in one window:
1. **DRAW** — click to place waypoints, live spline preview, Enter to confirm
2. **PREVIEW** — curvature heatmap, validation warnings, Space to start collection
3. **COLLECT** — Pure Pursuit drives and logs data automatically
4. **TRAIN** — MLP trains in background, live loss curve shown
5. **DEMO** — trained model drives your custom path with PP ghost overlay

---

## Controls summary

| Key | Effect |
|-----|--------|
| W / ↑ | Throttle forward |
| S / ↓ | Reverse / brake |
| A / ← | Steer left |
| D / → | Steer right |
| Space | Pause / resume |
| R | Restart lap (or clear obstacle in Phase 5) |
| O | Spawn obstacle manually (Phase 5) |
| G | Toggle PP ghost overlay (Phases 3–4) |
| Esc | Quit |

---

## Generated files

| Path | Contents |
|------|----------|
| `reports/expert_N.csv` | Phase 2 expert dataset |
| `reports/trial_N.csv` | Phase 1 manual drive logs |
| `reports/il_loss.png` | Phase 3 training curve |
| `reports/il_pred.png` | Phase 3 predicted vs actual scatter |
| `reports/rl_training.csv` | Phase 4 episode reward log |
| `reports/rl_curve.png` | Phase 4 training curve |
| `reports/phase5_stops_N.csv` | Phase 5 obstacle stop events |
| `reports/custom_expert.csv` | Phase 6 collected data |
| `reports/custom_path.json` | Phase 6 saved path (S/L keys) |
| `model_il.pt` | Phase 3 IL checkpoint |
| `models/sac_best.zip` | Phase 4 best SAC policy |
| `models/custom_il.pt` | Phase 6 custom-path IL model |