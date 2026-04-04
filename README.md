# Autonomous Agricultural Tractor Simulation

University project — AI & Machine Learning applied to autonomous vehicle navigation.

---

## Phase 1 — Simulation Environment ✅

A 2D top-down Pygame simulation with realistic Ackermann steering kinematics,
a GPS waypoint path (right turn → left turn S-curve), and live cross-track error
display.

### Setup

```bash
pip install -r requirements.txt
python main.py
```

### Controls

| Key       | Action           |
|-----------|------------------|
| W / ↑     | Throttle forward |
| S / ↓     | Reverse / brake  |
| A / ←     | Steer left       |
| D / →     | Steer right      |
| ESC       | Quit             |

### What this phase delivers

| Component             | File                | Details                                      |
|-----------------------|---------------------|----------------------------------------------|
| Ackermann kinematics  | `src/tractor.py`    | ω = v·tan(δ)/L — realistic turning radius    |
| GPS waypoint path     | `src/path.py`       | S-curve: right turn → left turn, 5 sections  |
| Cross-track error     | `src/path.py`       | Signed CTE (+ = right of path, − = left)     |
| Simulation rendering  | `src/renderer.py`   | Grass, dirt track, HUD, markers              |
| Game loop             | `main.py`           | 60 fps, capped dt, keyboard input            |

---

## Upcoming phases

| Phase | Description                              |
|-------|------------------------------------------|
| 2     | Pure Pursuit expert driver + data logger |
| 3     | Imitation learning (MLP warm-start)      |
| 4     | SAC / DDPG reinforcement learning        |
| 5     | Vision cone + obstacle detection         |
| 6     | Flutter teleoperation app                |
| 7     | Fine-tuning + final report               |
