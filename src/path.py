"""
src/path.py  —  GPS waypoint path  (vertical, bottom → top)

Path layout (screen coords, y increases downward):

  FIELD  (260, 60)
    |
    |  section 5 — straight north
    |
  [right turn: west → north]   arc center (360, 200)  R=100
    |
    ──────────────────────────  section 3 — straight west
    |
  [left turn: north → west]    arc center (540, 400)  R=100
    |
    |  section 1 — straight north
    |
  HOME  (640, 530)

No overlap: S1 at x=640, S3 at y=300 (x 360–540), S5 at x=260.
"""

import math
import numpy as np

DIRT_HALF_W   = 22    # half-width of the visual dirt strip (px)
R             = 100.0 # arc radius for both turns
ARRIVAL_RADIUS = 38   # px — how close counts as "reached destination"


class Path:
    def __init__(self):
        self.waypoints: list[tuple[float, float]] = _build_waypoints()

    def cross_track_error(self, px: float, py: float) -> float:
        """Signed CTE. +ve = right of path direction, −ve = left."""
        wps = self.waypoints
        best_dist, best_i = float('inf'), 0
        for i in range(len(wps) - 1):
            ax, ay = wps[i];  bx, by = wps[i + 1]
            dx, dy = bx - ax, by - ay
            seg_sq = dx*dx + dy*dy
            if seg_sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / seg_sq))
            d = math.hypot(px-(ax+t*dx), py-(ay+t*dy))
            if d < best_dist:
                best_dist, best_i = d, i

        ax, ay = wps[best_i];  bx, by = wps[best_i+1]
        dx, dy = bx-ax, by-ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return best_dist
        t  = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / (seg_len**2)))
        fx = ax + t*dx;  fy = ay + t*dy
        ux, uy = dx/seg_len, dy/seg_len
        sign = ux*(py-fy) - uy*(px-fx)
        return best_dist * (1.0 if sign > 0 else -1.0)

    def nearest_waypoint_index(self, px: float, py: float) -> int:
        wps = self.waypoints
        best_i, best_d = 0, float('inf')
        for i, (wx, wy) in enumerate(wps):
            d = math.hypot(px-wx, py-wy)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def reached_destination(self, px: float, py: float) -> bool:
        ex, ey = self.waypoints[-1]
        return math.hypot(px-ex, py-ey) < ARRIVAL_RADIUS


def _build_waypoints() -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []

    # ── Section 1: straight NORTH  (640,530) → (640,400) ─────────────────────
    for y in np.arange(530, 398, -4):
        pts.append((640.0, float(y)))

    # ── Section 2: LEFT turn  north → west ───────────────────────────────────
    # Center (540,400), θ: 0 → −π/2
    #   θ= 0    → (640,400)  heading north ✓
    #   θ=−π/2  → (540,300)  heading west  ✓
    for i in range(25):
        θ = -(math.pi / 2) * (i / 24.0)
        pts.append((540.0 + R*math.cos(θ), 400.0 + R*math.sin(θ)))

    # ── Section 3: straight WEST  (540,300) → (360,300) ──────────────────────
    for x in np.arange(540, 358, -4):
        pts.append((float(x), 300.0))

    # ── Section 4: RIGHT turn  west → north ──────────────────────────────────
    # Center (360,200), θ: π/2 → π
    #   θ=π/2 → (360,300)  heading west  ✓
    #   θ=π   → (260,200)  heading north ✓
    for i in range(25):
        θ = (math.pi / 2) + (math.pi / 2) * (i / 24.0)
        pts.append((360.0 + R*math.cos(θ), 200.0 + R*math.sin(θ)))

    # ── Section 5: straight NORTH  (260,200) → (260,60) ──────────────────────
    for y in np.arange(200, 58, -4):
        pts.append((260.0, float(y)))

    return pts