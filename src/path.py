"""
src/path.py  —  GPS waypoint path definition + cross-track error

Path layout (top-down, screen coordinates, y increases downward):

  HOME ──straight east──► [right turn 90°] ──straight south──► [left turn 90°] ──straight east──► FIELD
  (80, 220)               arc R=120                (320, 340→420)  arc R=120              (440→755, 300)

The two arcs give the classic S-curve:  ─────╮
                                               │
                                              ╰──────
"""

import math
import numpy as np

DIRT_HALF_W = 22   # half-width of the visual dirt strip in pixels


class Path:
    def __init__(self):
        self.waypoints: list[tuple[float, float]] = _build_waypoints()

    def cross_track_error(self, px: float, py: float) -> float:
        """
        Signed cross-track error (px).
          +ve → tractor is to the RIGHT of the path direction (screen space)
          -ve → tractor is to the LEFT
        """
        wps = self.waypoints
        best_dist, best_i = float('inf'), 0

        for i in range(len(wps) - 1):
            ax, ay = wps[i]
            bx, by = wps[i + 1]
            dx, dy = bx - ax, by - ay
            seg_sq = dx * dx + dy * dy
            if seg_sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
            d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d < best_dist:
                best_dist, best_i = d, i

        # Recompute nearest foot-point on best segment for sign
        ax, ay = wps[best_i]
        bx, by = wps[best_i + 1]
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return best_dist

        t  = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (seg_len * seg_len)))
        fx = ax + t * dx
        fy = ay + t * dy

        # Cross product of path direction with offset vector → sign of CTE
        ux, uy = dx / seg_len, dy / seg_len
        sign   = ux * (py - fy) - uy * (px - fx)
        return best_dist * (1.0 if sign > 0 else -1.0)

    def nearest_waypoint_index(self, px: float, py: float) -> int:
        """Return index of the closest waypoint to (px, py)."""
        wps  = self.waypoints
        best_i, best_d = 0, float('inf')
        for i, (wx, wy) in enumerate(wps):
            d = math.hypot(px - wx, py - wy)
            if d < best_d:
                best_d, best_i = d, i
        return best_i


# ── Waypoint generation ───────────────────────────────────────────────────────

def _build_waypoints() -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    R = 120.0   # arc radius for both turns

    # ── Section 1: straight east  (80, 220) → (200, 220) ────────────────────
    for x in np.arange(80, 200, 4):
        pts.append((float(x), 220.0))

    # ── Section 2: right turn 90°  centre=(200, 340) R=120 ──────────────────
    # Standard-math θ: π/2 → 0  (clockwise in Pygame screen coords)
    # Tangent at θ=π/2 → east; tangent at θ=0 → south  ✓
    for i in range(25):
        θ = (math.pi / 2) * (1.0 - i / 24.0)
        pts.append((200.0 + R * math.cos(θ),
                    340.0 - R * math.sin(θ)))

    # ── Section 3: straight south  (320, 340) → (320, 420) ──────────────────
    for y in np.arange(340, 422, 4):
        pts.append((320.0, float(y)))

    # ── Section 4: left turn 90°  centre=(440, 420) R=120 ───────────────────
    # θ: π → π/2  (counter-clockwise in screen = tractor turns left)
    # Tangent at θ=π → south; tangent at θ=π/2 → east  ✓
    for i in range(25):
        θ = math.pi - (math.pi / 2) * (i / 24.0)
        pts.append((440.0 + R * math.cos(θ),
                    420.0 - R * math.sin(θ)))

    # ── Section 5: straight east  (440, 300) → (755, 300) ───────────────────
    for x in np.arange(440, 760, 4):
        pts.append((float(x), 300.0))

    return pts
