"""
src/custom_path.py  —  User-drawn path with Catmull-Rom smoothing

Provides the same duck-typed interface as src/path.py so that PurePursuit,
Renderer, and the gym env all accept it without modification.

Public API
----------
  CustomPath(control_pts)   build from list of (x,y) click points
  .waypoints                list[(x,y)] — uniformly resampled spline
  .control_pts              original clicked points
  .cross_track_error(x,y)   signed CTE (px)
  .nearest_waypoint_index(x,y)
  .reached_destination(x,y)
  .validation()             PathValidation dataclass
  .arc_length()             total path length (px)
  .curvature_colors()       per-waypoint RGB for heatmap rendering

Module-level helpers
--------------------
  catmull_rom_chain(pts)    smooth spline through control points
  resample_uniform(pts)     uniform arc-length spacing
"""

import math
import json
from dataclasses import dataclass, field

DIRT_HALF_W    = 22       # matches src/path.py
ARRIVAL_RADIUS = 40       # px — destination reached threshold
RESAMPLE_STEP  = 4.0      # px between waypoints (same as fixed path)

# Ackermann min turning radius = WHEELBASE / tan(MAX_STEER) = 28 / tan(0.58)
MIN_TURN_RADIUS = 43.0    # px


# ── Spline helpers ────────────────────────────────────────────────────────────

def catmull_rom_chain(control_pts: list[tuple[float, float]],
                      samples_per_seg: int = 20) -> list[tuple[float, float]]:
    """
    Uniform Catmull-Rom spline through every control point.
    Pads the first/last point to keep the curve anchored at endpoints.
    """
    if len(control_pts) < 2:
        return list(control_pts)

    # Pad endpoints so spline reaches first and last control point
    padded = [control_pts[0]] + list(control_pts) + [control_pts[-1]]
    result: list[tuple[float, float]] = []

    for k in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[k-1], padded[k], padded[k+1], padded[k+2]
        for j in range(samples_per_seg):
            t  = j / samples_per_seg
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (2*p1[0] + (-p0[0]+p2[0])*t
                        + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2
                        + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5 * (2*p1[1] + (-p0[1]+p2[1])*t
                        + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2
                        + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((float(x), float(y)))

    result.append(tuple(control_pts[-1]))
    return result


def resample_uniform(pts: list[tuple[float, float]],
                     spacing: float = RESAMPLE_STEP) -> list[tuple[float, float]]:
    """Re-sample path to uniform arc-length spacing."""
    if len(pts) < 2:
        return list(pts)

    result = [pts[0]]
    acc    = 0.0

    for i in range(1, len(pts)):
        dx  = pts[i][0] - pts[i-1][0]
        dy  = pts[i][1] - pts[i-1][1]
        seg = math.hypot(dx, dy)
        if seg < 1e-6:
            continue

        while acc + seg >= spacing:
            t    = (spacing - acc) / seg
            newx = pts[i-1][0] + t * dx
            newy = pts[i-1][1] + t * dy
            result.append((newx, newy))
            # advance origin to newly placed point
            remaining = math.hypot(pts[i][0] - newx, pts[i][1] - newy)
            dx  = pts[i][0] - newx
            dy  = pts[i][1] - newy
            seg = remaining
            acc = 0.0

        acc += seg

    result.append(pts[-1])
    return result


def circumradius(p0, p1, p2) -> float:
    """Radius of the circumscribed circle of triangle p0-p1-p2."""
    a = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
    b = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    c = math.hypot(p2[0]-p0[0], p2[1]-p0[1])
    area = abs((p1[0]-p0[0])*(p2[1]-p0[1]) -
               (p2[0]-p0[0])*(p1[1]-p0[1])) / 2.0
    return (a * b * c / (4 * area)) if area > 1e-6 else float('inf')


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class PathValidation:
    ok:         bool
    warnings:   list[str] = field(default_factory=list)
    arc_length: float     = 0.0
    min_radius: float     = float('inf')
    sharp_turns:int       = 0


# ── CustomPath ────────────────────────────────────────────────────────────────

class CustomPath:
    """
    User-drawn path.  Identical duck-typed interface to src/path.py → Path.

    Parameters
    ----------
    control_pts : list of (x, y) screen coordinates clicked by the user.
                  Minimum 2 points; recommended 3+.
    """

    def __init__(self, control_pts: list[tuple[float, float]]):
        self.control_pts  = [(float(x), float(y)) for x, y in control_pts]
        spline            = catmull_rom_chain(self.control_pts, samples_per_seg=20)
        self.waypoints    = resample_uniform(spline, spacing=RESAMPLE_STEP)
        self._valid       = self._compute_validation()
        self._radii       = self._compute_radii()

    # ── Path interface (same as src/path.py) ──────────────────────────────────

    def cross_track_error(self, px: float, py: float) -> float:
        wps = self.waypoints
        best_dist, best_i = float('inf'), 0
        for i in range(len(wps) - 1):
            ax, ay = wps[i];  bx, by = wps[i+1]
            dx, dy = bx-ax, by-ay
            seg_sq = dx*dx + dy*dy
            if seg_sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/seg_sq))
            d = math.hypot(px-(ax+t*dx), py-(ay+t*dy))
            if d < best_dist:
                best_dist, best_i = d, i

        ax, ay = wps[best_i];  bx, by = wps[best_i+1]
        dx, dy = bx-ax, by-ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return best_dist
        t  = max(0.0, min(1.0, ((px-ax)*dx+(py-ay)*dy)/(seg_len**2)))
        fx = ax + t*dx;  fy = ay + t*dy
        ux, uy = dx/seg_len, dy/seg_len
        sign = ux*(py-fy) - uy*(px-fx)
        return best_dist * (1.0 if sign > 0 else -1.0)

    def nearest_waypoint_index(self, px: float, py: float) -> int:
        best_i, best_d = 0, float('inf')
        for i, (wx, wy) in enumerate(self.waypoints):
            d = math.hypot(px-wx, py-wy)
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def reached_destination(self, px: float, py: float) -> bool:
        ex, ey = self.waypoints[-1]
        return math.hypot(px-ex, py-ey) < ARRIVAL_RADIUS

    # ── Extra helpers ─────────────────────────────────────────────────────────

    def validation(self) -> PathValidation:
        return self._valid

    def arc_length(self) -> float:
        return self._valid.arc_length

    def start_heading(self) -> float:
        """Initial heading (radians) pointing from waypoint[0] toward waypoint[1]."""
        if len(self.waypoints) < 2:
            return 0.0
        dx = self.waypoints[1][0] - self.waypoints[0][0]
        dy = self.waypoints[1][1] - self.waypoints[0][1]
        return math.atan2(dy, dx)

    def curvature_colors(self) -> list[tuple[int, int, int]]:
        """
        Per-waypoint RGB colour encoding local turning radius.
          Green  → gentle (r ≥ 2 × MIN_TURN_RADIUS)
          Amber  → moderate
          Red    → tight  (r < MIN_TURN_RADIUS)
        """
        cols = []
        for r in self._radii:
            if r >= 2 * MIN_TURN_RADIUS:
                cols.append((72, 220, 88))    # green
            elif r >= MIN_TURN_RADIUS:
                cols.append((255, 162, 40))   # amber
            else:
                cols.append((255, 52, 52))    # red
        return cols

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps(self.control_pts)

    @staticmethod
    def from_json(s: str) -> "CustomPath":
        return CustomPath([tuple(p) for p in json.loads(s)])

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_validation(self) -> PathValidation:
        wps = self.waypoints
        if len(wps) < 2:
            return PathValidation(ok=False, warnings=["Need at least 2 points"])

        arc = sum(math.hypot(wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1])
                  for i in range(len(wps)-1))

        radii     = [circumradius(wps[i-1], wps[i], wps[i+1])
                     for i in range(1, len(wps)-1)]
        min_r     = min(radii) if radii else float('inf')
        sharp     = sum(1 for r in radii if r < MIN_TURN_RADIUS)

        warnings  = []
        if arc < 150:
            warnings.append(f"Path too short ({arc:.0f} px) — draw longer")
        if sharp > 0:
            warnings.append(f"{sharp} turn(s) tighter than Ackermann minimum "
                            f"({MIN_TURN_RADIUS:.0f} px radius)")

        return PathValidation(ok=not warnings, warnings=warnings,
                              arc_length=arc, min_radius=min_r,
                              sharp_turns=sharp)

    def _compute_radii(self) -> list[float]:
        wps = self.waypoints
        if len(wps) < 3:
            return [float('inf')] * len(wps)
        radii  = [float('inf')]
        radii += [circumradius(wps[i-1], wps[i], wps[i+1])
                  for i in range(1, len(wps)-1)]
        radii += [float('inf')]
        return radii