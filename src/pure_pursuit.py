"""
src/pure_pursuit.py  —  Pure Pursuit path-tracking controller

Algorithm
---------
1.  Find the nearest *segment* foot-point (not just the nearest vertex) so
    the lookahead walk always starts at or ahead of the tractor.
2.  Walk forward along the path until cumulative arc-length reaches L_d,
    interpolating the exact lookahead point and recording its segment tangent.
3.  Optionally shift the lookahead point laterally by `lateral_offset` pixels
    (positive = right of path direction, negative = left).  This makes the
    tractor converge to and maintain a constant offset from the centre-line
    without any extra CTE feedback loop.
4.  Compute α — signed angle from tractor heading to the (shifted) lookahead.
5.  Apply Pure Pursuit:  δ = atan2(2·L·sin(α),  L_d)
6.  Return a PPResult dataclass.

lateral_offset sign convention
--------------------------------
  +  →  right of the path travel direction
  −  →  left  of the path travel direction

  For a north-going segment (dy < 0 in Pygame):
    right = east (+x),   left = west (−x)    ✓  matches signed CTE convention.
"""

import math
from dataclasses import dataclass

from src.path    import Path
from src.tractor import Tractor

DEFAULT_LOOKAHEAD = 80.0   # pixels — works well at 75–100 px/s cruising speed


@dataclass(frozen=True)
class PPResult:
    steer_input  : float                # normalised [-1, 1] — straight into Tractor.update()
    lookahead_pt : tuple[float, float]  # screen (x, y) of the lookahead target
    alpha_rad    : float                # signed heading-to-lookahead angle (rad)


class PurePursuit:
    def __init__(self, lookahead: float = DEFAULT_LOOKAHEAD):
        self.lookahead = lookahead

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self,
                x: float, y: float, heading: float,
                path: Path,
                wheelbase: float = Tractor.WHEELBASE,
                lateral_offset: float = 0.0) -> PPResult:
        """
        Compute Pure Pursuit steering for the current pose.

        Parameters
        ----------
        x, y           : tractor position (px)
        heading        : tractor heading (rad;  0 = east,  −π/2 = north)
        path           : Path object
        wheelbase      : axle-to-axle distance (px)
        lateral_offset : signed offset from centre-line (px).
                         The lookahead point is shifted perpendicular to the
                         local path tangent, so the tractor tracks the offset
                         line, not the centre-line.
        """
        lp, seg_angle = self._find_lookahead(x, y, path)

        # ── Shift lookahead for offset laps ───────────────────────────────────
        if abs(lateral_offset) > 0.1:
            # Path tangent = (cos θ, sin θ)
            # Perpendicular rightward = (−sin θ, cos θ)
            perp_x = -math.sin(seg_angle)
            perp_y =  math.cos(seg_angle)
            lp = (lp[0] + perp_x * lateral_offset,
                  lp[1] + perp_y * lateral_offset)

        lx, ly = lp

        # ── Pure Pursuit formula ──────────────────────────────────────────────
        angle_to_lp = math.atan2(ly - y, lx - x)
        alpha       = (angle_to_lp - heading + math.pi) % (2 * math.pi) - math.pi
        Ld          = max(1e-3, math.hypot(lx - x, ly - y))
        delta       = math.atan2(2.0 * wheelbase * math.sin(alpha), Ld)
        steer_input = max(-1.0, min(1.0, delta / Tractor.MAX_STEER))

        return PPResult(steer_input  = steer_input,
                        lookahead_pt = lp,
                        alpha_rad    = alpha)

    def heading_error(self, x: float, y: float,
                      heading: float, path: Path) -> float:
        """
        Signed heading error (rad): tractor heading minus path tangent angle.

        +ve → tractor pointing right of path direction
        −ve → tractor pointing left
        """
        si = self._nearest_segment_index(x, y, path)
        wps = path.waypoints
        ax, ay = wps[si];   bx, by = wps[si + 1]
        path_angle = math.atan2(by - ay, bx - ax)
        err = heading - path_angle
        return (err + math.pi) % (2 * math.pi) - math.pi

    # ── Private helpers ───────────────────────────────────────────────────────

    def _nearest_segment_index(self, px: float, py: float, path: Path) -> int:
        """Index i of the segment (wps[i], wps[i+1]) whose foot-point is
        closest to (px, py).  More accurate than snapping to a vertex."""
        wps = path.waypoints
        best_i, best_d = 0, float('inf')
        for i in range(len(wps) - 1):
            ax, ay = wps[i];  bx, by = wps[i + 1]
            dx, dy = bx - ax, by - ay
            sq = dx*dx + dy*dy
            if sq < 1e-6:
                continue
            t = max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / sq))
            d = math.hypot(px - (ax + t*dx), py - (ay + t*dy))
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _find_lookahead(self, px: float, py: float,
                        path: Path) -> tuple[tuple[float, float], float]:
        """
        Walk forward from the nearest segment's foot-point until arc-length
        reaches self.lookahead.

        Returns
        -------
        (lookahead_point,  seg_angle)
          lookahead_point : (x, y) interpolated on the path
          seg_angle       : tangent angle (rad) of the segment that contains
                            the lookahead point — used for offset rotation
        """
        wps = path.waypoints
        si  = self._nearest_segment_index(px, py, path)

        # Foot-point on segment si
        ax, ay = wps[si];   bx, by = wps[si + 1]
        dx, dy = bx - ax,   by - ay
        sq     = dx*dx + dy*dy
        t0     = (max(0.0, min(1.0, ((px-ax)*dx + (py-ay)*dy) / sq))
                  if sq > 1e-6 else 0.0)
        fx     = ax + t0*dx;   fy = ay + t0*dy
        sa_si  = math.atan2(dy, dx)

        rem = math.hypot(bx - fx, by - fy)   # remaining on this segment
        acc = 0.0

        if rem >= self.lookahead:
            t  = self.lookahead / rem if rem > 1e-6 else 0.0
            return (fx + t*(bx-fx), fy + t*(by-fy)), sa_si

        acc += rem

        for i in range(si + 1, len(wps) - 1):
            ax, ay = wps[i];   bx, by = wps[i + 1]
            seg    = math.hypot(bx - ax, by - ay)
            sa     = math.atan2(by - ay, bx - ax)

            if acc + seg >= self.lookahead:
                t  = (self.lookahead - acc) / seg
                return (ax + t*(bx-ax), ay + t*(by-ay)), sa

            acc += seg

        # Fallback: destination, using last segment's tangent
        n       = len(wps)
        last_sa = math.atan2(wps[n-1][1] - wps[n-2][1],
                              wps[n-1][0] - wps[n-2][0])
        return wps[-1], last_sa