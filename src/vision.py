"""
src/vision.py  —  Vision cone geometry and obstacle detection

The cone is a triangle in world space:
  apex  = tractor front  (half_length + 5 px ahead of centre)
  left  = apex + depth rotated (heading − half_angle)
  right = apex + depth rotated (heading + half_angle)

Detection uses the Separating Axis Theorem lite: a point is inside
the cone triangle iff it is on the correct side of all three edges.

All geometry is pure Python / math — no Pygame dependency here.
"""

import math
from dataclasses import dataclass

from src.tractor import Tractor

# ── Defaults (can be overridden at construction) ──────────────────────────────
DEFAULT_HALF_ANGLE_DEG = 30.0    # half-width of the cone (degrees)
DEFAULT_DEPTH_PX       = 120.0   # how far ahead the cone reaches (px)


@dataclass
class ConeGeometry:
    """The three vertices of the cone triangle in screen coords."""
    apex : tuple[float, float]
    left : tuple[float, float]
    right: tuple[float, float]

    @property
    def polygon(self) -> list[tuple[float, float]]:
        return [self.apex, self.left, self.right]


def _cross(o, a, b) -> float:
    """2-D cross product of OA × OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(p, v0, v1, v2) -> bool:
    """Return True iff point p is inside or on the boundary of triangle (v0,v1,v2)."""
    d0 = _cross(v0, v1, p)
    d1 = _cross(v1, v2, p)
    d2 = _cross(v2, v0, p)
    has_neg = (d0 < 0) or (d1 < 0) or (d2 < 0)
    has_pos = (d0 > 0) or (d1 > 0) or (d2 > 0)
    return not (has_neg and has_pos)


class VisionCone:
    """
    Triangular forward-looking vision cone attached to the tractor.

    Usage
    -----
        cone    = VisionCone()
        geom    = cone.geometry(tractor)
        hit     = cone.detect(tractor, obstacles)   # list[Obstacle]
    """

    def __init__(self,
                 half_angle_deg: float = DEFAULT_HALF_ANGLE_DEG,
                 depth_px:       float = DEFAULT_DEPTH_PX):
        self.half_angle = math.radians(half_angle_deg)
        self.depth      = depth_px

    def geometry(self, tractor: Tractor) -> ConeGeometry:
        """Compute the current cone triangle from the tractor's pose."""
        # Apex: just in front of the tractor's nose
        nose_offset = Tractor.LENGTH / 2 + 5
        ax = tractor.x + nose_offset * math.cos(tractor.heading)
        ay = tractor.y + nose_offset * math.sin(tractor.heading)

        left_angle  = tractor.heading - self.half_angle
        right_angle = tractor.heading + self.half_angle

        lx = ax + self.depth * math.cos(left_angle)
        ly = ay + self.depth * math.sin(left_angle)
        rx = ax + self.depth * math.cos(right_angle)
        ry = ay + self.depth * math.sin(right_angle)

        return ConeGeometry(apex=(ax, ay), left=(lx, ly), right=(rx, ry))

    def detect(self, tractor: Tractor, obstacles: list) -> list:
        """
        Return the subset of obstacles whose centre falls inside the cone.

        Each obstacle must expose  .x  and  .y  attributes.
        """
        geom = self.geometry(tractor)
        v0, v1, v2 = geom.apex, geom.left, geom.right
        return [obs for obs in obstacles
                if _point_in_triangle((obs.x, obs.y), v0, v1, v2)]