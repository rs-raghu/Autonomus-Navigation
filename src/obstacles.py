"""
src/obstacles.py  —  Obstacle spawner and renderer

Two obstacle types:
  Log   — brown rounded rectangle, longer than wide, random rotation
  Rock  — dark gray irregular polygon (approximated as a circle with jitter)

Obstacles are placed by sampling a random waypoint from the middle 60 % of
the path (never right at HOME or FIELD) and offsetting slightly so they sit
visibly on the dirt strip.

Pygame is imported here only — vision.py stays dependency-free.
"""

import math
import random
import pygame

from src.path import Path

# ── Config ────────────────────────────────────────────────────────────────────
LOG_COLOR     = (101,  67,  33)    # dark brown
LOG_COLOR_OUT = ( 60,  40,  20)    # outline
ROCK_COLOR    = ( 80,  80,  80)    # dark grey
ROCK_COLOR_OUT= ( 45,  45,  45)

SPAWN_MARGIN  = 0.20   # skip first/last 20 % of waypoints on spawn


class Obstacle:
    """Base class — just holds position and type label."""
    def __init__(self, x: float, y: float, kind: str):
        self.x    = x
        self.y    = y
        self.kind = kind

    def draw(self, surface: pygame.Surface) -> None:  # pragma: no cover
        raise NotImplementedError


class Log(Obstacle):
    """A fallen branch: brown rectangle, random orientation."""

    def __init__(self, x: float, y: float):
        super().__init__(x, y, "log")
        self.length  = random.randint(12, 20)   # small enough to nudge around
        self.width   = random.randint(3, 6)
        self.angle   = random.uniform(0, math.pi)   # radians

    def draw(self, surface: pygame.Surface) -> None:
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        hl, hw = self.length / 2, self.width / 2

        corners = [
            (self.x + hl*cos_a - hw*sin_a,  self.y + hl*sin_a + hw*cos_a),
            (self.x + hl*cos_a + hw*sin_a,  self.y + hl*sin_a - hw*cos_a),
            (self.x - hl*cos_a + hw*sin_a,  self.y - hl*sin_a - hw*cos_a),
            (self.x - hl*cos_a - hw*sin_a,  self.y - hl*sin_a + hw*cos_a),
        ]
        int_corners = [(int(cx), int(cy)) for cx, cy in corners]
        pygame.draw.polygon(surface, LOG_COLOR,     int_corners)
        pygame.draw.polygon(surface, LOG_COLOR_OUT, int_corners, 1)
        # End circles for rounded-log look
        ex0 = (int(self.x + hl*cos_a), int(self.y + hl*sin_a))
        ex1 = (int(self.x - hl*cos_a), int(self.y - hl*sin_a))
        pygame.draw.circle(surface, LOG_COLOR,     ex0, self.width // 2)
        pygame.draw.circle(surface, LOG_COLOR_OUT, ex0, self.width // 2, 1)
        pygame.draw.circle(surface, LOG_COLOR,     ex1, self.width // 2)
        pygame.draw.circle(surface, LOG_COLOR_OUT, ex1, self.width // 2, 1)


class Rock(Obstacle):
    """An irregular rock: multi-point polygon approximating a lumpy circle."""

    def __init__(self, x: float, y: float):
        super().__init__(x, y, "rock")
        self.radius  = random.randint(4, 8)    # small enough to nudge around
        self._verts  = self._make_verts()

    def _make_verts(self) -> list[tuple[int, int]]:
        n = random.randint(7, 10)
        pts = []
        for i in range(n):
            angle = 2 * math.pi * i / n + random.uniform(-0.2, 0.2)
            r     = self.radius * random.uniform(0.7, 1.15)
            pts.append((int(self.x + r * math.cos(angle)),
                        int(self.y + r * math.sin(angle))))
        return pts

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.polygon(surface, ROCK_COLOR,     self._verts)
        pygame.draw.polygon(surface, ROCK_COLOR_OUT, self._verts, 1)
        # Small highlight circle
        pygame.draw.circle(surface, (110, 110, 110),
                           (int(self.x - self.radius * 0.2),
                            int(self.y - self.radius * 0.2)),
                           max(2, self.radius // 4))


# ── Manager ───────────────────────────────────────────────────────────────────

class ObstacleManager:
    """
    Maintains a list of obstacles and handles spawning.

    Call  spawn_random(path)  to add one obstacle at a random
    mid-path waypoint.  Call  draw_all(surface)  each frame.
    """

    def __init__(self):
        self.obstacles: list[Obstacle] = []

    def spawn_random(self, path: Path,
                     min_frac: float = SPAWN_MARGIN,
                     max_frac: float = 1.0 - SPAWN_MARGIN) -> Obstacle:
        """
        Place one random obstacle (log or rock) on a mid-path waypoint.

        min_frac / max_frac are fractions of total waypoints [0, 1].
        Pass the tractor's current progress as min_frac to guarantee the
        obstacle always lands *ahead* of the tractor.
        """
        wps   = path.waypoints
        lo    = int(len(wps) * min_frac)
        hi    = int(len(wps) * max_frac)
        lo    = max(0, min(lo, len(wps) - 2))
        hi    = max(lo + 1, min(hi, len(wps) - 1))
        wp    = wps[random.randint(lo, hi)]

        # Small random offset so it's not always dead-centre
        x = wp[0] + random.uniform(-10, 10)
        y = wp[1] + random.uniform(-10, 10)

        obs = random.choice([Log, Rock])(x, y)
        self.obstacles.append(obs)
        return obs

    def remove(self, obs: Obstacle) -> None:
        if obs in self.obstacles:
            self.obstacles.remove(obs)

    def clear(self) -> None:
        self.obstacles.clear()

    def draw_all(self, surface: pygame.Surface) -> None:
        for obs in self.obstacles:
            obs.draw(surface)

    def draw_detected(self, surface: pygame.Surface,
                      detected: list[Obstacle]) -> None:
        """Draw detected obstacles with a red highlight ring."""
        for obs in detected:
            r = getattr(obs, "radius", getattr(obs, "width", 14)) + 4
            pygame.draw.circle(surface, (255, 60, 60),
                               (int(obs.x), int(obs.y)), r, 2)  