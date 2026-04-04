"""
src/tractor.py  —  Ackermann-steering tractor model

Kinematics
----------
For a bicycle (single-track) model with front-wheel steering angle δ
and wheelbase L, the rear-axle angular velocity is:

    ω = v · tan(δ) / L

Heading and position update at each timestep dt:

    heading += ω · dt
    x       += v · cos(heading) · dt
    y       += v · sin(heading) · dt        (y positive downward in Pygame)

The front wheels are drawn at their actual steering angle, giving a clear
visual indicator of the current steering command.
"""

import math
import pygame


class Tractor:
    # ── Physical parameters ──────────────────────────────────────────────────
    WHEELBASE  = 28.0    # distance between axle centres (px)
    LENGTH     = 44.0    # full body length (px)
    WIDTH      = 26.0    # full body width  (px)
    MAX_STEER  = 0.58    # max front-wheel angle ≈ 33°
    MAX_SPEED  = 130.0   # forward speed cap  (px / s)
    MAX_REV    = 40.0    # reverse speed cap  (px / s)
    ACCEL      = 90.0    # throttle acceleration (px / s²)
    DRAG       = 0.84    # per-second speed multiplier (rolling friction)
    STEER_RATE = 7.5     # steering response constant  (higher = snappier)

    # ── Palette ──────────────────────────────────────────────────────────────
    C_BODY     = (218, 175, 36)
    C_CABIN    = (172, 136, 26)
    C_WHEEL    = (42, 42, 42)
    C_AXLE     = (90, 90, 90)
    C_FRONT    = (255, 62, 62)    # red dot at front

    def __init__(self, x: float, y: float, heading: float = 0.0):
        """
        x, y    : initial world position (px)
        heading : initial heading in radians  (0 = east, π/2 = south)
        """
        self.x           = x
        self.y           = y
        self.heading     = heading
        self.speed       = 0.0   # current speed (px / s); negative = reversing
        self.steer_angle = 0.0   # actual front-wheel angle (rad)

    # ── Physics ──────────────────────────────────────────────────────────────

    def update(self, steer_input: float, throttle_input: float, dt: float) -> None:
        """
        steer_input    : normalised [-1, 1]   negative = steer left
        throttle_input : normalised [-1, 1]   negative = reverse
        dt             : elapsed seconds
        """
        # — Steering with lag ————————————————————————————————————————————————
        target_steer     = steer_input * self.MAX_STEER
        alpha            = min(1.0, self.STEER_RATE * dt)
        self.steer_angle += (target_steer - self.steer_angle) * alpha
        self.steer_angle  = max(-self.MAX_STEER,
                                min( self.MAX_STEER, self.steer_angle))

        # — Speed ————————————————————————————————————————————————————————————
        if throttle_input > 0:
            self.speed += throttle_input * self.ACCEL * dt
        elif throttle_input < 0:
            self.speed += throttle_input * self.ACCEL * 1.6 * dt   # firmer braking
        self.speed *= (self.DRAG ** dt)
        self.speed  = max(-self.MAX_REV, min(self.MAX_SPEED, self.speed))

        # — Ackermann kinematics ——————————————————————————————————————————————
        if abs(self.steer_angle) > 1e-4:
            turning_radius = self.WHEELBASE / math.tan(self.steer_angle)
            omega          = self.speed / turning_radius
        else:
            omega = 0.0

        self.heading += omega * dt
        self.x       += self.speed * math.cos(self.heading) * dt
        self.y       += self.speed * math.sin(self.heading) * dt

    # ── Drawing ──────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface) -> None:
        """Render the tractor (top-down view) onto surface."""
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        def rot(lx: float, ly: float) -> tuple[int, int]:
            """Rotate local body coords → screen coords."""
            return (int(self.x + lx * cos_h - ly * sin_h),
                    int(self.y + lx * sin_h + ly * cos_h))

        half_l = self.LENGTH / 2
        half_w = self.WIDTH  / 2

        # ── Main body ──────────────────────────────────────────────────────
        body = [rot( half_l,  half_w),
                rot( half_l, -half_w),
                rot(-half_l, -half_w),
                rot(-half_l,  half_w)]
        pygame.draw.polygon(surface, self.C_BODY, body)
        pygame.draw.polygon(surface, (25, 25, 25), body, 1)

        # ── Cabin (rear ~55 % of body, slightly darker) ───────────────────
        cabin_front_x = half_l * 0.1
        cabin = [rot(cabin_front_x,  half_w - 2),
                 rot(cabin_front_x, -half_w + 2),
                 rot(-half_l + 3,   -half_w + 2),
                 rot(-half_l + 3,    half_w - 2)]
        pygame.draw.polygon(surface, self.C_CABIN, cabin)

        # ── Wheels ────────────────────────────────────────────────────────
        wl, ww       = 10, 4    # wheel length, width
        rear_ax_x    = -self.WHEELBASE / 2
        front_ax_x   =  self.WHEELBASE / 2
        axle_half_w  = half_w + 4   # wheels stick out slightly

        for (ax_x, wheel_steer) in [(rear_ax_x, 0.0),
                                    (front_ax_x, self.steer_angle)]:
            for side in (-1, 1):
                wx, wy = rot(ax_x, side * axle_half_w)
                wa     = self.heading + wheel_steer
                wcos, wsin = math.cos(wa), math.sin(wa)
                corners = [
                    (wx + wl/2*wcos - ww/2*wsin, wy + wl/2*wsin + ww/2*wcos),
                    (wx + wl/2*wcos + ww/2*wsin, wy + wl/2*wsin - ww/2*wcos),
                    (wx - wl/2*wcos + ww/2*wsin, wy - wl/2*wsin - ww/2*wcos),
                    (wx - wl/2*wcos - ww/2*wsin, wy - wl/2*wsin + ww/2*wcos),
                ]
                pygame.draw.polygon(surface, self.C_WHEEL, corners)

            # Axle line between wheels
            lw, rw = rot(ax_x, -axle_half_w), rot(ax_x, axle_half_w)
            pygame.draw.line(surface, self.C_AXLE, lw, rw, 2)

        # ── Front direction indicator (red dot) ───────────────────────────
        fx, fy = rot(half_l + 5, 0)
        pygame.draw.circle(surface, self.C_FRONT, (fx, fy), 5)
        pygame.draw.circle(surface, (255, 255, 255), (fx, fy), 5, 1)
