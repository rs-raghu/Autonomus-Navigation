"""
src/renderer.py  —  All Pygame drawing logic for the simulation.

Rendering order each frame:
  1. Grass background
  2. Dirt track strip  (wide brown line following waypoints)
  3. Track edge lines  (thin, darker)
  4. Dashed centre line
  5. Home / Field markers
  6. Tractor
  7. HUD overlay (position, speed, steer, CTE)
  8. Controls hint
"""

import math
import pygame

from src.path import Path, DIRT_HALF_W
from src.tractor import Tractor


# ── Palette ───────────────────────────────────────────────────────────────────
C_GRASS        = (55,  108,  42)
C_DIRT         = (128,  90,  48)
C_DIRT_EDGE    = (100,  70,  36)
C_CENTRE_LINE  = (162, 122,  70)
C_START        = ( 45, 210,  75)
C_END          = (210,  52,  52)
C_HUD_TEXT     = (212, 212, 212)
C_HUD_OK       = ( 72, 220,  88)
C_HUD_WARN     = (255, 162,  40)
C_HUD_DANGER   = (255,  52,  52)
C_WHITE        = (255, 255, 255)
C_BLACK        = (  0,   0,   0)


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surf     = surface
        self.font_hud = pygame.font.SysFont("monospace", 14)
        self.font_lbl = pygame.font.SysFont("monospace", 12)
        self.font_big = pygame.font.SysFont("monospace", 13, bold=True)
        # Semi-transparent HUD backing surface
        self.hud_surf = pygame.Surface((238, 128), pygame.SRCALPHA)

    # ── Public entry point ────────────────────────────────────────────────────

    def draw_frame(self, path: Path, tractor: Tractor, cte: float) -> None:
        wps = path.waypoints
        W, H = self.surf.get_size()

        # 1. Grass
        self.surf.fill(C_GRASS)

        # 2. Dirt strip (filled circles at each waypoint close gaps on curves)
        for i in range(len(wps) - 1):
            p1 = (int(wps[i][0]),     int(wps[i][1]))
            p2 = (int(wps[i + 1][0]), int(wps[i + 1][1]))
            pygame.draw.line(self.surf, C_DIRT, p1, p2, DIRT_HALF_W * 2)
        for wp in wps:
            pygame.draw.circle(self.surf, C_DIRT,
                               (int(wp[0]), int(wp[1])), DIRT_HALF_W)

        # 3. Track edge lines (thin, both sides)
        self._draw_edge_lines(wps, DIRT_HALF_W - 1)

        # 4. Dashed centre line
        self._draw_dashed_centreline(wps)

        # 5. Start / end markers
        self._draw_markers(wps)

        # 6. Tractor
        tractor.draw(self.surf)

        # 7. HUD
        self._draw_hud(tractor, cte)

        # 8. Controls hint
        hint = self.font_lbl.render(
            "W/↑ forward    S/↓ reverse    A/← left    D/→ right    ESC quit",
            True, (155, 155, 155))
        self.surf.blit(hint, (10, H - 20))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _draw_edge_lines(self, wps, offset: int) -> None:
        """Draw left and right edge lines along the path."""
        for side in (-1, 1):
            for i in range(len(wps) - 1):
                ax, ay = wps[i]
                bx, by = wps[i + 1]
                dx, dy = bx - ax, by - ay
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                nx, ny = -dy / seg_len * side, dx / seg_len * side  # normal
                p1 = (int(ax + nx * offset), int(ay + ny * offset))
                p2 = (int(bx + nx * offset), int(by + ny * offset))
                pygame.draw.line(self.surf, C_DIRT_EDGE, p1, p2, 1)

    def _draw_dashed_centreline(self, wps) -> None:
        """Dashed line down the centre of the path."""
        for i in range(0, len(wps) - 1, 4):
            p1 = (int(wps[i][0]),     int(wps[i][1]))
            p2 = (int(wps[i + 1][0]), int(wps[i + 1][1]))
            pygame.draw.line(self.surf, C_CENTRE_LINE, p1, p2, 1)

    def _draw_markers(self, wps) -> None:
        """Green circle = Home  |  Red circle = Field"""
        sx, sy = int(wps[0][0]),  int(wps[0][1])
        ex, ey = int(wps[-1][0]), int(wps[-1][1])

        # Home
        pygame.draw.circle(self.surf, C_START,  (sx, sy), 13)
        pygame.draw.circle(self.surf, C_WHITE,   (sx, sy), 13, 2)
        lbl = self.font_big.render("HOME", True, C_BLACK)
        self.surf.blit(lbl, (sx - lbl.get_width() // 2, sy - 28))

        # Field
        pygame.draw.circle(self.surf, C_END,    (ex, ey), 13)
        pygame.draw.circle(self.surf, C_WHITE,   (ex, ey), 13, 2)
        lbl = self.font_big.render("FIELD", True, C_BLACK)
        self.surf.blit(lbl, (ex - lbl.get_width() // 2, ey - 28))

    def _draw_hud(self, tractor: Tractor, cte: float) -> None:
        """Semi-transparent HUD panel — top-left corner."""
        # Backing rect
        self.hud_surf.fill((0, 0, 0, 150))
        pygame.draw.rect(self.hud_surf, (110, 110, 110, 90),
                         self.hud_surf.get_rect(), 1, border_radius=6)

        # CTE colour coding
        abs_cte = abs(cte)
        if abs_cte < 10:
            cte_col = C_HUD_OK
        elif abs_cte < 25:
            cte_col = C_HUD_WARN
        else:
            cte_col = C_HUD_DANGER

        speed_px_s = tractor.speed
        speed_kmh  = speed_px_s * 0.1   # rough display scale

        rows = [
            (f"  X       {tractor.x:>8.1f} px",       C_HUD_TEXT),
            (f"  Y       {tractor.y:>8.1f} px",       C_HUD_TEXT),
            (f"  Heading {math.degrees(tractor.heading):>7.1f} °",  C_HUD_TEXT),
            (f"  Speed   {speed_px_s:>7.1f} px/s",    C_HUD_TEXT),
            (f"  Steer   {math.degrees(tractor.steer_angle):>7.1f} °",  C_HUD_TEXT),
            (f"  CTE     {cte:>+8.1f} px",            cte_col),
        ]

        for i, (text, color) in enumerate(rows):
            surf = self.font_hud.render(text, True, color)
            self.hud_surf.blit(surf, (4, 6 + i * 19))

        self.surf.blit(self.hud_surf, (10, 10))
