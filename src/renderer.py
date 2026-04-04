"""
src/renderer.py  —  All Pygame drawing for the simulation.

Rendering order each frame:
  1. Grass background
  2. Dirt track strip
  3. Track edge lines (thin)
  4. Dashed centre line
  5. Home / Field markers
  6. Tractor
  7. HUD (pos, speed, steer, CTE)
  8. Arrival banner (when destination reached)
  9. Controls hint
"""

import math
import pygame

from src.path    import Path, DIRT_HALF_W
from src.tractor import Tractor


C_GRASS       = ( 55, 108,  42)
C_DIRT        = (128,  90,  48)
C_DIRT_EDGE   = (100,  70,  36)
C_CENTRE      = (162, 122,  70)
C_START       = ( 45, 210,  75)
C_END         = (210,  52,  52)
C_HUD_TEXT    = (212, 212, 212)
C_HUD_OK      = ( 72, 220,  88)
C_HUD_WARN    = (255, 162,  40)
C_HUD_DANGER  = (255,  52,  52)
C_WHITE       = (255, 255, 255)
C_BLACK       = (  0,   0,   0)
C_GOLD        = (255, 215,   0)


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surf     = surface
        self.font_hud = pygame.font.SysFont("monospace", 14)
        self.font_lbl = pygame.font.SysFont("monospace", 12)
        self.font_big = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_arr = pygame.font.SysFont("monospace", 22, bold=True)
        self.hud_surf = pygame.Surface((242, 130), pygame.SRCALPHA)

    def draw_frame(self, path: Path, tractor: Tractor,
                   cte: float, arrived: bool = False,
                   report_path: str = "") -> None:
        wps = path.waypoints
        W, H = self.surf.get_size()

        # 1. Grass
        self.surf.fill(C_GRASS)

        # 2. Dirt strip — circles at waypoints to fill curve gaps
        for i in range(len(wps) - 1):
            pygame.draw.line(self.surf, C_DIRT,
                             (int(wps[i][0]),   int(wps[i][1])),
                             (int(wps[i+1][0]), int(wps[i+1][1])),
                             DIRT_HALF_W * 2)
        for wp in wps:
            pygame.draw.circle(self.surf, C_DIRT,
                               (int(wp[0]), int(wp[1])), DIRT_HALF_W)

        # 3. Track edge lines
        self._draw_edge_lines(wps, DIRT_HALF_W - 1)

        # 4. Dashed centre line
        self._draw_dashed_centreline(wps)

        # 5. Markers
        self._draw_markers(wps)

        # 6. Tractor
        tractor.draw(self.surf)

        # 7. HUD
        self._draw_hud(tractor, cte)

        # 8. Arrival banner
        if arrived:
            self._draw_arrival_banner(W, H, report_path)

        # 9. Controls hint
        hint = self.font_lbl.render(
            "W/↑ fwd   S/↓ rev   A/← left   D/→ right   R restart   ESC quit",
            True, (155, 155, 155))
        self.surf.blit(hint, (10, H - 20))

    # ── Private ───────────────────────────────────────────────────────────────

    def _draw_edge_lines(self, wps, offset: int) -> None:
        for side in (-1, 1):
            for i in range(len(wps) - 1):
                ax, ay = wps[i];  bx, by = wps[i+1]
                dx, dy = bx-ax, by-ay
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-6:
                    continue
                nx = -dy / seg_len * side
                ny =  dx / seg_len * side
                pygame.draw.line(self.surf, C_DIRT_EDGE,
                                 (int(ax + nx*offset), int(ay + ny*offset)),
                                 (int(bx + nx*offset), int(by + ny*offset)), 1)

    def _draw_dashed_centreline(self, wps) -> None:
        for i in range(0, len(wps) - 1, 4):
            pygame.draw.line(self.surf, C_CENTRE,
                             (int(wps[i][0]),   int(wps[i][1])),
                             (int(wps[i+1][0]), int(wps[i+1][1])), 1)

    def _draw_markers(self, wps) -> None:
        # HOME — bottom of path
        sx, sy = int(wps[0][0]),  int(wps[0][1])
        pygame.draw.circle(self.surf, C_START, (sx, sy), 14)
        pygame.draw.circle(self.surf, C_WHITE,  (sx, sy), 14, 2)
        lbl = self.font_big.render("HOME", True, C_BLACK)
        self.surf.blit(lbl, (sx - lbl.get_width()//2, sy + 18))

        # FIELD — top of path
        ex, ey = int(wps[-1][0]), int(wps[-1][1])
        pygame.draw.circle(self.surf, C_END, (ex, ey), 14)
        pygame.draw.circle(self.surf, C_WHITE, (ex, ey), 14, 2)
        lbl = self.font_big.render("FIELD", True, C_BLACK)
        self.surf.blit(lbl, (ex - lbl.get_width()//2, ey - 30))

    def _draw_hud(self, tractor: Tractor, cte: float) -> None:
        self.hud_surf.fill((0, 0, 0, 148))
        pygame.draw.rect(self.hud_surf, (110, 110, 110, 90),
                         self.hud_surf.get_rect(), 1, border_radius=6)

        abs_cte = abs(cte)
        cte_col = C_HUD_OK if abs_cte < 10 else (C_HUD_WARN if abs_cte < 25 else C_HUD_DANGER)

        rows = [
            (f"  X       {tractor.x:>8.1f} px",                    C_HUD_TEXT),
            (f"  Y       {tractor.y:>8.1f} px",                    C_HUD_TEXT),
            (f"  Heading {math.degrees(tractor.heading):>7.1f} °",  C_HUD_TEXT),
            (f"  Speed   {tractor.speed:>7.1f} px/s",               C_HUD_TEXT),
            (f"  Steer   {math.degrees(tractor.steer_angle):>7.1f} °", C_HUD_TEXT),
            (f"  CTE     {cte:>+8.1f} px",                         cte_col),
        ]
        for i, (text, color) in enumerate(rows):
            self.hud_surf.blit(self.font_hud.render(text, True, color),
                               (4, 6 + i * 19))
        self.surf.blit(self.hud_surf, (10, 10))

    def _draw_arrival_banner(self, W: int, H: int, report_path: str) -> None:
        # Dark overlay strip
        overlay = pygame.Surface((W, 110), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 185))
        self.surf.blit(overlay, (0, H//2 - 55))

        title = self.font_arr.render("DESTINATION REACHED!", True, C_GOLD)
        self.surf.blit(title, (W//2 - title.get_width()//2, H//2 - 44))

        if report_path:
            sub = self.font_hud.render(
                f"Trial saved → {report_path}   |   Press R to restart",
                True, C_HUD_TEXT)
            self.surf.blit(sub, (W//2 - sub.get_width()//2, H//2 + 4))