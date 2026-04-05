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
                   report_path: str = "",
                   cone_geom=None,
                   obstacle_mgr=None,
                   detected=None,
                   stopped: bool = False) -> None:
        """
        cone_geom    : ConeGeometry | None   — vision cone triangle
        obstacle_mgr : ObstacleManager | None
        detected     : list[Obstacle] | None  — obstacles currently in cone
        stopped      : bool                   — show hard-stop banner
        """
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

        # 6a. Vision cone (drawn behind tractor)
        if cone_geom is not None:
            self._draw_vision_cone(cone_geom, bool(detected))

        # 6b. Obstacles (drawn on dirt, behind tractor)
        if obstacle_mgr is not None:
            obstacle_mgr.draw_all(self.surf)
            if detected:
                obstacle_mgr.draw_detected(self.surf, detected)

        # 7. Tractor
        tractor.draw(self.surf)

        # 8. HUD
        self._draw_hud(tractor, cte)

        # 9. Arrival banner
        if arrived:
            self._draw_arrival_banner(W, H, report_path)

        # 10. Hard-stop alert
        if stopped:
            self._draw_stop_alert(W, H)

        # 11. Controls hint
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

    def _draw_vision_cone(self, geom, alert: bool) -> None:
        """Draw the triangular vision cone with a semi-transparent fill."""
        poly = [(int(p[0]), int(p[1])) for p in geom.polygon]

        # Transparent fill — green normally, red on alert
        cone_surf = pygame.Surface(self.surf.get_size(), pygame.SRCALPHA)
        fill_col  = (255, 60, 60, 55) if alert else (80, 220, 120, 40)
        pygame.draw.polygon(cone_surf, fill_col, poly)
        self.surf.blit(cone_surf, (0, 0))

        # Outline
        edge_col = (220, 60, 60) if alert else (80, 220, 120)
        pygame.draw.polygon(self.surf, edge_col, poly, 1)

    def _draw_stop_alert(self, W: int, H: int) -> None:
        """Flashing red banner shown when the tractor has performed a hard stop."""
        overlay = pygame.Surface((W, 80), pygame.SRCALPHA)
        overlay.fill((180, 0, 0, 200))
        self.surf.blit(overlay, (0, H // 2 - 40))

        title = self.font_arr.render("OBSTACLE DETECTED — HARD STOP", True, (255, 255, 255))
        self.surf.blit(title, (W // 2 - title.get_width() // 2, H // 2 - 30))

        sub = self.font_hud.render(
            "Press  R  to clear obstacle and resume  |  ESC quit",
            True, (220, 180, 180))
        self.surf.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 + 10))

    def draw_nudge_banner(self, direction: str, offset_px: float) -> None:
        """Amber info strip shown while the tractor is nudging around an obstacle."""
        W, H = self.surf.get_size()
        overlay = pygame.Surface((W, 44), pygame.SRCALPHA)
        overlay.fill((160, 100, 0, 200))
        self.surf.blit(overlay, (0, H // 2 - 22))
        arrow = "◄ LEFT" if direction == "left" else "RIGHT ►"
        msg   = self.font_hud.render(
            f"NUDGING {arrow}  |  offset {offset_px:+.1f} px  |  returning to path after clear",
            True, (255, 220, 120))
        self.surf.blit(msg, (W // 2 - msg.get_width() // 2, H // 2 - 9))

    def draw_cannot_pass_banner(self) -> None:
        """Red banner shown when both sides are blocked and a hard stop is forced."""
        W, H = self.surf.get_size()
        overlay = pygame.Surface((W, 90), pygame.SRCALPHA)
        overlay.fill((160, 0, 0, 210))
        self.surf.blit(overlay, (0, H // 2 - 45))
        title = self.font_arr.render("CANNOT PASS — PATH TOO NARROW", True, (255, 255, 255))
        self.surf.blit(title, (W // 2 - title.get_width() // 2, H // 2 - 36))
        sub = self.font_hud.render(
            "Both sides blocked  |  Press R to remove obstacle and resume",
            True, (220, 160, 160))
        self.surf.blit(sub, (W // 2 - sub.get_width() // 2, H // 2 + 8))

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