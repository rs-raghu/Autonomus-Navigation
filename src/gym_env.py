"""
src/gym_env.py  —  Phase 4: Gymnasium wrapper for the tractor simulation

Wraps the Ackermann tractor + GPS path into a standard Gymnasium environment
that SAC (or any continuous-action algorithm) can train against.

Observation space  (6 floats, all normalised to roughly [-1, 1])
-----------------------------------------------------------------
  0  cte_norm          signed CTE / 60.0        (clip at ±60 px)
  1  heading_err_norm  heading error / π
  2  speed_norm        speed / MAX_SPEED
  3  alpha_norm        lookahead angle / π
  4  steer_norm        current front-wheel angle / MAX_STEER
  5  progress          fraction of waypoints passed  [0, 1]

Action space  (1 float)
-----------------------
  steer_input  [-1, 1]   passed straight to Tractor.update()

Reward  (shaped for smooth, on-path driving)
--------------------------------------------
  +1.0                  for each step alive on path (survival bonus)
  −|cte| / 60           per-step CTE penalty  (≈ 0 on path, −1 at edge)
  −|Δsteer| × 0.5       steering-rate smoothness penalty
  −10.0                 terminal penalty if |cte| > MAX_CTE_TERMINATE
  +20.0                 terminal bonus for reaching the destination

Episode termination
-------------------
  • |CTE| > MAX_CTE_TERMINATE  (tractor left the path)
  • Tractor reached the destination
  • Step count > MAX_STEPS
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.tractor      import Tractor
from src.path         import Path
from src.pure_pursuit import PurePursuit

# ── Episode config ────────────────────────────────────────────────────────────
MAX_STEPS         = 4_000    # ≈ 67 s at 60 fps — plenty for one full lap
MAX_CTE_TERMINATE = 55.0     # px — wider than dirt strip edge (DIRT_HALF_W=22)
                              # so partial recovery attempts don't truncate early
THROTTLE_RL       = 0.78     # fixed throttle — matches expert, agent only steers
SIM_DT            = 1 / 60   # fixed physics timestep (seconds)
LOOKAHEAD_PX      = 80.0     # must match Phase 2 / 3

# Normalisation constants
CTE_NORM_SCALE    = 60.0
SPEED_NORM_SCALE  = Tractor.MAX_SPEED

HOME_X, HOME_Y    = 640.0, 530.0
HOME_HEADING      = -math.pi / 2    # pointing north


class TractorEnv(gym.Env):
    """
    Gymnasium-compatible environment for the autonomous tractor.

    Observation: 6-dim float32 vector (see module docstring).
    Action:      1-dim continuous  steer_input ∈ [-1, 1].
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        self.render_mode = render_mode
        self._path       = Path()
        self._pp         = PurePursuit(lookahead=LOOKAHEAD_PX)
        self._tractor    = None   # created in reset()
        self._n_wps      = len(self._path.waypoints)

        # Pygame surface — only created if render_mode == "human"
        self._screen     = None
        self._renderer   = None
        self._clock      = None

        # ── Spaces ────────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low  = np.array([-1, -1, 0, -1, -1, 0], dtype=np.float32),
            high = np.array([ 1,  1, 1,  1,  1, 1], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low  = np.array([-1.0], dtype=np.float32),
            high = np.array([ 1.0], dtype=np.float32),
        )

        # ── Step tracking ─────────────────────────────────────────────────────
        self._step_count   = 0
        self._prev_steer   = 0.0    # for smoothness penalty

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._tractor    = Tractor(x=HOME_X, y=HOME_Y, heading=HOME_HEADING)
        self._step_count = 0
        self._prev_steer = 0.0

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        steer_input = float(np.clip(action[0], -1.0, 1.0))

        # ── Physics ───────────────────────────────────────────────────────────
        self._tractor.update(steer_input, THROTTLE_RL, SIM_DT)
        self._step_count += 1

        # ── Metrics ───────────────────────────────────────────────────────────
        cte      = self._path.cross_track_error(self._tractor.x, self._tractor.y)
        arrived  = self._path.reached_destination(self._tractor.x, self._tractor.y)
        off_path = abs(cte) > MAX_CTE_TERMINATE

        # ── Reward ────────────────────────────────────────────────────────────
        reward  =  1.0                                    # survival bonus
        reward -= abs(cte) / CTE_NORM_SCALE               # CTE penalty
        reward -= abs(steer_input - self._prev_steer) * 0.5  # smoothness

        if off_path:
            reward -= 10.0
        if arrived:
            reward += 20.0

        self._prev_steer = steer_input

        # ── Termination ───────────────────────────────────────────────────────
        terminated = arrived or off_path
        truncated  = self._step_count >= MAX_STEPS

        obs  = self._get_obs()
        info = {
            "cte":     cte,
            "arrived": arrived,
            "steps":   self._step_count,
        }

        # ── Optional render ───────────────────────────────────────────────────
        if self.render_mode == "human":
            self._render_frame(cte)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._render_frame(
                self._path.cross_track_error(self._tractor.x, self._tractor.y))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = self._renderer = self._clock = None

    # ── Private ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        t       = self._tractor
        cte     = self._path.cross_track_error(t.x, t.y)
        pp_res  = self._pp.compute(t.x, t.y, t.heading, self._path)
        h_err   = self._pp.heading_error(t.x, t.y, t.heading, self._path)
        wp_idx  = self._path.nearest_waypoint_index(t.x, t.y)
        progress = wp_idx / max(self._n_wps - 1, 1)

        return np.array([
            np.clip(cte / CTE_NORM_SCALE,          -1.0, 1.0),
            np.clip(h_err / math.pi,               -1.0, 1.0),
            np.clip(t.speed / SPEED_NORM_SCALE,     0.0, 1.0),
            np.clip(pp_res.alpha_rad / math.pi,    -1.0, 1.0),
            np.clip(t.steer_angle / Tractor.MAX_STEER, -1.0, 1.0),
            float(progress),
        ], dtype=np.float32)

    def _render_frame(self, cte: float) -> None:
        """Lazy-initialise Pygame and render one frame."""
        import pygame
        from src.renderer import Renderer

        if self._screen is None:
            pygame.init()
            self._screen   = pygame.display.set_mode((900, 580))
            self._renderer = Renderer(self._screen)
            self._clock    = pygame.time.Clock()
            pygame.display.set_caption("Autonomous Tractor — Phase 4 (SAC)")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._renderer.draw_frame(self._path, self._tractor, cte)
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])