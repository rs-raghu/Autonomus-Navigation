"""
phase4_rl.py  —  Phase 4: SAC Reinforcement Learning

Trains a Soft Actor-Critic (SAC) agent on the tractor simulation.
The actor is warm-started from the Phase 3 imitation learning checkpoint
so training begins from a reasonable policy — not random noise.

Run:
    python phase4_rl.py              # trains from scratch (uses IL warm-start)
    python phase4_rl.py --resume     # resumes from models/sac_latest.zip

Outputs
-------
  models/sac_best.zip        best model checkpoint (by mean episode reward)
  models/sac_latest.zip      checkpoint at every SAVE_EVERY steps
  reports/rl_training.csv    per-episode stats (reward, CTE, steps, arrived)
  reports/rl_curve.png       training reward curve (saved on completion)

Controls (render window, only when --render flag is used)
---------
  ESC / close window    stop training gracefully and save
"""

import argparse
import os
import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env   import DummyVecEnv

from src.gym_env import TractorEnv

# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_STEPS   = 300_000    # total environment steps
SAVE_EVERY    = 25_000     # save checkpoint every N steps
EVAL_EVERY    = 5_000      # evaluate (print stats) every N steps
LEARNING_RATE = 3e-4
BUFFER_SIZE   = 100_000
BATCH_SIZE    = 256
GAMMA         = 0.99
TAU           = 0.005      # soft update coefficient
ENT_COEF      = "auto"     # SAC entropy coefficient (auto-tune)
TRAIN_FREQ    = 1
GRADIENT_STEPS= 1

IL_MODEL_PATH = "model_il.pt"
MODEL_DIR     = "models"
REPORT_DIR    = "reports"
SEED          = 42

# ── IL warm-start ─────────────────────────────────────────────────────────────

def load_il_weights(sac_model: SAC, il_path: str) -> bool:
    """
    Copy the IL MLP weights into the SAC actor network.

    The IL model is:  Linear(4→64) Tanh Linear(64→64) Tanh Linear(64→32) Tanh Linear(32→1) Tanh
    SB3 SAC actor is: Linear(6→256) ReLU Linear(256→256) ReLU → mean + log_std heads

    The architectures don't match directly (different input dims, widths, activations).
    Instead of forcing a weight copy, we use the IL model to generate a pre-training
    dataset and do a brief supervised warm-up on the SAC actor.

    This is the correct approach for warm-starting from a different architecture:
    distil the IL policy into the SAC actor by behaviour cloning on rollouts.
    """
    if not os.path.exists(il_path):
        print(f"[WarmStart] {il_path} not found — training from random init.")
        return False

    print(f"[WarmStart] Loading IL model from {il_path} …")

    try:
        ckpt = torch.load(il_path, map_location="cpu", weights_only=False)

        # Rebuild IL model
        il_net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 32), nn.Tanh(),
            nn.Linear(32, 1),  nn.Tanh(),
        )
        # Rebuild state dict mapping from SteeringMLP.net.*
        il_state = {k.replace("net.", ""): v
                    for k, v in ckpt["model_state"].items()}
        il_net.load_state_dict(il_state)
        il_net.eval()

        norm_mean = torch.tensor(ckpt["norm_stats"]["mean"], dtype=torch.float32)
        norm_std  = torch.tensor(ckpt["norm_stats"]["std"],  dtype=torch.float32)

        print("[WarmStart] Running behaviour cloning warm-up (2 000 steps) …")

        # Generate IL demonstrations by rolling out a fresh env
        env = TractorEnv()
        obs, _ = env.reset(seed=SEED)

        bc_X, bc_y = [], []

        from src.pure_pursuit import PurePursuit
        from src.path         import Path
        pp   = PurePursuit()
        path = Path()

        for _ in range(2_000):
            # The SAC obs has 6 dims; IL uses [cte, heading_err, speed, alpha] (dims 0,1,2,3)
            il_input = torch.tensor(obs[:4], dtype=torch.float32)
            # IL was trained on un-normalised features; re-scale
            # obs[:4] are already normalised: cte/60, h_err/π, speed/max_speed, alpha/π
            # Recover raw values
            raw = il_input * torch.tensor([60.0, math.pi, 130.0, math.pi])
            # Apply IL normalisation
            x_norm = (raw - norm_mean) / norm_std
            with torch.no_grad():
                il_steer = float(il_net(x_norm.unsqueeze(0)).squeeze())

            bc_X.append(obs.copy())
            bc_y.append([il_steer])

            obs, _, terminated, truncated, _ = env.step(np.array([il_steer]))
            if terminated or truncated:
                obs, _ = env.reset()

        env.close()

        # Brief BC warm-up on SAC actor
        X_t = torch.tensor(np.array(bc_X), dtype=torch.float32)
        y_t = torch.tensor(np.array(bc_y), dtype=torch.float32)

        actor      = sac_model.actor
        actor_opt  = torch.optim.Adam(actor.parameters(), lr=1e-3)
        criterion  = nn.MSELoss()

        actor.train()
        for ep in range(20):
            actor_opt.zero_grad()
            # SB3 actor forward: get deterministic action
            action_dist = actor.get_action_dist_params(X_t)
            mean_act    = action_dist[0]           # (B, 1)
            loss        = criterion(mean_act, y_t)
            loss.backward()
            actor_opt.step()
            if (ep + 1) % 5 == 0:
                print(f"  BC epoch {ep+1:>2}/20  loss={loss.item():.5f}")

        actor.eval()
        print("[WarmStart] Behaviour cloning warm-up complete.\n")
        return True

    except Exception as e:
        print(f"[WarmStart] Warning: {e} — falling back to random init.")
        return False


# ── Training callback ─────────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """
    Logs per-episode stats and saves checkpoints.

    Episode stats are extracted by watching for `done` flags in the
    SB3 rollout buffer's info dicts.
    """

    FIELDS = ["episode", "steps", "reward", "mean_abs_cte",
              "episode_len", "arrived", "elapsed_s"]

    def __init__(self, save_every: int, eval_every: int,
                 report_dir: str, model_dir: str):
        super().__init__(verbose=0)
        self._save_every  = save_every
        self._eval_every  = eval_every
        self._report_dir  = report_dir
        self._model_dir   = model_dir
        self._episode     = 0
        self._best_reward = -float("inf")
        self._ep_rewards: list[float] = []
        self._t0          = time.time()

        os.makedirs(model_dir,  exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        self._csv_path = os.path.join(report_dir, "rl_training.csv")
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def _on_step(self) -> bool:
        # SB3 puts episode stats into infos["episode"] at the end of each episode
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is None:
                continue

            self._episode  += 1
            ep_reward       = ep_info["r"]
            ep_len          = ep_info["l"]
            arrived         = info.get("arrived", False)
            cte             = abs(info.get("cte", 0.0))

            self._ep_rewards.append(ep_reward)

            row = {
                "episode":      self._episode,
                "steps":        self.num_timesteps,
                "reward":       round(ep_reward, 3),
                "mean_abs_cte": round(cte, 2),
                "episode_len":  ep_len,
                "arrived":      int(arrived),
                "elapsed_s":    round(time.time() - self._t0, 1),
            }
            with open(self._csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)

            if self._episode % 10 == 0:
                recent = self._ep_rewards[-20:]
                mean_r = sum(recent) / len(recent)
                print(f"  ep {self._episode:>5}  "
                      f"steps {self.num_timesteps:>7,}  "
                      f"reward {ep_reward:>+8.1f}  "
                      f"mean20 {mean_r:>+7.1f}  "
                      f"len {ep_len:>5}  "
                      f"arrived={'Y' if arrived else 'N'}")

            # Save best model
            recent_mean = (sum(self._ep_rewards[-20:]) /
                           len(self._ep_rewards[-20:]))
            if recent_mean > self._best_reward and self._episode >= 20:
                self._best_reward = recent_mean
                self.model.save(os.path.join(self._model_dir, "sac_best"))
                print(f"  ★ New best mean-20 reward: {recent_mean:+.2f} "
                      f"→ models/sac_best.zip")

        # Save periodic checkpoint
        if self.num_timesteps % self._save_every == 0:
            self.model.save(os.path.join(self._model_dir, "sac_latest"))
            print(f"\n  [Checkpoint] {self.num_timesteps:,} steps "
                  f"→ models/sac_latest.zip\n")

        return True

    def _on_training_end(self) -> None:
        # Save final curve plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            rewards = self._ep_rewards
            window  = min(50, len(rewards))
            moving  = [
                sum(rewards[max(0, i-window):i+1]) /
                len(rewards[max(0, i-window):i+1])
                for i in range(len(rewards))
            ]

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(rewards, alpha=0.3, linewidth=0.8, label="Episode reward")
            ax.plot(moving,  linewidth=1.8, label=f"Moving avg ({window} eps)")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total reward")
            ax.set_title("Phase 4 — SAC training curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            out = os.path.join(self._report_dir, "rl_curve.png")
            fig.tight_layout()
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"\n  Training curve → {out}")
        except Exception as e:
            print(f"  [Plot] Could not save curve: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true",
                   help="Resume from models/sac_latest.zip")
    p.add_argument("--render", action="store_true",
                   help="Show Pygame window during training (slows training)")
    p.add_argument("--steps", type=int, default=TOTAL_STEPS,
                   help=f"Total training steps (default {TOTAL_STEPS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    render_mode = "human" if args.render else None

    print("=" * 60)
    print("  Phase 4 — SAC Reinforcement Learning")
    print("=" * 60)
    print(f"  Steps    : {args.steps:,}")
    print(f"  Render   : {args.render}")
    print(f"  Resume   : {args.resume}")
    print()

    # ── Environment ───────────────────────────────────────────────────────────
    def make_env():
        return TractorEnv(render_mode=render_mode)

    vec_env = DummyVecEnv([make_env])

    # ── Model ─────────────────────────────────────────────────────────────────
    resume_path = os.path.join(MODEL_DIR, "sac_latest.zip")

    if args.resume and os.path.exists(resume_path):
        print(f"[Phase 4] Resuming from {resume_path} …")
        model = SAC.load(resume_path, env=vec_env)
    else:
        print("[Phase 4] Building new SAC model …")
        model = SAC(
            policy       = "MlpPolicy",
            env          = vec_env,
            learning_rate = LEARNING_RATE,
            buffer_size   = BUFFER_SIZE,
            batch_size    = BATCH_SIZE,
            gamma         = GAMMA,
            tau           = TAU,
            ent_coef      = ENT_COEF,
            train_freq    = TRAIN_FREQ,
            gradient_steps= GRADIENT_STEPS,
            policy_kwargs = {"net_arch": [256, 256]},
            seed          = SEED,
            verbose       = 0,
        )

        # IL warm-start (behaviour cloning into SAC actor)
        load_il_weights(model, IL_MODEL_PATH)

    # ── Callback ──────────────────────────────────────────────────────────────
    callback = TrainingCallback(
        save_every  = SAVE_EVERY,
        eval_every  = EVAL_EVERY,
        report_dir  = REPORT_DIR,
        model_dir   = MODEL_DIR,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"[Phase 4] Training for {args.steps:,} steps …\n")
    t0 = time.time()

    model.learn(
        total_timesteps   = args.steps,
        callback          = callback,
        reset_num_timesteps = not args.resume,
        progress_bar      = True,
    )

    elapsed = time.time() - t0
    print(f"\n[Phase 4] Training complete in {elapsed/60:.1f} min.")

    # Final save
    final_path = os.path.join(MODEL_DIR, "sac_final.zip")
    model.save(final_path)
    print(f"[Phase 4] Final model → {final_path}")


if __name__ == "__main__":
    main()