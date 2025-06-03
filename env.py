# =========================
# env.py – HumanoidCustomEnv (anti‑hopping, symmetric gait)
# =========================
"""Custom wrapper um *Humanoid‑v5* mit stärkerer Bestrafung von einbeinigen
Hüpfern und expliziter *Gait‑Symmetrie*‑Förderung.

Änderungen gegen die vorige Version
────────────────────────────────────
* **AIRBORNE_PENALTY** +2 → -4.0  (härterer Malus pro Step ohne Bodenkontakt)
* **ONE_LEG_PENALTY**  neu: -1.5 / Step nach >6 Single‑leg‑Steps
* **symmetry_bonus** Skala ↑ 0.2
* Neues Feld info["one_leg_penalty"]
"""
from __future__ import annotations

import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from scipy.spatial.transform import Rotation

# ─── coefficients ──────────────────────────────────────────
FRAME_SKIP              = 4
ENERGY_COST             = 0.01
AIRBORNE_PENALTY        = -4.0        # pro Step ohne Bodenkontakt (war ‑2.0)
ONE_LEG_PENALTY         = -1.5        # sobald ein Bein > MAX_ONE_LEG_STEPS trägt
MAX_ONE_LEG_STEPS       = 6
VERT_VEL_PEN_SCALE      = 0.5
HEIGHT_PENALTY_SCALE    = 6.0
HEIGHT_THR, MAX_HEIGHT  = 1.25, 1.55
MAX_AIR_STEPS           = 10          # etwas strenger


class HumanoidCustomEnv(HumanoidEnv):
    def __init__(self, render_mode: str | None = None, **kwargs):
        kwargs.setdefault("frame_skip", FRAME_SKIP)
        kwargs.setdefault("render_mode", render_mode)
        kwargs.setdefault("terminate_when_unhealthy", False)
        kwargs.setdefault("healthy_z_range", (0.8, MAX_HEIGHT))
        super().__init__(**kwargs)

        self.id_root   = self.model.body("torso").id
        self._foot_ids = [self.model.body("left_foot").id,
                          self.model.body("right_foot").id]

        self.prev_action = np.zeros(self.model.nu)
        self.fall_steps = self.air_steps = 0
        self.one_leg_steps = 0
        self.last_stance: tuple[bool,bool] | None = None
        self.MAX_FALL_STEPS = 50



    # ───────────────────── helpers ────────────────────────
    def _com(self):    return np.average(self.data.xipos, axis=0, weights=self.model.body_mass)

    def _upright(self) -> float:
        q = self.data.qpos[3:7]; q /= np.linalg.norm(q)
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, 1])[2]

    def _sym_pen(self):
        l, r = self.data.qpos[7:13], self.data.qpos[13:19]
        return -np.square(l - r).sum()

    def _heading(self):
        return np.clip(self.data.geom_xmat[self.id_root, :3][0], 0, 1)

    def _feet_contact(self):
        f = self.data.cfrc_ext
        return tuple(np.linalg.norm(f[bid, :3]) > 1e-3 for bid in self._foot_ids)

    # ───────────────────── reset/step ─────────────────────
    def reset(self, **kwargs):
        self.prev_action[:] = 0
        self.fall_steps = self.air_steps = self.one_leg_steps = 0
        self.last_stance = None
        qpos, qvel = self.init_qpos.copy(), self.init_qvel.copy()
        qpos[3:7] = [1, 0, 0, 0]
        self.set_state(qpos, qvel)
        return super().reset(**kwargs)

    def step(self, a: np.ndarray):
        obs, _, terminated, truncated, info = super().step(a)

        com_z, v_z = self._com()[2], self.data.qvel[2]
        upright    = np.clip(self._upright(), 0, 1) * 2

        # --- Kontaktlogik ------------------------------------------------
        l_c, r_c = self._feet_contact()
        stance = (l_c, r_c)
        if (l_c ^ r_c):  # genau ein Bein am Boden
            if stance == self.last_stance:
                self.one_leg_steps += 1
            else:
                self.one_leg_steps = 1
        else:
            self.one_leg_steps = 0
        self.last_stance = stance

        # --- Rewards / Penalties ----------------------------------------
        alive       = 1.0
        height_pen  = -HEIGHT_PENALTY_SCALE * max(0, com_z - HEIGHT_THR)
        vert_pen    = -VERT_VEL_PEN_SCALE * abs(v_z)
        speed_rew   = np.clip(self.data.qvel[0], 0, 3)
        energy_pen  = -ENERGY_COST * np.square(a).sum()
        sym_bonus   = 0.2 * self._sym_pen()
        head_bonus  = self._heading() * 0.5
        smooth_pen  = -0.05 * np.square(a - self.prev_action).sum()

        # --- Kniehebe-Bonus für große Schritte (menschlicher Gang) ------
        left_knee = abs(self.data.qpos[15])
        right_knee = abs(self.data.qpos[21])
        knee_lift_bonus = 0.5 * (left_knee + right_knee)

        # --- airborne & hopping penalty ----------------------------------
        if not (l_c or r_c):
            self.air_steps += 1
            air_pen = AIRBORNE_PENALTY
        else:
            self.air_steps = 0
            air_pen = 0.0

        hop_pen = ONE_LEG_PENALTY if self.one_leg_steps > MAX_ONE_LEG_STEPS else 0.0

        # --- Finaler Reward ----------------------------------------------
        reward = (
            alive + height_pen + upright + speed_rew + energy_pen + sym_bonus +
            head_bonus + smooth_pen + vert_pen + air_pen + hop_pen +
            knee_lift_bonus
        )

        # --- Fallen & Terminierung ---------------------------------------
        fallen = (
            (com_z < 0.7) or (upright < 0.2) or (com_z > MAX_HEIGHT) or
            (self.air_steps > MAX_AIR_STEPS)
        )
        self.fall_steps = self.fall_steps + 1 if fallen else 0
        terminated = terminated or (self.fall_steps > self.MAX_FALL_STEPS)

        # --- housekeeping & info -----------------------------------------
        self.prev_action = a.copy()
        info.update(
            dict(
                height_penalty=height_pen,
                airborne_penalty=air_pen,
                vert_vel_penalty=vert_pen,
                upright_bonus=upright,
                speed_reward=speed_rew,
                energy_penalty=energy_pen,
                symmetry_bonus=sym_bonus,
                heading_bonus=head_bonus,
                smoothness_penalty=smooth_pen,
                air_steps=self.air_steps,
                hop_penalty=hop_pen,
                one_leg_steps=self.one_leg_steps,
                knee_lift_bonus=knee_lift_bonus,
            )
        )
        return obs, reward, terminated, truncated, info



def make_env(seed: int | None = None, render_mode: str | None = None):
    env = HumanoidCustomEnv(render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env  


