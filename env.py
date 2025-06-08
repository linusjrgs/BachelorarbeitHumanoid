"""
Diese Umgebung ist eine eigene Erweiterung der Humanoid-v5-Umgebung aus Gymnasium/Mujoco.

Hintergrund:
In der Standardumgebung zeigen viele trainierte Agenten schnell unrealistische Bewegungen, 
zum Beispiel dauerhaftes Hüpfen auf einem Bein oder stark asymmetrische Gangmuster. 
Gerade für Anwendungen in Robotik oder in der biomechanischen Forschung ist solches Verhalten 
wenig sinnvoll.

Ziel dieser Anpassung:
Das Ziel dieser modifizierten Umgebung ist es, ein möglichst symmetrisches und stabil wirkendes 
Gangverhalten zu fördern. Insbesondere wird das Hüpfen auf einem Bein explizit bestraft, während 
symmetrisches Gehen und kontrollierte, kontinuierliche Bewegungen belohnt werden.

Was ist neu:
- AIRBORNE_PENALTY wurde erhöht. Damit wird "komplett in der Luft sein" strenger bestraft.
- ONE_LEG_PENALTY wurde neu eingeführt. Dadurch wird das zu lange Gehen nur auf einem Bein gezielt unterbunden.
- Der Symmetriebonus wurde verstärkt, um beidseitiges, gleichmäßiges Gangverhalten noch attraktiver zu machen.
- Zusätzlich wurde ein neues Feld im Info-Objekt ergänzt, das den aktuellen one_leg_penalty mit ausgibt. 
  Dies erleichtert spätere Auswertungen und das Debugging im Training.

Insgesamt bietet diese Umgebung also bessere Voraussetzungen für die Entwicklung von realitätsnahen 
Gangmustern und stabilen Bewegungsabläufen.
"""


from __future__ import annotations

import numpy as np
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from scipy.spatial.transform import Rotation

# --- Parameter für Belohnungen und Bestrafungen ---
FRAME_SKIP              = 4
ENERGY_COST             = 0.01
AIRBORNE_PENALTY        = -4.0        # Strafe pro Schritt ohne Bodenkontakt
ONE_LEG_PENALTY         = -1.5        # Strafe, wenn nur ein Bein zu lange Bodenkontakt hat
MAX_ONE_LEG_STEPS       = 6           # Nach wie vielen Einbein-Schritten bestraft wird
VERT_VEL_PEN_SCALE      = 0.5         # Bestrafung für vertikale Geschwindigkeit
HEIGHT_PENALTY_SCALE    = 6.0         # Bestrafung bei zu großer Höhe
HEIGHT_THR, MAX_HEIGHT  = 1.25, 1.55  # Schwellenwerte für "zu hoch"
MAX_AIR_STEPS           = 10          # maximale Schritte komplett in der Luft

class HumanoidCustomEnv(HumanoidEnv):
    def __init__(self, render_mode: str | None = None, **kwargs):
        # Einige Standardwerte setzen
        kwargs.setdefault("frame_skip", FRAME_SKIP)
        kwargs.setdefault("render_mode", render_mode)
        kwargs.setdefault("terminate_when_unhealthy", False)
        kwargs.setdefault("healthy_z_range", (0.8, MAX_HEIGHT))
        super().__init__(**kwargs)

        # IDs für Torso und Füße merken, um z.B. Bodenkontakt zu überprüfen
        self.id_root   = self.model.body("torso").id
        self._foot_ids = [
            self.model.body("left_foot").id,
            self.model.body("right_foot").id
        ]

        # Zusätzliche Zustandsvariablen für Air-Time, Hüpf-Zustand etc.
        self.prev_action = np.zeros(self.model.nu)
        self.fall_steps = self.air_steps = 0
        self.one_leg_steps = 0
        self.last_stance: tuple[bool,bool] | None = None
        self.MAX_FALL_STEPS = 50

    # --- Hilfsfunktionen ---

    def _com(self):
        # Schwerpunkt (Center of Mass) berechnen
        return np.average(self.data.xipos, axis=0, weights=self.model.body_mass)

    def _upright(self) -> float:
        # Neigung aus Quaternion ableiten und z-Komponente des nach-oben-Vektors zur Bewertung nutzen
        q = self.data.qpos[3:7]; q /= np.linalg.norm(q)
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, 1])[2]

    def _sym_pen(self):
        # Differenz zwischen linker und rechter Beinhaltung als Symmetriestrafe (negativ)
        l, r = self.data.qpos[7:13], self.data.qpos[13:19]
        return -np.square(l - r).sum()

    def _heading(self):
        # Ausrichtung des Torsos: 1.0 bedeutet ideal nach vorne
        return np.clip(self.data.geom_xmat[self.id_root, :3][0], 0, 1)

    def _feet_contact(self):
        # Bodenkontakt beider Füße über externen Kraftsensor abfragen
        f = self.data.cfrc_ext
        return tuple(np.linalg.norm(f[bid, :3]) > 1e-3 for bid in self._foot_ids)

    # --- Reset ---

    def reset(self, **kwargs):
        # interner Zustand zurücksetzen
        self.prev_action[:] = 0
        self.fall_steps = self.air_steps = self.one_leg_steps = 0
        self.last_stance = None
        qpos, qvel = self.init_qpos.copy(), self.init_qvel.copy()
        qpos[3:7] = [1, 0, 0, 0]  # Orientierung zurücksetzen
        self.set_state(qpos, qvel)
        return super().reset(**kwargs)

    # --- Schrittweise Simulation ---

    def step(self, a: np.ndarray):
        # einen Simulationsschritt ausführen
        obs, _, terminated, truncated, info = super().step(a)

        # Zustandswerte berechnen
        com_z = self._com()[2]              # Höhe Schwerpunkt
        v_z = self.data.qvel[2]             # vertikale Geschwindigkeit
        upright = np.clip(self._upright(), 0, 1) * 2

        # --- Kontaktanalyse: erkennt einbeinige Gangarten ---
        l_c, r_c = self._feet_contact()
        stance = (l_c, r_c)
        if (l_c ^ r_c):  # nur ein Bein am Boden
            if stance == self.last_stance:
                self.one_leg_steps += 1
            else:
                self.one_leg_steps = 1
        else:
            self.one_leg_steps = 0
        self.last_stance = stance

        # --- Belohnungen und Strafen zusammensetzen ---
        alive = 1.0
        height_pen = -HEIGHT_PENALTY_SCALE * max(0, com_z - HEIGHT_THR)
        vert_pen = -VERT_VEL_PEN_SCALE * abs(v_z)
        speed_rew = np.clip(self.data.qvel[0], 0, 3)  # nur Vorwärtsgeschw.
        energy_pen = -ENERGY_COST * np.square(a).sum()
        sym_bonus = 0.2 * self._sym_pen()
        head_bonus = self._heading() * 0.5
        smooth_pen = -0.05 * np.square(a - self.prev_action).sum()

        # --- Kniebonus für "menschliches" Gehen ---
        left_knee = abs(self.data.qpos[15])
        right_knee = abs(self.data.qpos[21])
        knee_lift_bonus = 0.5 * (left_knee + right_knee)

        # --- Luftzeitbestrafung ---
        if not (l_c or r_c):
            self.air_steps += 1
            air_pen = AIRBORNE_PENALTY
        else:
            self.air_steps = 0
            air_pen = 0.0

        hop_pen = ONE_LEG_PENALTY if self.one_leg_steps > MAX_ONE_LEG_STEPS else 0.0

        # --- Gesamtbelohnung berechnen ---
        reward = (
            alive + height_pen + upright + speed_rew + energy_pen + sym_bonus +
            head_bonus + smooth_pen + vert_pen + air_pen + hop_pen +
            knee_lift_bonus
        )

        # --- Terminationslogik ---
        fallen = (
            (com_z < 0.7) or (upright < 0.2) or (com_z > MAX_HEIGHT) or
            (self.air_steps > MAX_AIR_STEPS)
        )
        self.fall_steps = self.fall_steps + 1 if fallen else 0
        terminated = terminated or (self.fall_steps > self.MAX_FALL_STEPS)

        # --- Infos für Logging / Analyse ---
        self.prev_action = a.copy()
        info.update(dict(
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
        ))
        return obs, reward, terminated, truncated, info


# Factory-Funktion zum einfachen Erzeugen der Umgebung

def make_env(seed: int | None = None, render_mode: str | None = None):
    env = HumanoidCustomEnv(render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env
