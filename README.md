# PPO Agent for Humanoid Locomotion

**Bachelor Thesis · Hochschule München · 2025**  
**Focus:** Reinforcement Learning · PPO · Reward Engineering · Robot Locomotion · MuJoCo

This project was my introduction to **Reinforcement Learning** and investigated how a humanoid robot can learn locomotion using **Proximal Policy Optimization (PPO)**.

The work was conducted as part of my Bachelor's thesis at **Hochschule München** using the `Humanoid-v5` environment from **Gymnasium** and **MuJoCo**.

The main focus was on investigating how **reward design influences the behavior learned by the agent**.

---

## Overview

The original `Humanoid-v5` environment was extended with additional reward components to investigate:

- forward locomotion
- movement symmetry
- reduced hopping and jumping
- body stabilization

A custom PPO agent with separate policy and value networks was implemented and evaluated.

---

## Result

The trained agent successfully learned a behavior that enabled **forward locomotion**.

However, the resulting policy did **not** develop the natural human-like walking pattern that was originally intended.

Instead, the agent converged to a short **stepping / "tippling" behavior**.

![Learned Humanoid locomotion](./Kurzesvideo.gif)

This result demonstrated how strongly the learned behavior depends on the design and weighting of the reward function. The agent found a locally effective movement strategy that did not fully correspond to the intended behavior.

---

## Technical Approach

### Reinforcement Learning

The agent uses **Proximal Policy Optimization (PPO)** with separate neural networks for:

- **Policy Network** – predicts the actions of the humanoid
- **Value Network** – estimates the expected future return

The PPO implementation includes:

- clipped policy updates
- advantage estimation
- entropy regularization
- value loss
- gradient clipping
- observation normalization

### Custom Humanoid Environment

The original `Humanoid-v5` environment was extended with additional reward components:

| Reward component | Purpose |
|---|---|
| Forward movement | Encourage locomotion |
| Symmetry | Encourage more balanced movement |
| Anti-hopping | Penalize undesired jumping / hopping |
| Body stabilization | Encourage a more stable posture |

---

## Key Takeaways

This project was my first practical experience with **Reinforcement Learning**. At the beginning of the project, my practical RL knowledge was mainly based on theoretical concepts and online lectures, so implementing and training a complete PPO agent was a significant learning process.

One of the main challenges was **reward engineering**. I learned that designing a reward function for complex behaviors such as humanoid locomotion is difficult, as different reward components can interact in unexpected ways and lead to behaviors that differ from the original intention.

Another major limitation was **computational resources**. Training the humanoid agent required substantial computational time, and some experiments took several days to complete. Limited hardware therefore restricted the number of experiments and iterations that could realistically be performed.

The project gave me practical experience with:

- Reinforcement Learning and PPO
- Reward engineering and iterative experimentation
- Simulated robot control
- Training and evaluating neural networks
- Understanding the impact of computational resources on machine learning experiments

---

## Project Structure

```text
BachelorarbeitHumanoid/
├── README.md
├── main.py
├── env.py
├── ppo.py
├── mlp.py
├── utils.py
├── requirements.txt
│
└── Kurzesvideo.gif
