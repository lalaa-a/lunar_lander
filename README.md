# Deep Q‑Learning for Lunar Lander

This project implements a **Deep Q‑Learning (DQN)** agent to solve the [LunarLander-v2](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment from OpenAI Gym. The agent learns to control a lunar module to land safely between two flags by adjusting its thrusters. The implementation uses a neural network as a function approximator for the Q‑function, along with experience replay and a target network for stable training.

---

## Overview

- **Environment:** LunarLander-v2 – a 2D physics simulation where a spacecraft must land on a landing pad.
- **State space:** 8 continuous features (position, velocity, angle, angular velocity, leg contact).
- **Action space:** 4 discrete actions – do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
- **Goal:** Achieve an average reward of ≥ 200 over 100 consecutive episodes.

The DQN agent uses:
- A **neural network** with two hidden layers (64 units each, ReLU activation) to approximate Q‑values.
- **Experience replay** to break correlation between consecutive samples.
- A **target network** with periodic updates to stabilise training.
- **Epsilon‑greedy** exploration with exponential decay.

---

## Features

- **DQN implementation** using TensorFlow/Keras.
- **Experience replay buffer** (capacity = 100,000) to store and sample transitions.
- **Target network** updated every 4 steps to compute stable TD targets.
- **Adaptive epsilon** decay from 1.0 to 0.01.
- **Training progress** displayed with running average reward.
- **Model saving** when the environment is solved (avg ≥ 200).
- **Video generation** of a trained agent’s performance (MP4).
- **Plotting** of the total reward per episode.

---

## Dependencies

- Python 3.7+
- [OpenAI Gym](https://github.com/openai/gym) (with Box2D)
- NumPy
- TensorFlow 2.x
- Matplotlib
- PIL (Pillow)
- pyvirtualdisplay (for headless rendering, optional)
- The custom `utils.py` module (provided separately)

Install required packages:

```bash
pip install gym[box2d] numpy tensorflow matplotlib pillow pyvirtualdisplay
