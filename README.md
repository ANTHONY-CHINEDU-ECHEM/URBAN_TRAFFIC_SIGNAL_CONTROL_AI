# Urban Traffic Signal Control AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Reference Implementation](https://img.shields.io/badge/status-reference%20implementation-orange.svg)]()

> A shared-policy tabular Q-learning controller for 12 corridor intersections—featuring a rigorous 50-seed Monte Carlo shadow-mode evaluation and an honest disclosure of performance relative to established traffic heuristics.

<img width="1748" height="899" alt="URBAN TRAFFIC CONTROL AI" src="https://github.com/user-attachments/assets/867c9209-0651-43ec-8827-37f37ec54198" />


## 📋 Executive Summary

This project implements a **multi-agent reinforcement learning (MARL)** system designed to coordinate signal phase timing across an arterial corridor of 12 sequential intersections. 

Rather than presenting a cherry-picked demonstration, this repository embraces rigorous evaluation discipline:
* **The Easy Win:** The RL agent cuts total vehicle delay by **-42.6%** compared to traditional fixed-time signal plans (surpassing an 18% target by 2.4x).
* **The Honest Reality:** Against a hand-tuned **max-pressure heuristic**, the tabular RL agent performs **40.5% worse** on total delay. 
* **The Takeaway:** A negative result reported honestly is more valuable than a misleading demo. This repository details *why* the information bottleneck of tabular state discretization limits performance, and lays out a concrete engineering roadmap (Deep RL/PPO) to close the gap.


Key Findings:

- The Fixed-Time Comparison: RL successfully reduces total delay by 42.6% over static clocks.

- The Heuristic Comparison: The lightweight, zero-training max-pressure heuristic outperforms the RL agent by 40.5% proving that simple reactive queue-balancing remains exceptionally strong for tabular control scales.

- The Equity Trade-Off: RL heavily prioritises main arterial progression, causing side-street max wait times to spike to 4.9 minutes (worse than the demand-blind fixed-time baseline)


## 🗂️ Repository Structure


URBAN_TRAFFIC_SIGNAL_CONTROL_AI/
│

├── demand_profiles.py          # AM/PM-peaked arrival-rate functions (veh/min)

├── baselines.py                # Fixed-time and max-pressure baseline controllers

├── corridor_qtable.npz         # Learned Q-table (138 states from 400 training episodes)

├── training_curve.json         # Per-episode reward tracking data

├── shadow_mode_report.json     # 50-seed Monte Carlo benchmark output

├── sample_detector_log.csv     # 10,368 rows of loop-detector telemetry

├── TRAFFIC OPTIMIZATION DASHBOARD.py # Interactive decision-support interface

├── EXTENDING.md                # Architectural upgrade path (SUMO/TraCI & Deep RL)

└── LICENSE


GETTING STARTED

# Clone the repository
git clone [https://github.com/ANTHONY-CHINEDU-ECHEM/URBAN_TRAFFIC_SIGNAL_CONTROL_AI.git](https://github.com/ANTHONY-CHINEDU-ECHEM/URBAN_TRAFFIC_SIGNAL_CONTROL_AI.git)
cd URBAN_TRAFFIC_SIGNAL_CONTROL_AI

# Run the performance evaluation dashboard
python "TRAFFIC OPTIMIZATION DASHBOARD.py"


Corridor & Data ProfileScale: 12 interacting intersections along a shared arterial corridor, with 4 approaches (North}/{South}/{East}/(West}$) per intersection.


<img width="1280" height="832" alt="TRAFFIC OPTIMIZATION DASHBOARD" src="https://github.com/user-attachments/assets/085865f2-a4cc-4c45-b0b5-45208d857026" />


Production Deployment Today: Deploy the max-pressure heuristic for live operations. It delivers superior performance, requires zero training data, has no hyperparameter drift, and imposes a milder side-street penalty[cite: 5].

Responsible AI Guardrails: Never deploy RL models directly to live signal controllers without rigorous shadow-mode logging and explicit human-in-the-loop safety override constraints[cite: 5].

The Engineering Upgrade Path (EXTENDING.md):

Transition from a queueing simulator to a fully calibrated SUMO / TraCI environment.

Replace tabular lookup with a continuous-state PPO agent (Ray RLlib) to eliminate the quantisation bottleneck.

Introduce a centralised critic (MADDPG-style) during training to optimise cross-intersection coordination before decentralising for inference.


