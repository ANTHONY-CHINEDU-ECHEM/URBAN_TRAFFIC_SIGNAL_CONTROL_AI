# Urban Traffic Signal Control AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Reference Implementation](https://img.shields.io/badge/status-reference%20implementation-orange.svg)]()
[![Evaluation: 50-seed Monte Carlo](https://img.shields.io/badge/evaluation-50--seed%20Monte%20Carlo-green.svg)]()

> A shared-policy tabular Q-learning controller for a 12-intersection arterial corridor, benchmarked against a hand-tuned max-pressure heuristic in a rigorous 50-seed Monte Carlo shadow-mode evaluation — with an honest disclosure of where the RL agent wins and where it doesn't.

<img width="1748" height="899" alt="Urban Traffic Signal Control AI" src="https://github.com/user-attachments/assets/867c9209-0651-43ec-8827-37f37ec54198" />

---

## TL;DR — Business Impact

| Question | Answer |
|---|---|
| Does RL beat the status quo? | **Yes** — 42.6% less total delay than fixed-time signals (target was 18%; beaten by 2.4×) |
| Does RL beat the best available alternative? | **No** — 40.5% *more* delay than a zero-training max-pressure heuristic |
| What's recommended for production today? | **Max-pressure heuristic** — lower delay, lower stops, milder side-street penalty, zero training cost |
| What's RL's role right now? | Documented reference implementation with a named upgrade path (Deep RL / PPO) to close the gap |

**The takeaway:** an AI solution that beats a weak baseline but loses to a strong, already-available heuristic is not a deployment case — it's a scoping exercise. This repository shows that evaluation discipline, not the headline metric, up front.

---

## 📊 Headline Results (50-seed Monte Carlo shadow-mode evaluation)

Mean total delay across 50 randomized demand seeds and 3 daily start times (8am / noon / 5:30pm), 300 simulated steps per episode, identical conditions for every controller:

| Controller | Mean Total Delay (veh-min) | vs. Fixed-Time | Mean Stops | Side-Street Max Wait |
|---|---:|---:|---:|---:|
| Fixed-Time (status quo) | 28,305.5 | — | 5,580.2 | 3.2 min |
| **RL (Tabular Q-Learning)** | 16,250.8 | **−42.6%** | 5,030.0 (−9.9%) | 4.9 min (**worst of the three**) |
| **Max-Pressure (heuristic)** | **11,570.8** | **−59.1%** | **4,385.4** (−21.4%) | 3.9 min |

**Reading this table honestly:**
- RL clearly beats the do-nothing baseline — a real, quantifiable win on delay and stops.
- RL is **40.5% worse than max-pressure** on the metric that actually matters for a deployment decision, and it's the *worst* of all three controllers on side-street equity.
- Max-pressure — a handful of lines of reactive logic, no training data, no hyperparameters — is the best-performing controller in the entire evaluation.

---

## 🧭 Why This Matters (and why the negative result is the point)

Most "AI beats traffic" portfolio pieces stop at the first comparison (RL vs. fixed-time) because it's the one that flatters the model. This project deliberately goes one step further and asks the harder question: **does RL beat the best simple alternative already sitting on the table?**

The answer is no — and the reason is diagnosable, not mysterious:

> Max-pressure reads the *exact* queue length on every approach, every step. The tabular Q-learning agent only reads which discretized *bucket* that queue falls into — the same policy for a 6-car queue and a 9-car queue. It's an information bottleneck in the state representation, not a training failure (the reward curve shows real, steady learning across 400 episodes).

This is a well-documented failure mode for tabular RL at small state-space scale, and it has a concrete, named fix (see [Roadmap](#-roadmap-closing-the-gap-with-max-pressure)) — but it means max-pressure is the correct production recommendation *today*.

---

## 🏙️ Problem & Corridor Profile

- **Scale:** 12 sequential intersections along a shared arterial corridor, 4 approaches (N/S/E/W) each — a genuinely multi-agent coordination problem, not 12 copies of one intersection.
- **Demand:** a calibrated 24-hour AM/PM-peaked profile swinging **~6×** between overnight trough (~1.2 veh/min/approach) and evening peak (~7.9 veh/min/approach) — the reason a single fixed-time plan is structurally unable to keep up.
- **Ground truth check:** 10,368 loop-detector readings (12 intersections × 4 approaches, sampled every 5 min) confirm a near-even 29–33% demand split across approaches, validating the simulator against the demand profile it claims to follow.
- **Simplification, disclosed up front:** this is a calibrated queueing-model simulation, not a SUMO/TraCI environment. `EXTENDING.md` names swapping to a fully calibrated micro-simulator as the first upgrade before any real-world pilot.

## 🎯 Evaluation Metrics

Every controller is scored on three metrics chosen to catch three different failure modes:

| Metric | What it catches |
|---|---|
| **Total delay** (vehicle-minutes) | Primary optimization target — corridor-wide efficiency |
| **Total stops** | Ride comfort / fuel & emissions proxy — a smoother corridor stops less even at similar delay |
| **Side-street max wait** | Equity — catches a controller that optimizes the main arterial by starving cross traffic |

RL's 42.6% delay win on the main corridor comes at the cost of the **worst side-street wait time of all three controllers (4.9 min)** — worse than even the demand-blind fixed-time plan. This equity trade-off, not called for in the original brief, turned out to be one of the evaluation's most important findings.

---

## 🧠 Modeling Approach

| | |
|---|---|
| **Architecture** | One shared Q-table (`defaultdict(state → [Q_stay, Q_switch])`), updated by and applied to all 12 intersections independently — 12× the effective training data of one policy per intersection, with decentralized execution |
| **State** | Discretized per-intersection queue lengths (N/S/E/W), current phase, and steps since last switch — 138 distinct states learned across training |
| **Action space** | Binary: `stay` or `switch` phase, decided independently at every intersection, every step |
| **Reward** | `−(delay incurred this step)` — the agent learns to minimize delay, not chase a score |
| **Training** | 400 episodes, ε-greedy (0.20 → 0.02, ×0.995/episode decay), α = 0.15, γ = 0.92 |
| **Convergence** | Mean episode reward improved from −4.16 to −2.09 (first 10 vs. last 10 episodes) — clear, steady learning despite high episode-to-episode variance from randomized start times |

## ⚖️ The Baselines

| | Fixed-Time | Max-Pressure |
|---|---|---|
| Logic | Switch phase every 12 steps, unconditionally | Switch to the direction with the larger total queue, after a 4-step minimum green |
| Demand awareness | None | Full — reads exact queue length every step |
| Tuning required | One number (cycle length) | None |
| Training data | None | None |

---

## ✅ Recommendation

**Deploy max-pressure for live signal control today.** It delivers the lowest delay, lowest stops, and a milder side-street penalty than RL — with no training cost, no hyperparameter drift to manage, and no Q-table to keep fresh as demand patterns shift.

**Keep RL on the roadmap, not on the roundabout.** The path to closing the gap is well-defined, not speculative:

## 🗺️ Roadmap: Closing the Gap with Max-Pressure

1. **A real simulator** — swap the queueing-model environment for a calibrated SUMO/TraCI environment using NGSIM-style trajectory data.
2. **Deep RL on continuous state** — replace the tabular Q-table with a PPO agent (Ray RLlib) operating on undiscretized observations, directly addressing the information-bottleneck finding above.
3. **True multi-agent credit assignment** — add a centralized critic (MADDPG-style) during training for genuine cross-intersection coordination, then decentralize for inference.
4. **NTCIP integration, shadow-mode first** — wire any trained policy behind a real controller interface in shadow mode, logged but never actuating live signals, until it clears max-pressure on the same evaluation discipline used here.

See `EXTENDING.md` for full architectural detail.

### Responsible deployment guardrails
- No RL model is deployed to a live signal controller without shadow-mode logging and human-in-the-loop override.
- The side-street wait regression under RL is treated as a blocking issue, not a footnote, until resolved or explicitly accepted by a human owner.
- Every future policy revision is benchmarked against max-pressure — not against the previous RL version — so the bar never quietly moves.

---

## 🗂️ Repository Structure

```
URBAN_TRAFFIC_SIGNAL_CONTROL_AI/
├── demand_profiles.py                 # AM/PM-peaked arrival-rate functions (veh/min)
├── baselines.py                       # Fixed-time and max-pressure baseline controllers
├── corridor_qtable.npz                # Learned Q-table (138 states, 400 training episodes)
├── training_curve.json                # Per-episode reward tracking data
├── shadow_mode_report.json            # 50-seed Monte Carlo benchmark output
├── sample_detector_log.csv            # 10,368 rows of loop-detector telemetry
├── TRAFFIC OPTIMIZATION DASHBOARD.py  # Interactive decision-support dashboard
├── EXTENDING.md                       # Architectural upgrade path (SUMO/TraCI & Deep RL)
└── LICENSE
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/URBAN_TRAFFIC_SIGNAL_CONTROL_AI.git
cd URBAN_TRAFFIC_SIGNAL_CONTROL_AI

# Run the performance evaluation dashboard
python "TRAFFIC OPTIMIZATION DASHBOARD.py"
```

<img width="1280" height="832" alt="Traffic Optimization Dashboard" src="https://github.com/user-attachments/assets/085865f2-a4cc-4c45-b0b5-45208d857026" />

---

## Skills Demonstrated

Multi-agent reinforcement learning · parameter sharing · Monte Carlo shadow-mode evaluation · honest negative-result reporting · responsible-deployment recommendation

---

**Anthony Chinedu Echem** — Machine Learning Engineer & Data Scientist
Portfolio contact available on request
