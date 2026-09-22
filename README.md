# Urban Traffic Signal Control AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A shared policy tabular Q learning controller for a twelve intersection arterial corridor, benchmarked against a hand tuned max pressure heuristic through a rigorous fifty seed Monte Carlo shadow mode evaluation. This repository provides a transparent, evidence based account of where the reinforcement learning agent outperforms the current standard, where it does not, and what would need to change for it to be deployment ready.

<img width="1748" height="899" alt="Urban Traffic Signal Control AI" src="https://github.com/user-attachments/assets/867c9209-0651-43ec-8827-37f37ec54198" />

---

## Table of Contents

1. [Executive Summary](#executive-summary-business-impact)
2. [Headline Results](#headline-results-fifty-seed-monte-carlo-shadow-mode-evaluation)
3. [Why This Matters](#why-this-matters-and-why-a-negative-result-is-the-point)
4. [Problem and Corridor Profile](#problem-and-corridor-profile)
5. [Evaluation Methodology](#evaluation-methodology)
6. [Modeling Approach](#modeling-approach)
7. [The Baselines](#the-baselines)
8. [Recommendation](#recommendation)
9. [Roadmap](#roadmap-closing-the-gap-with-max-pressure)
10. [Responsible Deployment Guardrails](#responsible-deployment-guardrails)
11. [Repository Structure](#repository-structure)
12. [Installation and Requirements](#installation-and-requirements)
13. [Getting Started and the Decision Support Dashboard](#getting-started-and-the-decision-support-dashboard)
14. [Limitations and Scope](#limitations-and-scope)
15. [Skills Demonstrated](#skills-demonstrated)
16. [License and Contact](#license-and-contact)

---

## Executive Summary: Business Impact

| Question | Answer |
|---|---|
| Does the reinforcement learning controller outperform the status quo? | Yes. It reduces total delay by 42.6 percent compared to fixed time signals, exceeding the original target of 18 percent by more than double. |
| Does it outperform the best available alternative? | No. It produces 40.5 percent more delay than a max pressure heuristic that requires zero training and zero ongoing maintenance. |
| What should be deployed in production today? | The max pressure heuristic. It delivers lower delay, fewer stops, a smaller side street penalty, and no training cost. |
| What is the role of reinforcement learning right now? | A fully documented reference implementation with a clearly defined, technically grounded upgrade path (Deep RL using PPO) to close the performance gap before any production consideration. |
| What is the primary business risk of over selling the reinforcement learning result? | Presenting only the comparison against fixed time signals would materially overstate the value of the model and could lead to a deployment decision that performs worse than an already available, essentially free alternative. |

**The core finding:** a solution that beats a weak baseline but loses to a strong, already available heuristic is not yet a deployment case. It is a scoping exercise. This repository is built to demonstrate that kind of evaluation discipline up front, rather than presenting only the headline metric that looks most favorable. That discipline, more than the model itself, is the deliverable.

---

## Headline Results (Fifty Seed Monte Carlo Shadow Mode Evaluation)

Mean total delay was measured across 50 randomized demand seeds and three daily start times (8:00 am, 12:00 pm, and 5:30 pm), with 300 simulated steps per episode and identical conditions applied to every controller. Running the evaluation across randomized seeds, rather than a single scenario, ensures the reported numbers reflect typical performance rather than a result that happens to favor one controller on one lucky draw.

| Controller | Mean Total Delay (vehicle minutes) | Change vs. Fixed Time | Mean Stops | Side Street Max Wait |
|---|---:|---:|---:|---:|
| Fixed Time (status quo) | 28,305.5 | baseline | 5,580.2 | 3.2 min |
| Reinforcement Learning (Tabular Q Learning) | 16,250.8 | 42.6 percent reduction | 5,030.0 (9.9 percent reduction) | 4.9 min (worst of the three) |
| Max Pressure (heuristic) | 11,570.8 | 59.1 percent reduction | 4,385.4 (21.4 percent reduction) | 3.9 min |

**An honest reading of this table:**
- The reinforcement learning controller clearly outperforms the do nothing baseline, delivering a real, quantifiable improvement in both delay and stops. This is not a trivial result and reflects genuine learned behavior rather than noise.
- It is nonetheless 40.5 percent worse than max pressure on the metric that actually determines a deployment decision, and it produces the weakest side street equity outcome of all three controllers, a finding that would be easy to miss if the evaluation stopped at the fixed time comparison.
- Max pressure, a small set of reactive logic rules with no training data and no hyperparameters to tune, is the best performing controller in the entire evaluation on every metric measured.

---

## Why This Matters (and Why a Negative Result Is the Point)

Most portfolio projects that claim artificial intelligence improves traffic outcomes stop at the first comparison, reinforcement learning against fixed time signals, because that is the comparison that flatters the model. This project deliberately goes one step further and asks the harder, commercially relevant question: does reinforcement learning outperform the best simple alternative that is already available at no cost?

The answer is no, and the reason is diagnosable rather than mysterious:

> Max pressure reads the exact queue length on every approach at every step. The tabular Q learning agent only reads which discretized bucket that queue falls into, applying the identical policy to a six car queue and a nine car queue. This is an information bottleneck built into the state representation, not a training failure. The reward curve shows real, steady learning across 400 episodes, which rules out an undertrained agent as the explanation for the gap.

This is a well documented limitation of tabular reinforcement learning at small state space scale, and it has a concrete, named remedy described in the roadmap section below. In the meantime, max pressure is the correct, evidence backed production recommendation today, and this repository states that plainly rather than burying it beneath a favorable headline number.

---

## Problem and Corridor Profile

- **Scale.** Twelve sequential intersections along a shared arterial corridor, each with four approaches (north, south, east, and west). This is a genuine multi agent coordination problem, not twelve independent copies of a single intersection, because a decision at one intersection changes the queue that arrives at the next.
- **Demand.** A calibrated 24 hour profile with morning and evening peaks, swinging by a factor of roughly six between the overnight trough (approximately 1.2 vehicles per minute per approach) and the evening peak (approximately 7.9 vehicles per minute per approach). This swing is the structural reason a single fixed time plan cannot keep pace with real conditions throughout the day.
- **Ground truth validation.** 10,368 loop detector readings (twelve intersections multiplied by four approaches, sampled every five minutes) confirm a near even split of 29 to 33 percent demand across approaches, validating the simulator against the demand profile it is designed to follow rather than relying on an unverified synthetic assumption.
- **Disclosed simplification.** This is a calibrated queueing model simulation, not a SUMO or TraCI microsimulation environment. EXTENDING.md names the move to a fully calibrated microsimulator as the first required upgrade before any real world pilot, and that limitation is treated as central to interpreting these results rather than a footnote.

## Evaluation Methodology

Rigor in the evaluation protocol is treated as a first class part of this project, not an afterthought:

- **Fifty seed Monte Carlo design.** Every controller is evaluated on the same 50 randomized demand seeds and the same three daily start times, for a total of 150 episodes per controller, so that differences in reported performance reflect the controller and not the luck of a single scenario.
- **Shadow mode framing.** All evaluation is conducted in a logged, non actuating context, consistent with how any future candidate policy would first be assessed against a live corridor before being trusted to control real signals.
- **Three metric scoring.** Every controller is scored on three metrics chosen because each one catches a different failure mode:

| Metric | What it catches |
|---|---|
| Total delay (vehicle minutes) | The primary optimization target: corridor wide efficiency |
| Total stops | A proxy for ride comfort, fuel consumption, and emissions; a smoother corridor stops less even at similar delay levels |
| Side street max wait | An equity check that catches a controller optimizing the main arterial at the expense of starving cross traffic |

The reinforcement learning controller's 42.6 percent delay improvement on the main corridor comes at the cost of the worst side street wait time of all three controllers, at 4.9 minutes, worse than even the demand blind fixed time plan. This equity trade off was not called for in the original project brief, and it turned out to be one of the most important findings of the evaluation, illustrating why a single headline metric is an unreliable basis for a deployment decision.

---

## Modeling Approach

| | |
|---|---|
| Architecture | A single shared Q table (a dictionary mapping state to the values of staying versus switching), updated by and applied to all twelve intersections independently. This gives twelve times the effective training data of a single per intersection policy, while still allowing fully decentralized execution at inference time. |
| State representation | Discretized per intersection queue lengths for each of the four approaches, the current signal phase, and the number of steps since the last switch, totaling 138 distinct states learned during training. |
| Action space | Binary: stay in the current phase or switch, decided independently at every intersection on every simulation step. |
| Reward function | The negative of the delay incurred at that step, so the agent learns to minimize delay directly rather than chase an indirect or proxy score. |
| Training configuration | 400 episodes, with an epsilon greedy exploration schedule decaying from 0.20 to 0.02 at a rate of 0.995 per episode, a learning rate of 0.15, and a discount factor of 0.92. |
| Convergence evidence | Mean episode reward improved from negative 4.16 to negative 2.09, comparing the first ten episodes to the last ten. This demonstrates clear, steady learning despite high episode to episode variance caused by randomized start times, and supports the conclusion that the performance gap versus max pressure is a representational limit rather than incomplete training. |

## The Baselines

| | Fixed Time | Max Pressure |
|---|---|---|
| Control logic | Switches phase every 12 steps, unconditionally, regardless of actual traffic present | Switches to the direction with the larger total queue, after a minimum green period of 4 steps, reacting directly to observed demand |
| Demand awareness | None | Full: reads exact queue length at every step for every approach |
| Tuning required | One parameter (cycle length) | None |
| Training data required | None | None |
| Ongoing maintenance | Periodic manual retiming as demand patterns drift | None |

---

## Recommendation

**Deploy the max pressure heuristic for live signal control today.** Across every metric measured, it delivers the lowest delay, the lowest number of stops, and a milder side street penalty than reinforcement learning, with no training cost, no hyperparameters to manage over time, and no Q table that needs to be kept current as demand patterns shift. From a total cost of ownership perspective, it is the stronger business case as well as the stronger technical one.

**Keep reinforcement learning on the roadmap, not on the roundabout.** The path to closing the performance gap is well defined and technically grounded rather than speculative, and is described in detail below.

## Roadmap: Closing the Gap With Max Pressure

1. **A real simulator.** Replace the queueing model environment with a calibrated SUMO or TraCI environment built on trajectory data comparable to the NGSIM dataset, so that results generalize more credibly to a real corridor.
2. **Deep reinforcement learning on continuous state.** Replace the tabular Q table with a PPO agent, implemented using Ray RLlib, operating on undiscretized observations. This directly addresses the information bottleneck identified in the analysis above, which is the primary diagnosed cause of the current performance gap.
3. **True multi agent credit assignment.** Add a centralized critic during training, following an approach comparable to MADDPG, to achieve genuine cross intersection coordination during learning, then decentralize the resulting policy for inference so that each intersection still acts independently in production.
4. **NTCIP integration, shadow mode first.** Connect any trained policy to a real controller interface in shadow mode, logging its decisions without ever actuating live signals, until it clears max pressure under the same evaluation discipline used throughout this repository.

See EXTENDING.md for full architectural detail on each of these steps.

### Responsible Deployment Guardrails

- No reinforcement learning model is deployed to a live signal controller without shadow mode logging and human oversight with override capability.
- The side street wait regression observed under reinforcement learning is treated as a blocking issue, not a footnote, until it is resolved or explicitly accepted in writing by a human owner accountable for that decision.
- Every future policy revision is benchmarked against max pressure, not against the previous reinforcement learning version, so the performance bar that matters for deployment never quietly moves.
- Any change to the demand profile, corridor geometry, or evaluation protocol triggers a full re run of the fifty seed Monte Carlo evaluation before a new recommendation is issued.

---

## Repository Structure

```
URBAN_TRAFFIC_SIGNAL_CONTROL_AI/
├── demand_profiles.py                 # AM/PM peaked arrival rate functions (vehicles per minute)
├── baselines.py                       # Fixed time and max pressure baseline controllers
├── corridor_qtable.npz                # Learned Q table (138 states, 400 training episodes)
├── training_curve.json                # Per episode reward tracking data
├── shadow_mode_report.json            # Fifty seed Monte Carlo benchmark output
├── sample_detector_log.csv            # 10,368 rows of loop detector telemetry
├── TRAFFIC OPTIMIZATION DASHBOARD.py  # Interactive decision support dashboard
├── EXTENDING.md                       # Architectural upgrade path (SUMO/TraCI and Deep RL)
└── LICENSE
```

## Installation and Requirements

This project targets Python 3.9 or later. A typical setup looks like the following:

```bash
# Clone the repository
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/URBAN_TRAFFIC_SIGNAL_CONTROL_AI.git
cd URBAN_TRAFFIC_SIGNAL_CONTROL_AI

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # on Windows use: venv\Scripts\activate

# Install project dependencies
pip install -r requirements.txt
```

Core dependencies include standard scientific Python tooling (NumPy and pandas for data handling, and a plotting or dashboarding library for the decision support dashboard). Consult requirements.txt in the repository for the exact pinned versions used to produce the results reported here.

## Getting Started and the Decision Support Dashboard

The fastest way to explore these results interactively is through the included decision support dashboard, which visualizes the fifty seed Monte Carlo evaluation, the training convergence curve, and the per controller comparison shown in the headline results table above.

```bash
# Run the performance evaluation dashboard
python "TRAFFIC OPTIMIZATION DASHBOARD.py"
```

<img width="1280" height="832" alt="Traffic Optimization Dashboard" src="https://github.com/user-attachments/assets/085865f2-a4cc-4c45-b0b5-45208d857026" />

The dashboard is intended as the primary artifact for a business or operational stakeholder who wants to interrogate the evaluation without reading source code. It surfaces the delay, stops, and equity comparison across all three controllers, alongside the training curve that supports the convergence claims made in the modeling section above.

If you want to reproduce the underlying evaluation rather than only viewing its output, shadow_mode_report.json contains the full fifty seed benchmark results, and training_curve.json contains the raw per episode reward data referenced in the convergence discussion.

---

## Limitations and Scope

This repository is presented as a reference implementation and an evaluation case study, not a production ready traffic control system. In addition to the simulator simplification noted above, the following should be considered before extending or citing this work:

- The corridor topology, demand profile, and baseline tuning are all specific to this evaluation and would need to be recalibrated against real detector data before any conclusions transfer to a different corridor.
- The tabular representation limits the state space that can practically be learned; the roadmap section describes the planned move to a continuous state representation.
- No live field trial has been conducted. All results in this repository come from simulation, and the responsible deployment guardrails above are written specifically to prevent premature field deployment ahead of that validation step.

## Skills Demonstrated

Multi agent reinforcement learning, parameter sharing across a shared policy, Monte Carlo shadow mode evaluation design, transparent negative result reporting, and a responsible, evidence based deployment recommendation.

---

## License and Contact

This project is released under the MIT License. See LICENSE for full terms.

**Anthony Chinedu Echem**
Machine Learning Engineer and Data Scientist
Portfolio contact available on request
