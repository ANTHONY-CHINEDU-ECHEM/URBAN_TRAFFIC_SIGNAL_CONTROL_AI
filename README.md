# Multi-Agent Reinforcement Learning for Adaptive Traffic-Signal Control

Decentralized RL agents coordinating signal timing across a 12-intersection
corridor to cut delay and stops versus fixed-time and max-pressure control.

> **Reference-implementation note.** The briefing specifies SUMO + Ray
> RLlib PPO. This repo ships a **fully self-contained corridor simulator**
> (`sim_env/corridor_env.py`, a discrete-event queueing model of arrivals,
> queues, and spillback across 12 intersections) and **tabular multi-agent
> Q-learning** (`agents/q_agent.py`) in place of SUMO/PPO — zero external
> simulator dependency, trains in under a minute on a laptop, and the same
> environment interface (`reset()`/`step()`) is what you'd swap a SUMO/Gym
> wrapper into. See `docs/EXTENDING.md` for the upgrade path to SUMO + Ray
> RLlib PPO.

## Quickstart

```bash
pip install -r requirements.txt
python training/train.py             # trains agents -> models/corridor_qtable.npz
python eval/shadow_mode_eval.py       # RL vs fixed-time vs max-pressure
python eval/decision_replay.py        # prints a step-by-step decision trace
```

## Structure
| Path | Purpose |
|---|---|
| `sim_env/corridor_env.py` | 12-intersection queueing simulator (arrivals, queues, spillback) |
| `agents/q_agent.py` | Per-intersection tabular Q-learning agent with local-communication state |
| `agents/baselines.py` | Fixed-time and max-pressure baseline controllers |
| `training/train.py` | Curriculum training loop (single -> 3-node -> full corridor) |
| `eval/shadow_mode_eval.py` | Monte Carlo evaluation across demand-profile seeds |
| `eval/decision_replay.py` | Human-readable decision trace for a sample episode |
| `data/demand_profiles.py` | AM/PM peak and off-peak arrival-rate profiles |

## License
MIT — portfolio/demonstration use.
