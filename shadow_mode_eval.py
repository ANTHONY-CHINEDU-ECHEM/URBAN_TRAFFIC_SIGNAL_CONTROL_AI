"""Monte Carlo evaluation: RL policy vs. fixed-time vs. max-pressure,
across 50 randomized demand-profile seeds, matching the briefing's
evaluation methodology.
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sim_env.corridor_env import CorridorEnv
from agents.q_agent import SharedQAgent
from agents.baselines import FixedTimeController, MaxPressureController
from data.demand_profiles import demand_profile

MODEL_DIR = ROOT / "models"
N_SEEDS = 50
STEPS = 300


def run_episode(controller_fn, seed, start_hour):
    env = CorridorEnv(demand_profile, seed=seed)
    obs = env.reset(start_hour=start_hour)
    for _ in range(STEPS):
        actions = controller_fn(env, obs)
        obs, reward, done, info = env.step(actions)
        if done:
            break
    side_street_wait = float(env.queues[:, :2].max())
    return info["total_delay"], info["total_stops"], side_street_wait


def main():
    agent = SharedQAgent.load(MODEL_DIR / "corridor_qtable.npz")
    fixed = FixedTimeController(cycle=12)
    maxp = MaxPressureController()

    def rl_fn(env, obs):
        states = [env.discretize_obs(o) for o in obs]
        return [agent.act(s, greedy=True) for s in states]

    def fixed_fn(env, obs):
        return fixed.act_all(env)

    def maxp_fn(env, obs):
        return maxp.act_all(env)

    results = {"rl": [], "fixed_time": [], "max_pressure": []}
    rng = np.random.default_rng(999)
    for _ in range(N_SEEDS):
        seed = int(rng.integers(0, 1_000_000))
        start_hour = float(rng.choice([8.0, 17.5, 12.0]))
        for name, fn in [("rl", rl_fn), ("fixed_time", fixed_fn), ("max_pressure", maxp_fn)]:
            delay, stops, side_wait = run_episode(fn, seed, start_hour)
            results[name].append({"delay": delay, "stops": stops, "side_street_max_wait": side_wait})

    summary = {}
    for name in results:
        delays = [r["delay"] for r in results[name]]
        stops = [r["stops"] for r in results[name]]
        side = [r["side_street_max_wait"] for r in results[name]]
        summary[name] = {"mean_total_delay": round(float(np.mean(delays)), 1),
                          "mean_stops": round(float(np.mean(stops)), 1),
                          "mean_side_street_max_wait": round(float(np.mean(side)), 1)}

    fixed_delay = summary["fixed_time"]["mean_total_delay"]
    rl_delay = summary["rl"]["mean_total_delay"]
    delay_reduction_pct = round((fixed_delay - rl_delay) / fixed_delay * 100, 2)

    fixed_stops = summary["fixed_time"]["mean_stops"]
    rl_stops = summary["rl"]["mean_stops"]
    stops_reduction_pct = round((fixed_stops - rl_stops) / fixed_stops * 100, 2) if fixed_stops > 0 else 0.0

    report = {"n_seeds": N_SEEDS, "summary": summary,
              "rl_delay_reduction_pct_vs_fixed_time": delay_reduction_pct,
              "rl_stops_reduction_pct_vs_fixed_time": stops_reduction_pct,
              "meets_18pct_delay_target": bool(delay_reduction_pct >= 18.0)}
    with open(MODEL_DIR / "shadow_mode_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
