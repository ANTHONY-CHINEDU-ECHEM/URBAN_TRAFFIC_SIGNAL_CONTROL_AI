"""Human-readable decision trace: prints each intersection's state -> action
for a short episode, standing in for the SUMO-GUI replay visualizer
described in the briefing."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sim_env.corridor_env import CorridorEnv
from agents.q_agent import SharedQAgent
from data.demand_profiles import demand_profile

MODEL_DIR = ROOT / "models"


def main():
    agent = SharedQAgent.load(MODEL_DIR / "corridor_qtable.npz")
    env = CorridorEnv(demand_profile, seed=5)
    obs = env.reset(start_hour=17.5)
    print(f"{'step':>4} | {'int0_NS':>8} {'int0_EW':>8} {'int0_phase':>10} {'action':>7} || {'int5_NS':>8} {'int5_EW':>8}")
    for step in range(15):
        states = [env.discretize_obs(o) for o in obs]
        actions = [agent.act(s, greedy=True) for s in states]
        i0, i5 = obs[0], obs[5]
        print(f"{step:>4} | {i0[0]:>8.1f} {i0[2]:>8.1f} {int(i0[4]):>10} {actions[0]:>7} || {i5[0]:>8.1f} {i5[2]:>8.1f}")
        obs, reward, done, info = env.step(actions)
    print(f"\nAfter 15 steps: total_delay={info['total_delay']:.0f}, mean_queue={info['mean_queue']:.2f}")


if __name__ == "__main__":
    main()
