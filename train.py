"""Curriculum training: the shared Q-agent trains across many episodes of
randomized demand-profile seeds, on the full 12-intersection corridor
(the "curriculum" here is epsilon annealing rather than a literal
node-count ramp, since tabular Q-learning at this scale converges fine
directly on the full corridor — a true deep-RL PPO agent would use the
staged 1->3->12 node curriculum described in the briefing).
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sim_env.corridor_env import CorridorEnv, N_INTERSECTIONS
from agents.q_agent import SharedQAgent
from data.demand_profiles import demand_profile

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

N_EPISODES = 400
STEPS_PER_EPISODE = 200  # sub-sampled from the full 720-step hour for training speed


def demand_with_noise(rng, noise_pct=0.3):
    def f(hour):
        base = demand_profile(hour)
        return max(0.2, base * (1 + rng.uniform(-noise_pct, noise_pct)))
    return f


def main():
    agent = SharedQAgent(alpha=0.15, gamma=0.92, epsilon=0.3, seed=42)
    rng = np.random.default_rng(123)
    episode_rewards = []

    for ep in range(N_EPISODES):
        seed = int(rng.integers(0, 1_000_000))
        start_hour = float(rng.choice([8.0, 12.0, 17.5, 21.0]))  # AM peak / midday / PM peak / off-peak
        env = CorridorEnv(demand_with_noise(np.random.default_rng(seed)), seed=seed)
        obs = env.reset(start_hour=start_hour)
        states = [env.discretize_obs(o) for o in obs]

        ep_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            actions = [agent.act(s) for s in states]
            next_obs, reward, done, info = env.step(actions)
            next_states = [env.discretize_obs(o) for o in next_obs]
            for i in range(N_INTERSECTIONS):
                agent.update(states[i], actions[i], reward, next_states[i])
            states = next_states
            ep_reward += reward
            if done:
                break
        agent.decay_epsilon()
        episode_rewards.append(ep_reward / STEPS_PER_EPISODE)

        if (ep + 1) % 50 == 0:
            recent = np.mean(episode_rewards[-50:])
            print(f"Episode {ep+1}/{N_EPISODES}  avg_reward(last50)={recent:.3f}  epsilon={agent.epsilon:.3f}  q_states={len(agent.q)}")

    agent.save(MODEL_DIR / "corridor_qtable.npz")
    with open(MODEL_DIR / "training_curve.json", "w") as f:
        json.dump({"episode_rewards": episode_rewards}, f)
    print(f"Training complete. Q-table has {len(agent.q)} learned states.")
    print(f"Saved -> {MODEL_DIR / 'corridor_qtable.npz'}")


if __name__ == "__main__":
    main()
