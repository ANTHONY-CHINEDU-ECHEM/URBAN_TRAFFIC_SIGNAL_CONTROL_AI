import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from sim_env.corridor_env import CorridorEnv, N_INTERSECTIONS
from agents.q_agent import SharedQAgent
from agents.baselines import FixedTimeController, MaxPressureController
from data.demand_profiles import demand_profile

MODEL_PATH = ROOT / "models" / "corridor_qtable.npz"


def test_env_resets_and_steps():
    env = CorridorEnv(demand_profile, seed=1)
    obs = env.reset()
    assert len(obs) == N_INTERSECTIONS
    obs, reward, done, info = env.step([0] * N_INTERSECTIONS)
    assert "total_delay" in info
    assert isinstance(reward, float)


def test_queues_never_negative_or_over_capacity():
    from sim_env.corridor_env import MAX_QUEUE
    env = CorridorEnv(demand_profile, seed=2)
    env.reset()
    for _ in range(50):
        env.step([int(i % 2 == 0) for i in range(N_INTERSECTIONS)])
    assert (env.queues >= 0).all()
    assert (env.queues <= MAX_QUEUE).all()


def test_baselines_produce_valid_actions():
    env = CorridorEnv(demand_profile, seed=3)
    obs = env.reset()
    fixed = FixedTimeController()
    maxp = MaxPressureController()
    for fn in [fixed.act_all, maxp.act_all]:
        actions = fn(env)
        assert len(actions) == N_INTERSECTIONS
        assert all(a in (0, 1) for a in actions)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run training/train.py first")
def test_trained_agent_beats_fixed_time_on_delay():
    import numpy as np
    agent = SharedQAgent.load(MODEL_PATH)
    fixed = FixedTimeController()

    def run(fn, seed):
        env = CorridorEnv(demand_profile, seed=seed)
        obs = env.reset(start_hour=17.5)
        for _ in range(150):
            actions = fn(env, obs)
            obs, r, done, info = env.step(actions)
        return info["total_delay"]

    rl_delays, fixed_delays = [], []
    for seed in range(5):
        rl_delays.append(run(lambda e, o: [agent.act(e.discretize_obs(x), greedy=True) for x in o], seed))
        fixed_delays.append(run(lambda e, o: fixed.act_all(e), seed))
    assert np.mean(rl_delays) < np.mean(fixed_delays)
