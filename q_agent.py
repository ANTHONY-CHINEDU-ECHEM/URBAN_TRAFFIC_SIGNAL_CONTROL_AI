"""Tabular Q-learning multi-agent controller with parameter sharing: all 12
intersections share ONE Q-table (keyed by discretized local state), the
lightweight stand-in for the "shared policy network" PPO design in the
briefing. Each intersection acts independently at each step using its own
local (discretized) observation.
"""
import numpy as np
from collections import defaultdict

ACTIONS = [0, 1]  # 0 = stay, 1 = switch phase


class SharedQAgent:
    def __init__(self, alpha=0.15, gamma=0.92, epsilon=0.2, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.q = defaultdict(lambda: np.zeros(len(ACTIONS)))

    def act(self, state, greedy=False):
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(ACTIONS)))
        return int(np.argmax(self.q[state]))

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.alpha * td_error

    def decay_epsilon(self, factor=0.995, min_eps=0.02):
        self.epsilon = max(min_eps, self.epsilon * factor)

    def save(self, path):
        # convert defaultdict to plain arrays for np.savez
        keys = list(self.q.keys())
        values = np.array([self.q[k] for k in keys])
        np.savez_compressed(path, keys=np.array(keys, dtype=object), values=values, allow_pickle=True)

    @classmethod
    def load(cls, path):
        agent = cls()
        data = np.load(path, allow_pickle=True)
        for k, v in zip(data["keys"], data["values"]):
            agent.q[tuple(k)] = v
        return agent
