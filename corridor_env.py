"""A 12-intersection arterial corridor queueing simulator.

Each intersection has 4 approaches (N/S/E/W). Vehicles arrive stochastically
(Poisson) per approach; a signal phase (NS-green or EW-green) determines
which approaches discharge. Downstream spillback: if the receiving
intersection's queue on the entry approach exceeds capacity, the upstream
discharge rate for that direction is throttled — this is what makes
corridor-level coordination matter (a purely local controller can create a
green wave that just moves the jam one block).

Interface mirrors a Gym-style env: reset() -> obs, step(actions) -> (obs,
reward, done, info), so a SUMO/Gym wrapper is a drop-in replacement later.
"""
import numpy as np

N_INTERSECTIONS = 12
MAX_QUEUE = 40           # vehicles; queue capacity per approach before spillback
DISCHARGE_RATE = 3       # vehicles/step that can pass when green
STEP_SECONDS = 5
EPISODE_STEPS = 720      # 1 hour of simulated time at 5s/step


class CorridorEnv:
    def __init__(self, demand_fn, seed=0, comm_radius=1):
        self.demand_fn = demand_fn  # hour -> base arrival rate per approach
        self.rng = np.random.default_rng(seed)
        self.comm_radius = comm_radius
        self.reset()

    def reset(self, start_hour=8.0):
        self.t = 0
        self.hour = start_hour
        # queues[i, d] : queue length at intersection i, direction d (0=N,1=S,2=E,3=W)
        self.queues = np.zeros((N_INTERSECTIONS, 4), dtype=np.float32)
        self.phase = np.zeros(N_INTERSECTIONS, dtype=int)  # 0 = NS green, 1 = EW green
        self.phase_elapsed = np.zeros(N_INTERSECTIONS, dtype=int)
        self.total_delay = 0.0
        self.total_stops = 0
        return self._obs()

    def _obs(self):
        """Each intersection's local obs: own queues + phase + neighbor queue summary."""
        obs = []
        for i in range(N_INTERSECTIONS):
            own = self.queues[i]
            neighbor_q = []
            for j in range(max(0, i - self.comm_radius), min(N_INTERSECTIONS, i + self.comm_radius + 1)):
                if j != i:
                    neighbor_q.append(self.queues[j].sum())
            neighbor_mean = np.mean(neighbor_q) if neighbor_q else 0.0
            obs.append(np.concatenate([own, [self.phase[i], self.phase_elapsed[i], neighbor_mean]]))
        return obs

    def _arrivals(self):
        rate = self.demand_fn(self.hour % 24)
        arrivals = self.rng.poisson(lam=rate * STEP_SECONDS / 60.0, size=(N_INTERSECTIONS, 4))
        return arrivals.astype(np.float32)

    def step(self, actions):
        """actions[i] in {0: stay, 1: switch phase}"""
        arrivals = self._arrivals()
        self.queues += arrivals

        for i in range(N_INTERSECTIONS):
            if actions[i] == 1 and self.phase_elapsed[i] >= 4:  # min green ~20s before switching
                self.phase[i] = 1 - self.phase[i]
                self.phase_elapsed[i] = 0
            else:
                self.phase_elapsed[i] += 1

        # discharge, with downstream spillback throttling
        for i in range(N_INTERSECTIONS):
            green_dirs = [0, 1] if self.phase[i] == 0 else [2, 3]
            red_dirs = [2, 3] if self.phase[i] == 0 else [0, 1]
            for d in red_dirs:
                if self.queues[i, d] > 0.5:
                    self.total_stops += 1  # a red-phase approach with queued vehicles = a stop event
            for d in green_dirs:
                discharge = min(DISCHARGE_RATE, self.queues[i, d])
                # downstream spillback: E-direction feeds intersection i+1's W approach
                if d == 2 and i + 1 < N_INTERSECTIONS:
                    downstream_room = MAX_QUEUE - self.queues[i + 1, 3]
                    discharge = min(discharge, max(0, downstream_room))
                self.queues[i, d] -= discharge

        self.queues = np.clip(self.queues, 0, MAX_QUEUE)
        self.total_delay += float(self.queues.sum())

        self.t += 1
        self.hour += STEP_SECONDS / 3600.0
        done = self.t >= EPISODE_STEPS

        # reward: negative total delay, penalize side-street (N/S, i.e. minor) starvation
        side_street_wait_penalty = float(np.clip(self.queues[:, :2] - 25, 0, None).sum()) * 0.5
        reward = -(float(self.queues.sum()) + side_street_wait_penalty) / N_INTERSECTIONS

        info = {"total_delay": self.total_delay, "total_stops": self.total_stops,
                "mean_queue": float(self.queues.mean())}
        return self._obs(), reward, done, info

    def discretize_obs(self, obs_i):
        """Coarse-bin an intersection's obs into a small discrete state for tabular Q-learning."""
        own_total = int(np.clip(obs_i[0] + obs_i[1], 0, MAX_QUEUE) // 8)   # NS queue bucket
        own_total_ew = int(np.clip(obs_i[2] + obs_i[3], 0, MAX_QUEUE) // 8)  # EW queue bucket
        phase = int(obs_i[4])
        elapsed_bucket = int(min(obs_i[5], 8) // 2)
        neighbor_bucket = int(np.clip(obs_i[6], 0, MAX_QUEUE) // 10)
        return (own_total, own_total_ew, phase, elapsed_bucket, neighbor_bucket)
