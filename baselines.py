"""Fixed-time and max-pressure baseline controllers for comparison."""
import numpy as np


class FixedTimeController:
    """Switches phase every `cycle` steps regardless of demand."""
    def __init__(self, cycle=12):
        self.cycle = cycle

    def act_all(self, env):
        actions = []
        for i in range(len(env.phase)):
            actions.append(1 if env.phase_elapsed[i] >= self.cycle else 0)
        return actions


class MaxPressureController:
    """Switches to whichever phase (NS/EW) currently has higher total queue
    pressure, once the minimum green time has elapsed."""
    def act_all(self, env):
        actions = []
        for i in range(len(env.phase)):
            if env.phase_elapsed[i] < 4:
                actions.append(0)
                continue
            ns_pressure = env.queues[i, 0] + env.queues[i, 1]
            ew_pressure = env.queues[i, 2] + env.queues[i, 3]
            current_is_ns = env.phase[i] == 0
            should_be_ns = ns_pressure >= ew_pressure
            actions.append(0 if current_is_ns == should_be_ns else 1)
        return actions
