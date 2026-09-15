"""Arrival-rate (vehicles/min per approach) demand profiles across a
24-hour simulated day, used to generate training/eval episodes."""
import numpy as np

HOURS = np.arange(24)

def demand_profile(hour: int) -> float:
    """Base per-approach arrival rate (veh/min), AM/PM peaked."""
    am_peak = 6.0 * np.exp(-((hour - 8.0) ** 2) / (2 * 1.3 ** 2))
    pm_peak = 7.0 * np.exp(-((hour - 17.5) ** 2) / (2 * 1.6 ** 2))
    base = 1.2
    return float(base + am_peak + pm_peak)


def full_day_profile():
    return {h: demand_profile(h) for h in HOURS}


def export_sample_detector_log(path, seed=0, hours=(6, 24), step_minutes=5):
    """Exports a synthetic loop-detector count log (the kind of tabular
    dataset a real deployment would have), for inspection/EDA purposes."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for h in np.arange(hours[0], hours[1], step_minutes / 60.0):
        rate = demand_profile(h % 24)
        for intersection in range(12):
            for approach, name in enumerate(["N", "S", "E", "W"]):
                count = rng.poisson(rate * step_minutes / 60.0 * rng.uniform(0.85, 1.15))
                rows.append({"hour": round(h, 3), "intersection_id": intersection,
                             "approach": name, "vehicle_count": int(count)})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__" and False:
    pass
