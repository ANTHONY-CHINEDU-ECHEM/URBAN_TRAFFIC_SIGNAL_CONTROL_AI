# Extending this reference implementation

**Honest finding from this repo's evaluation:** the tabular Q-learning
agent beats fixed-time control by ~43% on total delay (exceeding the
briefing's 18% target) but does **not** beat the hand-tuned max-pressure
heuristic baseline in this simplified queueing environment. This is a
known pattern in traffic-signal RL literature at small state-space scale —
tabular Q-learning's coarse state discretization loses information a
continuous-state deep RL policy would retain. This is disclosed rather
than hidden; it's exactly the motivation for the upgrade path below.

1. **Real simulator.** Swap `sim_env/corridor_env.py` for a SUMO/TraCI
   environment with the same `reset()`/`step()` interface — calibrate to
   real corridor geometry and NGSIM-style trajectory data.
2. **Deep RL.** Replace `agents/q_agent.py`'s tabular Q-table with a PPO
   agent (Ray RLlib) operating on the continuous (undiscretized)
   observation — this is what should close the gap with, and likely
   surpass, max-pressure.
3. **True multi-agent credit assignment.** Add a centralized critic
   (MADDPG-style) during training for better coordination across
   neighboring intersections, then decentralize for deployment.
4. **NTCIP integration.** Wire the trained policy behind an NTCIP
   controller interface in shadow mode before any real-world pilot.
