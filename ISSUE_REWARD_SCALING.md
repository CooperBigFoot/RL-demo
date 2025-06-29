# Improve Reward Scaling and Minor Refinements for DiscreteReservoirEnv

## Summary
The TorchRL environment implementation is excellent and ready for training with minor improvements needed. The most critical recommendation is to implement proper reward scaling to ensure stable training.

## Current State
- ✅ Correct TorchRL integration with proper inheritance from `EnvBase`
- ✅ Proper use of specs (Composite, Bounded, Categorical, Unbounded)
- ✅ Clean data flow with TensorDict
- ✅ Modular design with separated logic
- ✅ Smart use of `avg_volume` for hydropower calculation

## Key Recommendations

### 1. **Revisit Reward Scaling (Critical for Training)**

**Issue**: Current reward components are unscaled, leading to significant imbalance:
- `r_hydro`: ~95 for large release from full reservoir
- `p_flood`: -100 after weight applied (w_flood=10.0)
- `p_env`: -50 after weight applied (w_env=5.0)

The penalties are much larger than positive rewards, which can make learning unstable.

**Solution**: Normalize each reward component to [-1, 1] range before applying weights:

```python
def _calculate_reward(self, release: float, new_volume: float, old_volume: float) -> float:
    # --- Normalized Hydropower Reward ---
    max_hydro_reward = (0.10 * self.v_max) * 1.0
    avg_volume = (old_volume + new_volume) / 2
    r_hydro = (release * (avg_volume / self.v_max)) / max_hydro_reward

    # --- Normalized Flood Penalty ---
    p_flood = 0.0
    if new_volume > self.v_safe:
        violation_depth = new_volume - self.v_safe
        max_possible_violation = self.v_max - self.v_safe
        p_flood = -1.0 * (violation_depth / max_possible_violation)

    # --- Normalized Environmental Flow Penalty ---
    min_env_flow = 0.01 * self.v_max
    p_env = 0.0
    if release < min_env_flow:
        p_env = -1.0 * ((min_env_flow - release) / min_env_flow)
        
    reward = self.w_hydro * r_hydro + self.w_flood * p_flood + self.w_env * p_env
    return reward
```

### 2. **Simplify `old_volume` Retrieval**

**Current**: Recalculating `old_volume` in `_step`
**Improvement**: Have simulator return it in `info` dict

In `ReservoirSimulator.step`:
```python
old_volume = self.current_volume  # Store before modification
# ... rest of step logic ...
info = {
    "old_volume": old_volume,
    # ... other info keys ...
}
```

In `DiscreteReservoirEnv._step`:
```python
reward = self._calculate_reward(
    release=info["actual_release"],
    new_volume=new_state["volume"],
    old_volume=info["old_volume"],  # More direct
)
```

### 3. **Minor Code Refinement in `_get_observation`**

**Issue**: Inefficient conversions from tensor to item and back
**Solution**: Build tensor more directly

```python
def _get_observation(self, state: dict) -> torch.Tensor:
    day_of_year = state["day_of_year"]
    theta = 2 * torch.pi * day_of_year / 365
    
    forecast = self.simulator.get_forecast(days_ahead=10)
    normalized_forecast = torch.tensor(forecast / self.max_historical_inflow, dtype=torch.float32)

    observation = torch.cat([
        torch.tensor([state["volume_pct"]], dtype=torch.float32),
        torch.tensor([torch.sin(theta), torch.cos(theta)], dtype=torch.float32),
        normalized_forecast
    ]).to(self.device)

    return observation
```

## Priority
1. **High**: Reward scaling - Critical for stable training
2. **Medium**: Simplify old_volume retrieval - Improves code clarity
3. **Low**: Observation tensor building - Minor efficiency improvement

## Acceptance Criteria
- [ ] Implement normalized reward scaling with all components in [-1, 1] range
- [ ] Update simulator to return old_volume in info dict
- [ ] Refactor observation building to avoid unnecessary conversions
- [ ] Test that rewards are balanced and weights are meaningful
- [ ] Verify environment still works correctly with TorchRL after changes