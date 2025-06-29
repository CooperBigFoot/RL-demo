# Phase 1: Environment Implementation Plan
## Reservoir Control with Discrete Actions

### 1. Environment Wrapper Design

**Core Structure:**
```python
class DiscreteReservoirEnv(EnvBase):
    """TorchRL wrapper for discrete-action reservoir control"""
    
    def __init__(self, config):
        super().__init__()
        # Core reservoir parameters
        self.v_max = config.v_max  # Maximum capacity
        self.v_min = config.v_min  # Minimum operational level
        self.v_dead = config.v_dead  # Dead storage
        self.v_safe = 0.85  # 85% for flood safety
        
        # Discrete action space: 11 actions (0%, 1%, ..., 10%)
        self.n_actions = 11
        self.action_percentages = torch.linspace(0, 0.10, self.n_actions)
```

**TensorDict Structure:**
- Minimal keys: `observation`, `action`, `reward`, `done`
- Optional keys: `truncated`, `info` (for debugging)
- Batch dimension support for parallel environments

### 2. State Representation

**State Vector (13 dimensions):**
```python
# State components:
# - v_pct: Current volume percentage (1D)
# - sin_t, cos_t: Cyclical time encoding (2D) 
# - inflow_forecast: 10-day forecast (10D)
state_dim = 13

# Simple normalization:
# - v_pct: Already in [0, 1]
# - sin_t, cos_t: Already in [-1, 1]
# - inflow_forecast: Divide by historical max inflow
```

**Implementation:**
```python
def _get_observation(self):
    day_of_year = self.current_step % 365
    theta = 2 * np.pi * day_of_year / 365
    
    return torch.tensor([
        self.current_volume / self.v_max,  # v_pct
        np.sin(theta),  # sin_t
        np.cos(theta),  # cos_t
        *self.forecast_inflows / self.max_historical_inflow  # Normalized forecast
    ], dtype=torch.float32)
```

### 3. Action Mapping

**Discrete to Continuous Conversion:**
```python
def _map_discrete_action(self, action_idx):
    """Convert discrete action index to release percentage"""
    # action_idx in {0, 1, 2, ..., 10}
    # Maps to {0%, 1%, 2%, ..., 10%} of releasable volume
    release_pct = self.action_percentages[action_idx]
    
    # Calculate actual release
    releasable_volume = self.current_volume - self.v_min
    desired_release = release_pct * releasable_volume
    
    # Apply safety constraints
    max_daily_release = 0.10 * self.current_volume
    safe_release = self.current_volume - self.v_dead
    
    actual_release = min(desired_release, max_daily_release, safe_release)
    return actual_release
```

### 4. Reward Function

**Direct Implementation from Spec:**
```python
def _calculate_reward(self, release, new_volume):
    # Weights (to be tuned)
    w_hydro = 1.0
    w_flood = 10.0  # High penalty for flood risk
    w_env = 5.0     # Moderate penalty for env flow
    
    # Hydropower reward (proportional to head * flow)
    r_hydro = release * (self.current_volume / self.v_max)
    
    # Flood penalty
    if new_volume > self.v_safe * self.v_max:
        p_flood = -1.0 * (new_volume - self.v_safe * self.v_max)
    else:
        p_flood = 0.0
    
    # Environmental flow penalty
    min_env_flow = 0.01 * self.v_max  # 1% minimum flow
    if release < min_env_flow:
        p_env = -1.0 * (min_env_flow - release)
    else:
        p_env = 0.0
    
    # Total reward
    reward = w_hydro * r_hydro + w_flood * p_flood + w_env * p_env
    
    return reward
```

### 5. Basic Testing

**Environment Validation Tests:**

1. **Reset/Step Cycle:**
```python
def test_basic_functionality():
    env = DiscreteReservoirEnv(config)
    
    # Test reset
    td = env.reset()
    assert td["observation"].shape == (13,)
    assert not td["done"]
    
    # Test step with each action
    for action in range(11):
        td["action"] = torch.tensor(action)
        td = env.step(td)
        assert "reward" in td
        assert td["observation"].shape == (13,)
```

2. **Action Space Validation:**
```python
def test_action_constraints():
    # Verify all discrete actions map to valid releases
    # Check safety constraints are enforced
    # Ensure no negative releases
```

3. **Episode Termination:**
```python
def test_termination_conditions():
    # Test 365-day episodes
    # Test catastrophic flood (volume > v_max)
    # Test critical drought (volume < v_min)
```

4. **Spec Validation:**
```python
def test_torchrl_specs():
    env = DiscreteReservoirEnv(config)
    
    # Check observation spec
    assert env.observation_spec.shape == (13,)
    assert env.observation_spec.dtype == torch.float32
    
    # Check action spec
    assert env.action_spec.shape == ()
    assert env.action_spec.dtype == torch.int64
    assert env.action_spec.space.n == 11
```

### Implementation Timeline

**Phase 1A (Core Environment):**
- Basic reservoir dynamics
- Discrete action mapping
- Simple inflow model (deterministic + noise)

**Phase 1B (TorchRL Integration):**
- EnvBase wrapper implementation
- TensorDict input/output
- Spec definitions

**Phase 1C (Testing & Validation):**
- Unit tests for each component
- Integration test with random policy
- Performance profiling

### Key Simplifications

1. **No Complex Forecasting:** Use simple Gaussian noise for forecast uncertainty
2. **Fixed Inflow Pattern:** Sinusoidal base + noise (no complex hydrology)
3. **No Multi-Objective Weighting:** Fixed weights for now
4. **No Parallel Environments:** Start with single environment
5. **Basic State Only:** Just the essential 13 dimensions

### Next Steps

After environment is working:
1. Implement DQN for discrete action learning
2. Add basic logging and visualization
3. Create simple evaluation metrics
4. Prepare for policy training