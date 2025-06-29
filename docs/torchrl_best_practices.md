# TorchRL Best Practices and Implementation Guide

## 1. Latest TorchRL Version and Installation Requirements

### Installation

```bash
# Basic installation
uv add torch torchrl

# With specific CUDA version (if needed)
uv add torch torchrl --index-url https://download.pytorch.org/whl/cu118

# Additional useful packages
uv add gymnasium tensordict tqdm wandb
```

### Version Information

- **TorchRL**: Latest stable version (0.3.0+) recommended
- **PyTorch**: 2.0+ required for full feature support
- **Python**: 3.8+ recommended

## 2. Key TorchRL Modules and Components

### Core Components

1. **TensorDict**: The fundamental data structure in TorchRL
   - Handles batched, nested tensor operations
   - Enables modular policy and environment design
   - Supports lazy operations and memory mapping

2. **Environments (`torchrl.envs`)**
   - `GymEnv` / `GymWrapper`: Integration with Gymnasium
   - `TransformedEnv`: Apply transforms to observations/actions
   - `ParallelEnv`: Run multiple environments in parallel
   - `EnvCreator`: Factory pattern for environment creation

3. **Modules (`torchrl.modules`)**
   - `TensorDictModule`: Wrap neural networks for TensorDict I/O
   - `ProbabilisticActor`: Stochastic policy networks
   - `ValueOperator`: Value function approximators
   - `ActorValueOperator`: Combined actor-critic networks

4. **Data Collection (`torchrl.collectors`)**
   - `SyncDataCollector`: Single-process collection
   - `MultiSyncDataCollector`: Multi-process parallel collection
   - Supports various exploration strategies

5. **Replay Buffers (`torchrl.data`)**
   - `ReplayBuffer`: Standard experience replay
   - `PrioritizedReplayBuffer`: Prioritized experience replay
   - `TensorDictReplayBuffer`: Optimized for TensorDict storage

## 3. Best Practices for Environment Integration

### Basic Environment Setup

```python
from torchrl.envs import GymEnv, TransformedEnv, Compose
from torchrl.envs.transforms import (
    ToTensorImage, Resize, GrayScale, ObservationNorm, RewardScaling
)

# Create base environment
base_env = GymEnv("CartPole-v1")

# Apply transforms
env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        RewardScaling(scale=0.1),
    )
)
```

### Custom Environment Wrapper

```python
from torchrl.envs import EnvBase
from tensordict import TensorDict

class CustomReservoirEnv(EnvBase):
    def __init__(self, reservoir_config):
        super().__init__()
        self.reservoir = ReservoirSimulator(reservoir_config)
        
    def _reset(self, tensordict):
        # Reset reservoir state
        state = self.reservoir.reset()
        return TensorDict({
            "observation": torch.tensor(state),
            "done": torch.tensor(False),
        })
        
    def _step(self, tensordict):
        action = tensordict["action"]
        next_state, reward, done = self.reservoir.step(action)
        return TensorDict({
            "observation": torch.tensor(next_state),
            "reward": torch.tensor(reward),
            "done": torch.tensor(done),
        })
```

## 4. Recommended Algorithms for Continuous Control

### For Reservoir Control (Continuous Action Space)

1. **SAC (Soft Actor-Critic)** - Recommended for stability

   ```python
   from torchrl.objectives import SACLoss
   from torchrl.modules import TanhNormal
   
   # Stable for continuous control with exploration
   loss_module = SACLoss(
       actor_network=actor,
       qvalue_network=qvalue,
       value_network=value,
       num_qvalue_nets=2,
       loss_function="smooth_l1",
   )
   ```

2. **TD3 (Twin Delayed DDPG)** - Good for deterministic policies

   ```python
   from torchrl.objectives import TD3Loss
   
   loss_module = TD3Loss(
       actor_network=actor,
       qvalue_network=qvalue,
       action_spec=env.action_spec,
   )
   ```

3. **PPO (Proximal Policy Optimization)** - For on-policy learning

   ```python
   from torchrl.objectives import ClipPPOLoss
   
   loss_module = ClipPPOLoss(
       actor=actor,
       critic=critic,
       clip_epsilon=0.2,
       entropy_bonus=True,
       entropy_coef=0.01,
   )
   ```

## 5. Code Structure and Project Organization

### Recommended Project Structure

```
rl_demo/
├── src/
│   └── rl_demo/
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── reservoir_env.py
│       │   └── wrappers.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── actor_networks.py
│       │   └── critic_networks.py
│       ├── trainers/
│       │   ├── __init__.py
│       │   ├── sac_trainer.py
│       │   └── base_trainer.py
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── replay_buffer.py
│       │   └── logger.py
│       └── configs/
│           └── default_config.yaml
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
├── pyproject.toml
└── README.md
```

### Training Loop Pattern

```python
# Main training loop structure
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=1000,
    total_frames=1_000_000,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=100_000),
    sampler=SamplerWithoutReplacement(),
)

for i, batch in enumerate(collector):
    # Add to replay buffer
    replay_buffer.extend(batch)
    
    # Sample and train
    if len(replay_buffer) > batch_size:
        sample = replay_buffer.sample(batch_size)
        loss_dict = loss_module(sample)
        
        # Optimize
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()
        
    # Update target networks (if applicable)
    if i % target_update_freq == 0:
        update_target_networks()
```

## 6. Common Pitfalls and Solutions

### 1. **Device Mismatch**

- **Problem**: Tensors on different devices (CPU vs GPU)
- **Solution**: Use `.to(device)` consistently or set default device

```python
torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
```

### 2. **Memory Leaks with Replay Buffers**

- **Problem**: Unbounded memory growth
- **Solution**: Use `LazyTensorStorage` with proper max_size

```python
storage = LazyTensorStorage(max_size=100_000, device="cpu")
```

### 3. **Slow Data Collection**

- **Problem**: Sequential environment stepping is slow
- **Solution**: Use `ParallelEnv` or `MultiSyncDataCollector`

```python
env = ParallelEnv(
    num_workers=4,
    create_env_fn=lambda: GymEnv("CartPole-v1"),
)
```

### 4. **Gradient Explosion/Vanishing**

- **Problem**: Unstable training
- **Solution**: Use gradient clipping and proper initialization

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
```

### 5. **Poor Exploration**

- **Problem**: Policy gets stuck in local optima
- **Solution**: Use appropriate exploration strategies

```python
from torchrl.modules import AdditiveGaussianNoise

exploration_module = AdditiveGaussianNoise(
    sigma=0.1,
    action_key="action",
)
```

## 7. Reservoir Control Specific Considerations

### State Representation

- Include water levels, inflow predictions, demand forecasts
- Normalize states using running statistics or known bounds
- Consider temporal features (time of day, season)

### Action Space Design

- Continuous: Release rates, gate positions
- Discrete: Operational modes, rule activations
- Multi-objective: Balance flood control, water supply, ecology

### Reward Engineering

- Multi-objective reward combining:
  - Water supply reliability
  - Flood damage minimization
  - Environmental flow requirements
  - Energy generation (if applicable)
- Consider shaped rewards for faster learning

### Example Reservoir Environment

```python
class ReservoirControlEnv(EnvBase):
    def __init__(self, config):
        super().__init__()
        self.dt = config.timestep
        self.capacity = config.capacity
        self.min_release = config.min_release
        self.max_release = config.max_release
        
        # Define specs
        self.observation_spec = TensorSpec(
            shape=(5,),  # [storage, inflow, demand, time_of_day, day_of_year]
            dtype=torch.float32,
        )
        self.action_spec = BoundedTensorSpec(
            low=self.min_release,
            high=self.max_release,
            shape=(1,),
            dtype=torch.float32,
        )
        
    def _step(self, tensordict):
        release = tensordict["action"].item()
        storage = tensordict["storage"]
        inflow = tensordict["inflow"]
        
        # Update storage
        new_storage = storage + (inflow - release) * self.dt
        new_storage = torch.clamp(new_storage, 0, self.capacity)
        
        # Calculate reward
        reward = self._calculate_reward(new_storage, release, demand)
        
        return TensorDict({
            "observation": self._get_observation(new_storage),
            "reward": reward,
            "done": False,
        })
```

## 8. Advanced Tips

### 1. **Checkpointing**

```python
from torchrl.trainers import Trainer

trainer = Trainer(
    collector=collector,
    loss_module=loss_module,
    optimizer=optimizer,
    logger=logger,
    checkpoint_interval=10000,
)
```

### 2. **Hyperparameter Tuning**

- Use `hydra` or `ray[tune]` for systematic search
- Track experiments with `wandb` or `tensorboard`

### 3. **Distributed Training**

```python
from torchrl.envs import ParallelEnv
from torchrl.collectors import MultiSyncDataCollector

# Scale data collection
collector = MultiSyncDataCollector(
    [env_fn] * num_workers,
    policy=policy,
    storing_device="cpu",
    cat_results="stack",
)
```

### 4. **Custom Transforms**

```python
from torchrl.envs.transforms import Transform

class ReservoirNormalizer(Transform):
    def __init__(self, capacity):
        super().__init__(in_keys=["storage"], out_keys=["storage"])
        self.capacity = capacity
        
    def _apply_transform(self, storage):
        return storage / self.capacity
```

## Resources and References

1. **Official Documentation**: <https://pytorch.org/rl/>
2. **GitHub Examples**: <https://github.com/pytorch/rl/tree/main/examples>
3. **Community Tutorials**: TorchRL tutorials on PyTorch website
4. **Paper**: "TorchRL: A data-driven decision-making library for PyTorch"

## Getting Started Example

See `hello_torch.py` in the project for a minimal working example that demonstrates:

- Environment setup with Gymnasium
- Policy creation
- Data collection
- Basic training loop
