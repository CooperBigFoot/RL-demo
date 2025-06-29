# Phase 2: Agent Implementation Plan - TorchRL DQN

## Overview
Simple DQN implementation using TorchRL's pre-built components for discrete reservoir control (11 actions: 0-10% release).

## 1. DQN Architecture

### Network Structure
```python
from torchrl.modules import MLP, QValueModule

# Simple MLP for Q-network
q_net = MLP(
    in_features=13,           # State dimension
    out_features=11,          # Number of actions
    num_cells=[128, 128],     # Two hidden layers
    activation_class=nn.ReLU,
    activate_last_layer=False
)

# Wrap in QValueModule for TensorDict compatibility
qvalue_module = QValueModule(
    action_value_key="action_value",
    out_keys=["action_value", "chosen_action_value"],
    spec=env.action_spec,
    action_space="one-hot"    # For discrete actions
)
```

### Standard Hyperparameters
- Learning rate: 1e-3
- Discount factor (gamma): 0.99
- Epsilon start: 1.0
- Epsilon end: 0.01
- Epsilon decay: 10,000 steps

## 2. TorchRL Components

### DQNLoss Configuration
```python
from torchrl.objectives import DQNLoss

loss_module = DQNLoss(
    value_network=qvalue_module,
    loss_function="smooth_l1",
    delay_value=True,         # Use target network
    gamma=0.99,
    action_space="one-hot"
)
```

### Epsilon-Greedy Exploration
```python
from torchrl.modules import EGreedyModule

exploration_module = EGreedyModule(
    spec=env.action_spec,
    eps_init=1.0,
    eps_end=0.01,
    annealing_num_steps=10_000,
    action_key="action",
    action_value_key="action_value"
)
```

### Target Network Updates
```python
# Soft update every 100 steps
target_net_update_frequency = 100
tau = 0.005  # Soft update parameter

# In training loop:
if step % target_net_update_frequency == 0:
    target_net_updater.step()  # TorchRL's built-in updater
```

## 3. Network Design Details

### Input Processing
- **Input**: 13-dimensional state vector
  - Current storage level
  - Inflow (current + predictions)
  - Demand (current + forecast)
  - Time features (hour, day, season)
  - Operating constraints

### Output Structure
- **Output**: 11 Q-values
  - Each represents expected return for actions 0% to 10% release
  - Action selection: argmax(Q-values) during evaluation
  - Epsilon-greedy during training

### Hidden Layers
- **Layer 1**: 128 units, ReLU activation
- **Layer 2**: 128 units, ReLU activation
- **Output**: Linear (no activation)

## 4. Training Configuration

### Core Parameters
```python
config = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "buffer_size": 10_000,
    "min_buffer_size": 1_000,  # Start training after this
    "update_frequency": 4,       # Train every 4 steps
    "target_update_freq": 100,
    "gamma": 0.99,
    "grad_clip": 10.0
}
```

### Replay Buffer
```python
from torchrl.data import ReplayBuffer, LazyTensorStorage

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=10_000),
    sampler=SamplerWithoutReplacement(),
    batch_size=32
)
```

### Optimizer Setup
```python
optimizer = torch.optim.Adam(
    qvalue_module.parameters(),
    lr=1e-3
)
```

## 5. Evaluation Setup

### Greedy Policy
```python
# Disable exploration for evaluation
eval_policy = TensorDictSequential(
    qvalue_module,
    # No exploration module - pure greedy
)

# Or set epsilon to 0
exploration_module.set_eps(0.0)
```

### Performance Metrics
```python
metrics = {
    "episode_reward": [],      # Total reward per episode
    "constraint_violations": [], # Count of constraint violations
    "storage_variance": [],     # Stability metric
    "demand_satisfaction": [],  # % of demand met
    "avg_q_value": []          # Track Q-value estimates
}
```

### Evaluation Loop
```python
def evaluate(env, policy, num_episodes=10):
    metrics = defaultdict(list)
    
    for episode in range(num_episodes):
        td = env.reset()
        episode_reward = 0
        
        while not td["done"]:
            with torch.no_grad():
                td = policy(td)
            td = env.step(td)
            episode_reward += td["reward"].item()
            
        metrics["episode_reward"].append(episode_reward)
    
    return metrics
```

## Implementation Timeline

### Phase 1: Basic Setup (Day 1)
- Environment wrapper for TensorDict
- Q-network implementation
- Basic training loop

### Phase 2: Training Infrastructure (Day 2)
- Replay buffer integration
- Target network updates
- Logging and checkpointing

### Phase 3: Evaluation & Tuning (Day 3)
- Evaluation metrics
- Hyperparameter tuning
- Performance analysis

## Key Simplifications

1. **No Prioritized Replay**: Use uniform sampling initially
2. **Fixed Architecture**: Start with 2-layer MLP, tune later
3. **Standard DQN**: No double DQN or dueling architectures
4. **Simple Exploration**: Linear epsilon decay
5. **Basic Features**: Use raw state, no feature engineering

## Success Criteria

- Stable Q-value convergence
- Consistent constraint satisfaction (>95%)
- Improved reward over baseline policy
- Smooth control actions (minimal oscillation)

## Next Steps

1. Implement basic DQN agent
2. Integrate with reservoir environment
3. Run initial training experiments
4. Analyze results and iterate