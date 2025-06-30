# Issue 6: DQN Training Components Implementation

## Overview
Implemented DQN training components using TorchRL, including loss module, exploration, replay buffer, and target network updates.

## Key Implementation Details

### 1. DQN Loss Module
- Uses TorchRL's `DQNLoss` with configurable parameters
- Supports smooth L1 and L2 loss functions
- Includes proper gamma setting via `make_value_estimator()`
- Enables target network usage with `delay_value=True`

### 2. Q-Value Actor
- Implemented `create_qvalue_actor()` using TorchRL's `QValueActor`
- Directly wraps Q-network without manual TensorDictModule
- Handles action selection based on Q-values
- Compatible with exploration modules

### 3. Exploration Module
- Epsilon-greedy exploration with linear decay
- Configurable start/end epsilon and decay steps
- Works seamlessly with QValueActor

### 4. Target Network Updates
- Supports both soft and hard updates
- Soft updates use TorchRL's eps parameter (tau equivalent)
- Updates applied to loss module directly

### 5. Replay Buffer
- Uses `TensorDictReplayBuffer` with lazy memory mapping
- Configurable size and batch size
- Supports device placement and prefetching

## Important Lessons Learned

### Tensor Shape Requirements
- Rewards and done/terminated flags must have shape `[1]` not `[]` when adding to replay buffer
- This ensures proper batching: `[batch_size, 1]` instead of `[batch_size]`
- Required for `td0_return_estimate` to compute TD targets correctly

### API Differences
- `QValueActor` requires module as first positional argument
- TorchRL uses `eps` parameter for soft updates (not `tau`)
- `TensorDictModule` should be imported from `tensordict.nn`

## Testing Coverage
- All components have comprehensive unit tests
- Integration test verifies components work together
- Tests cover custom parameters and edge cases
- Device placement tests ensure GPU compatibility

## Usage Example
```python
from rl_demo.models.qnet import create_qnet
from rl_demo.trainers.dqn_components import create_dqn_training_components
from torchrl.data import OneHot

# Create Q-network
qnet = create_qnet()

# Define action space
action_spec = OneHot(11)  # 11 discrete actions

# Create all components
components = create_dqn_training_components(
    qnet=qnet,
    action_spec=action_spec,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay_steps=10000,
    buffer_size=10000,
    batch_size=32,
    tau=0.995,
)

# Access components
actor = components["actor"]
loss_module = components["loss_module"]
exploration = components["exploration_module"]
replay_buffer = components["replay_buffer"]
target_updater = components["target_updater"]
```

## Integration with Training Loop
The components are designed to work together in a standard DQN training loop:
1. Use `actor` with `exploration_module` for action selection during training
2. Store transitions in `replay_buffer` with proper tensor shapes
3. Sample batches and compute loss with `loss_module`
4. Update target network periodically with `target_updater.step()`