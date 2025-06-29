# Phase 3: DQN Training Pipeline Implementation Plan

## Overview
Minimal implementation plan for integrating TorchRL's training patterns with our DQN agent and CartPole environment.

## 1. Training Pipeline Structure

### Main Training Script (`scripts/train_dqn.py`)
```python
# Core structure
def main():
    # 1. Load config
    config = load_config()
    
    # 2. Setup environment
    env = make_env(config.env_name)
    
    # 3. Create DQN agent
    agent = DQNAgent(env.observation_space, env.action_space, config)
    
    # 4. Setup data collector
    collector = SyncDataCollector(
        env,
        agent.policy,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
    )
    
    # 5. Setup replay buffer
    replay_buffer = create_replay_buffer(config)
    
    # 6. Training loop
    trainer = DQNTrainer(agent, collector, replay_buffer, config)
    trainer.train()
```

### Key Components
- Single file initially: `train_dqn.py`
- Later split into: `trainer.py`, `utils.py`, `config.py`
- Use TorchRL's patterns but keep our existing DQN implementation

## 2. Data Collection

### SyncDataCollector Setup
```python
from torchrl.collectors import SyncDataCollector
from torchrl.envs import GymEnv

# Wrap environment
env = GymEnv("CartPole-v1")

# Create collector
collector = SyncDataCollector(
    env,
    policy=agent.exploration_policy,  # ε-greedy wrapper
    frames_per_batch=256,             # Collect 256 frames per iteration
    total_frames=100_000,             # Total training frames
    device="cpu",                     # Keep on CPU for CartPole
)
```

### Experience Collection Pattern
```python
for i, batch in enumerate(collector):
    # Store in replay buffer
    replay_buffer.extend(batch)
    
    # Train when buffer has enough samples
    if len(replay_buffer) >= config.min_replay_size:
        train_batch = replay_buffer.sample(config.batch_size)
        loss = agent.train_step(train_batch)
        
    # Update epsilon
    agent.update_epsilon(i / config.total_frames)
```

## 3. Logging and Monitoring

### Simple Tensorboard Setup
```python
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir="runs/dqn_cartpole"):
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []
        
    def log_training(self, step, loss, epsilon):
        self.writer.add_scalar("train/loss", loss, step)
        self.writer.add_scalar("train/epsilon", epsilon, step)
        
    def log_episode(self, step, reward, length):
        self.writer.add_scalar("episode/reward", reward, step)
        self.writer.add_scalar("episode/length", length, step)
```

### Key Metrics to Track
- Training loss (TD error)
- Episode rewards (running average)
- Episode lengths
- Epsilon decay
- Q-value estimates
- Buffer size

### Checkpointing
```python
def save_checkpoint(agent, step, checkpoint_dir):
    torch.save({
        'step': step,
        'model_state': agent.q_network.state_dict(),
        'target_state': agent.target_network.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'epsilon': agent.epsilon,
    }, f"{checkpoint_dir}/checkpoint_{step}.pt")
```

## 4. Configuration

### Simple Config Dataclass
```python
from dataclasses import dataclass

@dataclass
class DQNConfig:
    # Environment
    env_name: str = "CartPole-v1"
    
    # Training
    total_frames: int = 100_000
    frames_per_batch: int = 256
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # DQN specific
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10_000
    target_update_freq: int = 500
    
    # Replay buffer
    buffer_size: int = 10_000
    min_replay_size: int = 1000
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
```

### YAML Config Option
```yaml
# config/dqn_cartpole.yaml
env:
  name: CartPole-v1

training:
  total_frames: 100000
  frames_per_batch: 256
  batch_size: 32
  learning_rate: 0.001

dqn:
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_decay: 10000
  target_update_freq: 500
```

## 5. Running Experiments

### Training Command
```bash
# Basic training
uv run python scripts/train_dqn.py

# With custom config
uv run python scripts/train_dqn.py --config config/dqn_cartpole.yaml

# With specific hyperparameters
uv run python scripts/train_dqn.py --lr 0.001 --batch-size 64 --gamma 0.95
```

### Evaluation Pattern
```python
def evaluate(agent, env, num_episodes=10):
    """Run evaluation episodes without exploration."""
    rewards = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs, explore=False)  # Greedy action
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)
```

### Results Visualization
```python
# During training: Tensorboard
tensorboard --logdir runs/

# Post-training: Simple plots
def plot_training_curves(log_dir):
    # Load tensorboard logs
    # Plot reward curves
    # Save as PNG
```

## Implementation Steps

1. **Start Simple** (Day 1)
   - Basic training script with hardcoded config
   - Minimal logging (print statements)
   - Verify DQN learns CartPole

2. **Add TorchRL Integration** (Day 2)
   - Replace manual collection with SyncDataCollector
   - Use TorchRL's replay buffer
   - Add proper device handling

3. **Enhance Logging** (Day 3)
   - Add tensorboard logging
   - Implement checkpointing
   - Add evaluation runs

4. **Configuration & CLI** (Day 4)
   - Create config dataclass/yaml
   - Add command-line arguments
   - Support hyperparameter sweeps

## Next Steps
- Implement `scripts/train_dqn.py` following this plan
- Test with CartPole-v1 first
- Gradually add features while maintaining simplicity
- Document any TorchRL-specific quirks encountered