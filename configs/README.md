# Configuration Management

This directory contains YAML configuration files for DQN experiments on the Discrete Reservoir Environment.

## Usage

### Basic Training with Default Config
```bash
uv run python scripts/train_dqn.py
```

### Training with a Config File
```bash
uv run python scripts/train_dqn.py --config configs/experiments/baseline.yaml
```

### Override Config Parameters from CLI
```bash
uv run python scripts/train_dqn.py --config configs/experiments/baseline.yaml --lr 0.001 --batch-size 64
```

### Resume Training from Checkpoint
```bash
uv run python scripts/train_dqn.py --resume checkpoints/experiment_id/checkpoint_10000.pt
```

## Available Configurations

- `baseline.yaml` - Standard DQN configuration for reservoir control
- `fast_learning.yaml` - Higher learning rate and exploration for faster convergence
- `deep_network.yaml` - Larger network capacity for complex patterns

## Configuration Structure

Each config file contains the following sections:

### Environment
```yaml
environment:
  name: DiscreteReservoir
  inflow_scenario: mixed  # Options: stable, seasonal, extreme, mixed
  v_max: 1000.0
  v_min: 100.0
  device: cpu
```

### Network Architecture
```yaml
network:
  hidden_dim: 256
  n_layers: 2
  activation: relu
  state_dim: 13
  action_dim: 11
```

### Training Hyperparameters
```yaml
training:
  total_frames: 50000
  frames_per_batch: 256
  batch_size: 32
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.01
  epsilon_frames: 10000
  target_update_freq: 500
  buffer_size: 10000
  min_replay_size: 1000
```

### Logging and Checkpointing
```yaml
logging:
  log_interval: 100
  eval_interval: 1000
  eval_episodes: 5
  checkpoint_interval: 5000
  log_dir: runs/dqn_reservoir
  checkpoint_dir: checkpoints
  save_best: true
```

### Experiment Tracking
```yaml
experiment:
  exp_name: dqn_reservoir
  seed: 42
  notes: "Description of the experiment"
  tags:
    - baseline
    - dqn
```

## Creating Custom Configurations

1. Copy an existing config file:
```bash
cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml
```

2. Modify parameters as needed

3. Run with your custom config:
```bash
uv run python scripts/train_dqn.py --config configs/experiments/my_experiment.yaml
```

## Experiment Tracking

Each experiment automatically generates:
- Unique experiment ID with timestamp
- Git commit hash for reproducibility
- Saved configuration in checkpoint directory
- TensorBoard logs in designated directory

## Command Line Arguments

All configuration parameters can be overridden via CLI:
- `--exp-name`: Override experiment name
- `--seed`: Set random seed
- `--device`: Choose cpu/cuda
- `--lr`: Learning rate
- `--batch-size`: Training batch size
- `--total-frames`: Total training frames
- `--debug`: Enable debug mode with frequent logging

See `uv run python scripts/train_dqn.py --help` for all options.