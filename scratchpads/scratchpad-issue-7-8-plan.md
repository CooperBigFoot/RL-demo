# Implementation Plan: Issues 7 & 8 - DQN Training Script and TensorBoard Logging

## Overview
Implementing a minimal DQN training script with comprehensive logging for the DiscreteReservoirEnv using existing TorchRL components.

## Current State Analysis
### ✅ Already Implemented:
- `DiscreteReservoirEnv`: Complete TorchRL environment with discrete actions
- `dqn_components.py`: All DQN training components (loss, exploration, replay buffer)
- `qnet.py`: Q-network architecture
- Comprehensive test coverage

### ❌ Need to Implement:
- Main training script (`scripts/train_dqn.py`)
- Logger module (`src/rl_demo/utils/logger.py`)
- Configuration system
- Training orchestration

## Implementation Approach

### Issue 7: Training Script
1. **Structure**: Single file `scripts/train_dqn.py` with clear sections
2. **Key Components**:
   - Environment creation with proper configuration
   - DQN component initialization using existing functions
   - Training loop with TorchRL's SyncDataCollector
   - Checkpointing every 5000 steps
   - Basic console logging initially

### Issue 8: TensorBoard Logging
1. **Logger Module**: `src/rl_demo/utils/logger.py`
2. **Metrics to Track**:
   - **Training**: Loss, epsilon, learning rate
   - **Episode**: Rewards (mean/min/max), length, success rate
   - **Reservoir-specific**: Storage levels, hydropower generation, constraint violations
   - **Agent**: Q-value statistics, action distribution

3. **Visualization Features**:
   - Episode trajectory plots
   - Storage level over time
   - Action histograms
   - Reward component breakdown

## Code Structure

### Training Script Flow
```python
# scripts/train_dqn.py
1. Parse arguments / load config
2. Create environment
3. Initialize DQN components (using existing functions)
4. Setup logger
5. Create SyncDataCollector
6. Training loop:
   - Collect experience
   - Add to replay buffer
   - Sample and train
   - Update target network
   - Log metrics
   - Save checkpoints
```

### Logger Architecture
```python
# src/rl_demo/utils/logger.py
class ReservoirLogger:
    - TensorBoard integration
    - Episode tracking
    - Training metrics
    - Reservoir-specific metrics
    - Visualization helpers
```

## Implementation Steps

1. **Create feature branch**: `fix/issue-7-8-training-and-logging`
2. **Implement basic training script** without logging
3. **Verify training works** with minimal configuration
4. **Implement logger module** with TensorBoard support
5. **Integrate logger** into training script
6. **Add reservoir-specific metrics**
7. **Test complete pipeline**
8. **Create PR** linking both issues

## Configuration Approach
- Start with hardcoded values in script
- Add argparse for key hyperparameters
- Future: YAML config support

## Testing Strategy
- Manual test with reduced frames (5000) for quick validation
- Verify all metrics appear in TensorBoard
- Check checkpoint saving/loading
- Ensure no memory leaks during long training

## Success Criteria
- ✅ Training runs without errors
- ✅ Loss decreases over time
- ✅ Agent shows learning progress
- ✅ All metrics visible in TensorBoard
- ✅ Checkpoints saved correctly
- ✅ Can resume from checkpoint