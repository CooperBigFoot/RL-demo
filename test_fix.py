#!/usr/bin/env python3
"""Test script to verify the dimension mismatch fix."""

import torch
from rl_demo.envs.discrete_reservoir_env import DiscreteReservoirEnv
from rl_demo.utils.logger import ReservoirLogger

def test_episode_logging():
    """Test that episode logging works without dimension mismatches."""
    
    # Create environment and logger
    env = DiscreteReservoirEnv(device="cpu")
    logger = ReservoirLogger("test_runs/dimension_fix")
    
    # Run a single episode
    td = env.reset()
    
    # Track episode data
    episode_rewards = []
    episode_storages = []
    episode_actions = []
    episode_inflows = []
    episode_demands = []
    violations = {"flood": 0, "drought": 0, "env_flow": 0}
    
    done = False
    steps = 0
    max_steps = 50  # Limit steps for testing
    
    while not done and steps < max_steps:
        # Get observation components
        obs = td["observation"]
        obs_dict = DiscreteReservoirEnv.parse_observation(obs)
        
        # Track pre-step data
        episode_storages.append(obs_dict["volume_pct"])
        
        # Random action
        action = torch.randint(0, 11, (1,))
        td["action"] = action
        episode_actions.append(action.item())
        
        # Step environment
        td = env.step(td)
        
        # Get reward
        reward = td["reward"].item()
        episode_rewards.append(reward)
        
        # Check violations
        if obs_dict["volume_pct"] > 0.9:
            violations["flood"] += 1
        elif obs_dict["volume_pct"] < 0.1:
            violations["drought"] += 1
        
        done = td["done"].item()
        steps += 1
    
    # Since we don't have actual inflow/demand data, use placeholders
    episode_inflows = [0.0] * len(episode_rewards)
    episode_demands = [0.0] * len(episode_rewards)
    
    # Log episode - storage_levels should match rewards length
    print(f"Episode lengths - Rewards: {len(episode_rewards)}, Storage: {len(episode_storages)}, Actions: {len(episode_actions)}")
    
    try:
        logger.log_episode(
            rewards=episode_rewards,
            storage_levels=episode_storages[:len(episode_rewards)],  # Trim to match
            actions=episode_actions,
            inflows=episode_inflows,
            demands=episode_demands,
            violations=violations
        )
        print("✓ Episode logged successfully!")
    except ValueError as e:
        print(f"✗ Error logging episode: {e}")
        raise
    
    logger.close()
    print(f"Test completed. Check logs at: {logger.log_dir}")

if __name__ == "__main__":
    test_episode_logging()