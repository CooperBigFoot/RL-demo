"""
TensorBoard logging utilities for Reservoir RL training.

This module provides comprehensive logging and visualization capabilities
for monitoring DQN training on the Discrete Reservoir Environment.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import torch
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict


class ReservoirLogger:
    """Logger for tracking training progress and reservoir-specific metrics."""
    
    def __init__(self, log_dir: str = "runs/dqn_reservoir"):
        """Initialize the logger with TensorBoard writer.
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        # Create unique run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / timestamp
        self.writer = SummaryWriter(self.log_dir)
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        
        # Buffers for aggregating metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_metrics = []
        
        print(f"Logging to: {self.log_dir}")
    
    def log_training_step(self, 
                         loss: float,
                         epsilon: float,
                         learning_rate: float,
                         buffer_size: int,
                         q_values: Optional[torch.Tensor] = None):
        """Log training metrics for a single optimization step.
        
        Args:
            loss: TD loss value
            epsilon: Current exploration epsilon
            learning_rate: Current learning rate
            buffer_size: Current replay buffer size
            q_values: Optional Q-values for the batch
        """
        self.writer.add_scalar("train/loss", loss, self.step_count)
        self.writer.add_scalar("train/epsilon", epsilon, self.step_count)
        self.writer.add_scalar("train/learning_rate", learning_rate, self.step_count)
        self.writer.add_scalar("train/buffer_size", buffer_size, self.step_count)
        
        if q_values is not None:
            self.writer.add_scalar("train/q_mean", q_values.mean().item(), self.step_count)
            self.writer.add_scalar("train/q_max", q_values.max().item(), self.step_count)
            self.writer.add_scalar("train/q_min", q_values.min().item(), self.step_count)
            self.writer.add_histogram("train/q_values", q_values, self.step_count)
    
    def log_episode(self,
                   rewards: List[float],
                   storage_levels: List[float],
                   actions: List[int],
                   inflows: List[float],
                   demands: List[float],
                   violations: Dict[str, int]):
        """Log comprehensive episode metrics.
        
        Args:
            rewards: List of rewards for each timestep
            storage_levels: List of storage levels (fraction of capacity)
            actions: List of actions taken (0-10 for release percentages)
            inflows: List of inflow values
            demands: List of demand values
            violations: Dictionary with counts of constraint violations
        """
        # Episode summary statistics
        total_reward = sum(rewards)
        episode_length = len(rewards)
        
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # Basic episode metrics
        self.writer.add_scalar("episode/total_reward", total_reward, self.episode_count)
        self.writer.add_scalar("episode/length", episode_length, self.episode_count)
        self.writer.add_scalar("episode/avg_reward_per_step", total_reward / episode_length, self.episode_count)
        
        # Storage statistics
        storage_array = np.array(storage_levels)
        self.writer.add_scalar("reservoir/avg_storage", storage_array.mean(), self.episode_count)
        self.writer.add_scalar("reservoir/min_storage", storage_array.min(), self.episode_count)
        self.writer.add_scalar("reservoir/max_storage", storage_array.max(), self.episode_count)
        self.writer.add_scalar("reservoir/storage_variance", storage_array.var(), self.episode_count)
        
        # Action statistics
        action_array = np.array(actions)
        self.writer.add_scalar("actions/mean_release", action_array.mean() * 10, self.episode_count)  # Convert to percentage
        self.writer.add_histogram("actions/distribution", action_array, self.episode_count)
        
        # Constraint violations
        total_violations = sum(violations.values())
        self.writer.add_scalar("violations/total", total_violations, self.episode_count)
        self.writer.add_scalar("violations/flood", violations.get("flood", 0), self.episode_count)
        self.writer.add_scalar("violations/drought", violations.get("drought", 0), self.episode_count)
        self.writer.add_scalar("violations/env_flow", violations.get("env_flow", 0), self.episode_count)
        
        # Hydropower estimation (simplified)
        # Assuming hydropower is proportional to release * storage
        releases = action_array * 0.1  # Convert to fraction
        # Ensure arrays have same length
        storage_for_hydro = storage_array[:len(releases)] if len(storage_array) > len(releases) else storage_array
        hydropower = releases * storage_for_hydro
        total_hydropower = hydropower.sum()
        self.writer.add_scalar("reservoir/total_hydropower", total_hydropower, self.episode_count)
        
        # Create episode trajectory plot every 10 episodes
        if self.episode_count % 10 == 0:
            self._plot_episode_trajectory(
                storage_levels, actions, rewards, inflows, demands
            )
        
        # Running statistics (last 100 episodes)
        if len(self.episode_rewards) >= 100:
            recent_rewards = self.episode_rewards[-100:]
            self.writer.add_scalar("episode/reward_mean_100", np.mean(recent_rewards), self.episode_count)
            self.writer.add_scalar("episode/reward_std_100", np.std(recent_rewards), self.episode_count)
        
        self.episode_count += 1
    
    def log_evaluation(self,
                      eval_rewards: List[float],
                      eval_lengths: List[float],
                      eval_violations: List[Dict[str, int]]):
        """Log evaluation metrics from multiple episodes.
        
        Args:
            eval_rewards: List of total rewards from evaluation episodes
            eval_lengths: List of episode lengths
            eval_violations: List of violation dictionaries from each episode
        """
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
        mean_length = np.mean(eval_lengths)
        
        self.writer.add_scalar("eval/mean_reward", mean_reward, self.step_count)
        self.writer.add_scalar("eval/std_reward", std_reward, self.step_count)
        self.writer.add_scalar("eval/mean_length", mean_length, self.step_count)
        
        # Aggregate violations
        total_floods = sum(v.get("flood", 0) for v in eval_violations)
        total_droughts = sum(v.get("drought", 0) for v in eval_violations)
        total_env_violations = sum(v.get("env_flow", 0) for v in eval_violations)
        
        self.writer.add_scalar("eval/total_floods", total_floods, self.step_count)
        self.writer.add_scalar("eval/total_droughts", total_droughts, self.step_count)
        self.writer.add_scalar("eval/total_env_violations", total_env_violations, self.step_count)
    
    def _plot_episode_trajectory(self,
                                storage_levels: List[float],
                                actions: List[int],
                                rewards: List[float],
                                inflows: List[float],
                                demands: List[float]):
        """Create a detailed plot of episode trajectory.
        
        Args:
            storage_levels: Storage levels throughout episode
            actions: Actions taken
            rewards: Rewards received
            inflows: Inflow values
            demands: Demand values
        """
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        timesteps = range(len(rewards))
        
        # Storage levels with safety bounds
        ax = axes[0]
        # Ensure storage_levels matches timesteps length
        storage_to_plot = storage_levels[:len(timesteps)] if len(storage_levels) > len(timesteps) else storage_levels
        ax.plot(timesteps, storage_to_plot, 'b-', label='Storage Level')
        ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Flood Level')
        ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Drought Level')
        ax.set_ylabel('Storage (fraction)')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Actions (release percentages)
        ax = axes[1]
        release_pct = [a * 10 for a in actions]  # Convert to percentage
        ax.bar(timesteps, release_pct, alpha=0.7, color='green')
        ax.set_ylabel('Release (%)')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Rewards
        ax = axes[2]
        ax.plot(timesteps, rewards, 'g-', alpha=0.7)
        ax.set_ylabel('Reward')
        ax.grid(True, alpha=0.3)
        
        # Inflows and Demands
        ax = axes[3]
        # Handle case where inflows/demands might be placeholders or have wrong length
        inflows_to_plot = inflows[:len(timesteps)] if len(inflows) > len(timesteps) else inflows
        demands_to_plot = demands[:len(timesteps)] if len(demands) > len(timesteps) else demands
        
        # Only plot if we have actual data (not all zeros)
        if any(v != 0 for v in inflows_to_plot):
            ax.plot(timesteps, inflows_to_plot, 'c-', label='Inflow', alpha=0.7)
        if any(v != 0 for v in demands_to_plot):
            ax.plot(timesteps, demands_to_plot, 'm-', label='Demand', alpha=0.7)
        
        ax.set_ylabel('Flow')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cumulative reward
        ax = axes[4]
        cumulative_rewards = np.cumsum(rewards)
        ax.plot(timesteps, cumulative_rewards, 'b-')
        ax.set_ylabel('Cumulative Reward')
        ax.set_xlabel('Time Step')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Episode {self.episode_count} Trajectory')
        plt.tight_layout()
        
        # Save to tensorboard
        self.writer.add_figure('episode/trajectory', fig, self.episode_count)
        plt.close(fig)
    
    def log_action_values(self, 
                         observation: torch.Tensor,
                         q_values: torch.Tensor,
                         action: int):
        """Log Q-values for all actions given an observation.
        
        Args:
            observation: Current observation
            q_values: Q-values for all actions
            action: Selected action
        """
        # Log Q-value distribution
        self.writer.add_histogram("q_values/all_actions", q_values, self.step_count)
        
        # Log individual action Q-values
        for i, q_val in enumerate(q_values):
            self.writer.add_scalar(f"q_values/action_{i}", q_val.item(), self.step_count)
        
        # Log selected action Q-value
        self.writer.add_scalar("q_values/selected_action_value", q_values[action].item(), self.step_count)
        
        # Log action selection
        self.writer.add_scalar("actions/selected", action, self.step_count)
    
    def step(self):
        """Increment the global step counter."""
        self.step_count += 1
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
    
    def log_hyperparameters(self, hparams: Dict):
        """Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        # Convert all values to strings for TensorBoard
        hparams_str = {k: str(v) for k, v in hparams.items()}
        
        # Log hyperparameters with dummy metrics (will be updated during training)
        self.writer.add_hparams(
            hparams_str,
            {"hparam/final_reward": 0}  # Will be updated at the end
        )
    
    def update_hyperparameter_metrics(self, final_reward: float):
        """Update hyperparameter metrics at the end of training.
        
        Args:
            final_reward: Final evaluation reward
        """
        # This updates the hyperparameter metrics
        self.writer.add_scalar("hparam/final_reward", final_reward, self.step_count)