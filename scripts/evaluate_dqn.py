#!/usr/bin/env python3
"""Evaluation script for trained DQN policies on reservoir control.

This script evaluates trained DQN models and compares them against baseline
policies. It computes comprehensive performance metrics including:
- Average episode reward
- Success rate (no floods/droughts)
- Hydropower efficiency
- Environmental flow compliance
- Storage variance
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from rl_demo.envs.discrete_reservoir_env import DiscreteReservoirEnv
from rl_demo.models.qnet import DQNValueNetwork


class BaselinePolicy:
    """Abstract base class for baseline policies."""
    
    def __init__(self, action_dim: int = 11):
        self.action_dim = action_dim
    
    def get_action(self, obs: TensorDict) -> torch.Tensor:
        """Get action given observation."""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Policy name for reporting."""
        raise NotImplementedError


class RandomPolicy(BaselinePolicy):
    """Random action selection policy."""
    
    def get_action(self, obs: TensorDict) -> torch.Tensor:
        batch_size = obs["observation"].shape[0] if obs["observation"].dim() > 1 else 1
        return torch.randint(0, self.action_dim, (batch_size,), device=obs.device)
    
    @property
    def name(self) -> str:
        return "Random"


class ConservativePolicy(BaselinePolicy):
    """Conservative policy - always release minimum (10%)."""
    
    def get_action(self, obs: TensorDict) -> torch.Tensor:
        batch_size = obs["observation"].shape[0] if obs["observation"].dim() > 1 else 1
        # Action 1 corresponds to 10% release
        return torch.ones(batch_size, dtype=torch.long, device=obs.device)
    
    @property
    def name(self) -> str:
        return "Conservative"


class AggressivePolicy(BaselinePolicy):
    """Aggressive policy - maximize hydropower based on demand."""
    
    def get_action(self, obs: TensorDict) -> torch.Tensor:
        # Extract current demand from observation (index 3)
        demand = obs["observation"][..., 3]
        
        # Scale demand to action space (0-100%)
        # Assume max demand corresponds to ~50% release for safety
        action = torch.clamp((demand * 5).long(), 0, self.action_dim - 1)
        
        if action.dim() == 0:
            action = action.unsqueeze(0)
        
        return action
    
    @property
    def name(self) -> str:
        return "Aggressive"


class DQNPolicy:
    """Wrapper for trained DQN model."""
    
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.name = "DQN"
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Initialize Q-network
        self.q_net = DQNValueNetwork(
            state_dim=13,
            action_dim=11,
            hidden_dim=256,
            device=device
        )
        
        # Load weights
        self.q_net.qnet.load_state_dict(checkpoint["q_net_state_dict"])
        self.q_net.eval()
    
    def get_action(self, obs: TensorDict) -> torch.Tensor:
        """Get greedy action from Q-network."""
        with torch.no_grad():
            state = obs["observation"]
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            q_values = self.q_net(state)
            action = q_values.argmax(dim=-1)
            
            return action.squeeze()


def evaluate_policy(
    env: DiscreteReservoirEnv,
    policy: BaselinePolicy | DQNPolicy,
    n_episodes: int = 100,
    seed: int = 0,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Evaluate a policy over multiple episodes.
    
    Args:
        env: Reservoir environment
        policy: Policy to evaluate
        n_episodes: Number of evaluation episodes
        seed: Random seed for reproducibility
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary of metrics collected during evaluation
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    metrics = defaultdict(list)
    
    # Episode-level metrics
    episode_rewards = []
    episode_lengths = []
    flood_counts = []
    drought_counts = []
    
    # Step-level metrics for aggregation
    all_storages = []
    all_actions = []
    all_hydropower = []
    all_env_violations = []
    
    iterator = tqdm(range(n_episodes), desc=f"Evaluating {policy.name}") if verbose else range(n_episodes)
    
    for episode in iterator:
        td = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_floods = 0
        episode_droughts = 0
        episode_storages = []
        episode_actions = []
        episode_hydropower = []
        episode_env_violations = []
        
        done = False
        while not done:
            # Get action from policy
            action = policy.get_action(td)
            td["action"] = action
            
            # Step environment
            td = env.step(td)
            
            # Collect metrics
            episode_reward += td["next"]["reward"].item()
            episode_length += 1
            
            # Extract info from observation
            storage_level = td["next"]["observation"][0].item()  # Current storage
            episode_storages.append(storage_level)
            episode_actions.append(action.item())
            
            # Check for violations
            if storage_level >= 0.95:  # Near maximum capacity
                episode_floods += 1
            elif storage_level <= 0.15:  # Near minimum
                episode_droughts += 1
            
            # Estimate hydropower (proportional to release and head)
            release_fraction = action.item() * 0.1  # Convert to percentage (0%, 10%, 20%, ..., 100%)
            hydropower = release_fraction * storage_level  # Simplified
            episode_hydropower.append(hydropower)
            
            # Check environmental flow (assume min 1% required)
            if release_fraction < 0.01:
                episode_env_violations.append(1)
            else:
                episode_env_violations.append(0)
            
            # Update for next iteration
            td = td["next"]
            done = td["done"].any().item()
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        flood_counts.append(episode_floods)
        drought_counts.append(episode_droughts)
        
        # Store step-level data
        all_storages.extend(episode_storages)
        all_actions.extend(episode_actions)
        all_hydropower.extend(episode_hydropower)
        all_env_violations.extend(episode_env_violations)
    
    # Compute aggregate metrics
    metrics["avg_episode_reward"] = np.mean(episode_rewards)
    metrics["std_episode_reward"] = np.std(episode_rewards)
    metrics["avg_episode_length"] = np.mean(episode_lengths)
    metrics["success_rate"] = np.mean([f == 0 and d == 0 for f, d in zip(flood_counts, drought_counts)])
    metrics["flood_rate"] = np.mean([f > 0 for f in flood_counts])
    metrics["drought_rate"] = np.mean([d > 0 for d in drought_counts])
    metrics["avg_storage_level"] = np.mean(all_storages)
    metrics["storage_variance"] = np.var(all_storages)
    metrics["avg_hydropower"] = np.mean(all_hydropower)
    metrics["env_compliance_rate"] = 1.0 - np.mean(all_env_violations)
    
    # Action distribution
    action_counts = np.bincount(all_actions, minlength=11)
    metrics["action_distribution"] = action_counts / len(all_actions)
    
    # Store raw data for plotting
    metrics["episode_rewards"] = episode_rewards
    metrics["all_storages"] = all_storages[:1000]  # Limit for plotting
    metrics["all_actions"] = all_actions[:1000]
    
    return dict(metrics)


def plot_evaluation_results(results: Dict[str, Dict], output_dir: Path):
    """Generate evaluation plots.
    
    Args:
        results: Dictionary mapping policy names to metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Episode rewards comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for policy_name, metrics in results.items():
        rewards = metrics["episode_rewards"]
        ax.plot(rewards, alpha=0.7, label=f"{policy_name} (μ={np.mean(rewards):.1f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Rewards Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "episode_rewards.png", dpi=150)
    plt.close()
    
    # 2. Action distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (policy_name, metrics) in enumerate(results.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        action_dist = metrics["action_distribution"]
        actions = np.arange(11)
        ax.bar(actions, action_dist)
        ax.set_xlabel("Action (% Release)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{policy_name} Action Distribution")
        ax.set_xticks(actions)
        ax.set_xticklabels([f"{i*10}%" for i in actions])
    
    plt.tight_layout()
    plt.savefig(output_dir / "action_distributions.png", dpi=150)
    plt.close()
    
    # 3. Performance metrics comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics_to_plot = [
        "avg_episode_reward",
        "success_rate",
        "flood_rate",
        "drought_rate",
        "avg_hydropower",
        "env_compliance_rate",
        "storage_variance"
    ]
    
    policy_names = list(results.keys())
    n_policies = len(policy_names)
    n_metrics = len(metrics_to_plot)
    
    x = np.arange(n_metrics)
    width = 0.8 / n_policies
    
    for i, policy_name in enumerate(policy_names):
        values = [results[policy_name].get(metric, 0) for metric in metrics_to_plot]
        ax.bar(x + i * width, values, width, label=policy_name)
    
    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Comparison")
    ax.set_xticks(x + width * (n_policies - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_to_plot], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=150)
    plt.close()
    
    # 4. Storage trajectory sample
    fig, ax = plt.subplots(figsize=(12, 6))
    for policy_name, metrics in results.items():
        storages = metrics["all_storages"]
        ax.plot(storages, alpha=0.7, label=policy_name)
    
    ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Flood threshold')
    ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Drought threshold')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Storage Level")
    ax.set_title("Sample Storage Trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "storage_trajectories.png", dpi=150)
    plt.close()


def generate_report(results: Dict[str, Dict], output_path: Path):
    """Generate evaluation report.
    
    Args:
        results: Dictionary mapping policy names to metrics
        output_path: Path to save report
    """
    report = {
        "summary": {},
        "detailed_metrics": {}
    }
    
    # Summary statistics
    for policy_name, metrics in results.items():
        summary = {
            "average_reward": float(metrics["avg_episode_reward"]),
            "reward_std": float(metrics["std_episode_reward"]),
            "success_rate": float(metrics["success_rate"]),
            "flood_rate": float(metrics["flood_rate"]),
            "drought_rate": float(metrics["drought_rate"]),
            "hydropower_efficiency": float(metrics["avg_hydropower"]),
            "environmental_compliance": float(metrics["env_compliance_rate"]),
            "storage_stability": float(1.0 / (1.0 + metrics["storage_variance"]))
        }
        report["summary"][policy_name] = summary
    
    # Detailed metrics (excluding raw data)
    for policy_name, metrics in results.items():
        detailed = {k: v for k, v in metrics.items() 
                   if k not in ["episode_rewards", "all_storages", "all_actions", "action_distribution"]}
        detailed["action_distribution"] = metrics["action_distribution"].tolist()
        report["detailed_metrics"][policy_name] = detailed
    
    # Save report
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Policy':<15} {'Avg Reward':<12} {'Success %':<12} {'Flood %':<10} {'Drought %':<12} {'Env Comply %':<15}")
    print("-"*80)
    
    for policy_name, summary in report["summary"].items():
        print(f"{policy_name:<15} "
              f"{summary['average_reward']:<12.2f} "
              f"{summary['success_rate']*100:<12.1f} "
              f"{summary['flood_rate']*100:<10.1f} "
              f"{summary['drought_rate']*100:<12.1f} "
              f"{summary['environmental_compliance']*100:<15.1f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN policies for reservoir control")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best/checkpoint_20000.pt",
                        help="Path to DQN checkpoint")
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline policy evaluation")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {device}")
    print(f"Evaluating with {args.n_episodes} episodes\n")
    
    # Initialize environment
    env = DiscreteReservoirEnv(device=device)
    
    # Initialize policies
    policies = {}
    
    # Load DQN policy if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading DQN checkpoint from: {checkpoint_path}")
        policies["DQN"] = DQNPolicy(str(checkpoint_path), device)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Evaluating only baseline policies")
    
    # Add baseline policies unless disabled
    if not args.no_baseline:
        policies["Random"] = RandomPolicy()
        policies["Conservative"] = ConservativePolicy()
        policies["Aggressive"] = AggressivePolicy()
    
    # Evaluate all policies
    results = {}
    for policy_name, policy in policies.items():
        print(f"\nEvaluating {policy_name} policy...")
        metrics = evaluate_policy(env, policy, args.n_episodes, args.seed)
        results[policy_name] = metrics
    
    # Generate plots
    print("\nGenerating plots...")
    plot_evaluation_results(results, output_dir)
    
    # Generate report
    print("Generating report...")
    report_path = output_dir / "evaluation_report.json"
    generate_report(results, report_path)
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()