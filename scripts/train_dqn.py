#!/usr/bin/env python3
"""
DQN training script for the Discrete Reservoir Environment.

This script implements the main training loop using TorchRL's SyncDataCollector
and the DQN components from the rl_demo package.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs import TransformedEnv
from tensordict import TensorDict

from rl_demo.envs.discrete_reservoir_env import DiscreteReservoirEnv
from rl_demo.trainers.dqn_components import (
    create_dqn_training_components,
    create_replay_buffer,
)
from rl_demo.models.qnet import create_qnet
from rl_demo.utils.logger import ReservoirLogger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN on Discrete Reservoir Environment")
    
    # Environment
    parser.add_argument("--inflow-scenario", type=str, default="mixed",
                        choices=["stable", "seasonal", "extreme", "mixed"],
                        help="Inflow scenario for the reservoir")
    
    # Training
    parser.add_argument("--total-frames", type=int, default=50000,
                        help="Total number of frames to train for")
    parser.add_argument("--frames-per-batch", type=int, default=256,
                        help="Number of frames to collect per batch")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    
    # DQN specific
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0,
                        help="Starting epsilon for exploration")
    parser.add_argument("--epsilon-end", type=float, default=0.01,
                        help="Final epsilon for exploration")
    parser.add_argument("--epsilon-frames", type=int, default=10000,
                        help="Number of frames for epsilon decay")
    parser.add_argument("--target-update-freq", type=int, default=500,
                        help="Frequency of target network updates")
    
    # Replay buffer
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Size of the replay buffer")
    parser.add_argument("--min-replay-size", type=int, default=1000,
                        help="Minimum replay buffer size before training")
    
    # Logging and checkpointing
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Logging interval (in frames)")
    parser.add_argument("--eval-interval", type=int, default=1000,
                        help="Evaluation interval (in frames)")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                        help="Checkpoint saving interval (in frames)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/dqn_reservoir",
                        help="Directory for TensorBoard logs")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    
    return parser.parse_args()


def create_env(inflow_scenario="mixed", device="cpu"):
    """Create the discrete reservoir environment."""
    # Note: DiscreteReservoirEnv uses default reservoir parameters
    # and generates scenarios internally
    env = DiscreteReservoirEnv(device=device)
    # The inflow_scenario is handled by the ReservoirSimulator internally
    return env


def evaluate(policy_module, env, num_episodes=5, logger=None):
    """Evaluate the policy without exploration."""
    rewards = []
    episode_lengths = []
    violations_list = []
    
    for _ in range(num_episodes):
        td = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Track episode data for logging
        episode_rewards = []
        episode_storages = []
        episode_actions = []
        episode_inflows = []
        episode_demands = []
        violations = {"flood": 0, "drought": 0, "env_flow": 0}
        
        while not td.get("done", False).any():
            # Get action without exploration
            with torch.no_grad():
                td = policy_module(td)
            
            # Log pre-step data
            if logger:
                obs = td["observation"]
                episode_storages.append(obs[0].item())  # Storage level (volume_pct is at index 0)
                episode_actions.append(td["action"].item())
                # Note: Current inflow/demand not available in observation
                # We'll use empty lists for now or get from environment if needed
            
            td = env.step(td)
            
            reward = td.get("next", {}).get("reward", td.get("reward", 0)).item()
            episode_reward += reward
            episode_rewards.append(reward)
            episode_length += 1
            
            # Track violations (simplified - you might need to check env internals)
            storage = td.get("next", {}).get("observation", td.get("observation"))[1].item()
            if storage > 0.9:
                violations["flood"] += 1
            elif storage < 0.1:
                violations["drought"] += 1
        
        # Final storage level
        if logger and episode_storages:
            final_obs = td.get("next", {}).get("observation", td.get("observation"))
            episode_storages.append(final_obs[0].item())  # volume_pct is at index 0
            
            # Log the episode
            # Note: We need to handle the fact that storage_levels has one extra element
            # Also, inflows and demands are not available in the current observation
            logger.log_episode(
                rewards=episode_rewards,
                storage_levels=episode_storages[:-1],  # Remove the final storage to match rewards length
                actions=episode_actions,
                inflows=[0.0] * len(episode_rewards),  # Placeholder - actual inflows not tracked
                demands=[0.0] * len(episode_rewards),  # Placeholder - demands not used in this env
                violations=violations
            )
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        violations_list.append(violations)
    
    return np.mean(rewards), np.std(rewards), np.mean(episode_lengths), violations_list


def save_checkpoint(q_net, target_net_params, optimizer, frame_count, checkpoint_dir):
    """Save a training checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{frame_count}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # target_net_params is a TensorDictParams object, convert to state dict
    target_state_dict = {k: v for k, v in target_net_params.items()} if hasattr(target_net_params, 'items') else target_net_params
    
    torch.save({
        'frame_count': frame_count,
        'q_net_state_dict': q_net.state_dict(),
        'target_net_params': target_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training loop."""
    args = parse_args()
    
    # Create environment
    print(f"Creating environment with {args.inflow_scenario} scenario...")
    env = create_env(args.inflow_scenario, args.device)
    
    # Create Q-network
    print("Creating Q-network...")
    q_net = create_qnet(
        state_dim=13,  # Observation dimension for reservoir env
        action_dim=11,  # 11 discrete actions (0-10% release)
        hidden_dim=256,
        device=args.device
    )
    
    # Create DQN training components
    print("Setting up DQN components...")
    components = create_dqn_training_components(
        qnet=q_net,
        action_spec=env.action_spec,
        gamma=args.gamma,
        eps_start=args.epsilon_start,
        eps_end=args.epsilon_end,
        eps_decay_steps=args.epsilon_frames,
        device=args.device
    )
    
    actor = components["actor"]
    loss_module = components["loss_module"]
    exploration_module = components["exploration_module"]
    target_updater = components["target_updater"]
    
    # Create exploration policy by wrapping actor with exploration module
    from tensordict.nn import TensorDictSequential
    exploration_policy = TensorDictSequential(actor, exploration_module)
    
    # Create optimizer
    optimizer = torch.optim.Adam(q_net.parameters(), lr=args.lr)
    
    # Create replay buffer
    print(f"Creating replay buffer with capacity {args.buffer_size}...")
    replay_buffer = create_replay_buffer(
        buffer_size=args.buffer_size,
        device=args.device
    )
    
    # Create logger
    print(f"Setting up TensorBoard logger at {args.log_dir}...")
    logger = ReservoirLogger(args.log_dir)
    
    # Log hyperparameters
    logger.log_hyperparameters(vars(args))
    
    # Create collector with exploration policy
    print("Setting up data collector...")
    collector = SyncDataCollector(
        env,
        exploration_policy,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        device=args.device,
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_storages = []
    episode_actions = []
    episode_inflows = []
    episode_demands = []
    current_episode_rewards = []
    current_violations = {"flood": 0, "drought": 0, "env_flow": 0}
    training_losses = []
    frame_count = 0
    episodes_completed = 0
    best_reward = float('-inf')
    
    print(f"\nStarting training for {args.total_frames} frames...")
    print("=" * 60)
    
    # Main training loop
    for i, batch in enumerate(collector):
        frame_count += args.frames_per_batch
        
        # Add batch to replay buffer
        replay_buffer.extend(batch)
        
        # Process batch data for episode tracking
        for t in range(batch.shape[0]):
            # Extract data from current timestep
            obs = batch["observation"][t]
            action = batch["action"][t].item()
            
            # Track episode data
            episode_storages.append(obs[0].item())  # Storage level (volume_pct is at index 0)
            episode_actions.append(action)
            # Note: Current inflow/demand not available in observation
            
            # Check for violations
            storage = obs[0].item()  # volume_pct is at index 0
            if storage > 0.9:
                current_violations["flood"] += 1
            elif storage < 0.1:
                current_violations["drought"] += 1
            
            if "next" in batch.keys():
                reward = batch["next"]["reward"][t].item()
                current_episode_rewards.append(reward)
                
                # Check if episode is done
                if batch["next"]["done"][t].item():
                    # Episode completed - log it
                    episode_storages.append(batch["next"]["observation"][t][0].item())  # Final storage (volume_pct at index 0)
                    
                    logger.log_episode(
                        rewards=current_episode_rewards,
                        storage_levels=episode_storages[:-1],  # Remove final storage to match rewards length
                        actions=episode_actions,
                        inflows=[0.0] * len(current_episode_rewards),  # Placeholder
                        demands=[0.0] * len(current_episode_rewards),  # Placeholder
                        violations=current_violations
                    )
                    
                    total_reward = sum(current_episode_rewards)
                    episode_rewards.append(total_reward)
                    episode_lengths.append(len(current_episode_rewards))
                    episodes_completed += 1
                    
                    # Track best reward
                    if total_reward > best_reward:
                        best_reward = total_reward
                        save_checkpoint(
                            q_net,
                            loss_module.target_value_network_params,
                            optimizer,
                            frame_count,
                            args.checkpoint_dir + "/best"
                        )
                    
                    # Reset episode tracking
                    episode_storages = []
                    episode_actions = []
                    current_episode_rewards = []
                    current_violations = {"flood": 0, "drought": 0, "env_flow": 0}
        
        # Train if buffer has enough samples
        if len(replay_buffer) >= args.min_replay_size:
            # Sample batch and train
            train_batch = replay_buffer.sample(args.batch_size)
            
            # Compute loss
            loss_td = loss_module(train_batch)
            loss = loss_td["loss"]
            
            # Get Q-values for logging
            with torch.no_grad():
                q_values = q_net(train_batch["observation"])
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target network
            target_updater.step()
            
            training_losses.append(loss.item())
            
            # Log training step
            logger.log_training_step(
                loss=loss.item(),
                epsilon=exploration_module.eps,
                learning_rate=args.lr,
                buffer_size=len(replay_buffer),
                q_values=q_values
            )
            logger.step()
        
        # Logging
        if frame_count % args.log_interval == 0:
            avg_loss = np.mean(training_losses[-10:]) if training_losses else 0
            current_epsilon = exploration_module.eps
            
            print(f"Frame {frame_count}/{args.total_frames} | "
                  f"Episodes: {episodes_completed} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {current_epsilon:.3f} | "
                  f"Buffer: {len(replay_buffer)}")
            
            if episode_rewards:
                recent_rewards = episode_rewards[-10:]
                print(f"  Recent episode rewards: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
        
        # Evaluation
        if frame_count % args.eval_interval == 0:
            print("\nRunning evaluation...")
            eval_mean, eval_std, eval_length, eval_violations = evaluate(
                actor,  # Use the base actor without exploration
                env,
                num_episodes=5,
                logger=logger
            )
            print(f"Evaluation: {eval_mean:.2f} ± {eval_std:.2f} (avg length: {eval_length:.1f})")
            
            # Log evaluation metrics
            logger.log_evaluation(
                eval_rewards=[eval_mean] * 5,  # Simplified - in practice track each episode
                eval_lengths=[eval_length] * 5,
                eval_violations=eval_violations
            )
            print()
        
        # Checkpointing
        if frame_count % args.checkpoint_interval == 0:
            save_checkpoint(
                q_net,
                loss_module.target_value_network_params,
                optimizer,
                frame_count,
                args.checkpoint_dir
            )
        
        # Update exploration
        exploration_module.step(frame_count)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training completed! Running final evaluation...")
    final_mean, final_std, final_length, final_violations = evaluate(
        actor,  # Use the base actor without exploration
        env,
        num_episodes=10,
        logger=logger
    )
    print(f"Final evaluation: {final_mean:.2f} ± {final_std:.2f} (avg length: {final_length:.1f})")
    
    # Update hyperparameter metrics
    logger.update_hyperparameter_metrics(final_mean)
    
    # Save final checkpoint
    save_checkpoint(
        q_net,
        loss_module.target_value_network_params,
        optimizer,
        frame_count,
        args.checkpoint_dir
    )
    
    # Print summary statistics
    if episode_rewards:
        print("\nTraining Summary:")
        print(f"  Total episodes: {episodes_completed}")
        print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Best episode: {max(episode_rewards):.2f}")
        print(f"  Worst episode: {min(episode_rewards):.2f}")
    
    # Close logger
    logger.close()
    print(f"\nTensorBoard logs saved to: {logger.log_dir}")
    print("Run 'tensorboard --logdir runs/' to view the results.")


if __name__ == "__main__":
    main()