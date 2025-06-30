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
import yaml
import subprocess
import uuid
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
from rl_demo.configs.default_config import DQNConfig, get_default_config


def get_git_hash():
    """Get the current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()[:8]
    except:
        return "unknown"


def parse_args():
    """Parse command line arguments with config file support."""
    parser = argparse.ArgumentParser(
        description="Train DQN on Discrete Reservoir Environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (highest priority)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    
    # Experiment tracking
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment name (overrides config)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    # Environment overrides
    parser.add_argument("--inflow-scenario", type=str, default=None,
                        choices=["stable", "seasonal", "extreme", "mixed"],
                        help="Inflow scenario for the reservoir")
    
    # Training overrides
    parser.add_argument("--total-frames", type=int, default=None,
                        help="Total number of frames to train for")
    parser.add_argument("--frames-per-batch", type=int, default=None,
                        help="Number of frames to collect per batch")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    
    # DQN specific overrides
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=None,
                        help="Starting epsilon for exploration")
    parser.add_argument("--epsilon-end", type=float, default=None,
                        help="Final epsilon for exploration")
    parser.add_argument("--epsilon-frames", type=int, default=None,
                        help="Number of frames for epsilon decay")
    parser.add_argument("--target-update-freq", type=int, default=None,
                        help="Frequency of target network updates")
    
    # Device override
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (cpu/cuda)")
    
    # Additional options
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more frequent logging")
    
    return parser.parse_args()


def create_env(config: DQNConfig):
    """Create the discrete reservoir environment from config."""
    # Note: DiscreteReservoirEnv uses default reservoir parameters
    # and generates scenarios internally
    env = DiscreteReservoirEnv(device=config.environment.device)
    # The inflow_scenario is handled by the ReservoirSimulator internally
    return env


def load_config(args) -> DQNConfig:
    """Load configuration from file and command line arguments."""
    # Start with default config
    if args.config:
        print(f"Loading config from {args.config}")
        config = DQNConfig.from_yaml(args.config)
    else:
        print("Using default configuration")
        config = get_default_config()
    
    # Build override dict from command line arguments
    overrides = {}
    
    # Experiment overrides
    if args.exp_name is not None:
        overrides["experiment.exp_name"] = args.exp_name
    if args.seed is not None:
        overrides["experiment.seed"] = args.seed
    if args.resume is not None:
        overrides["experiment.resume"] = args.resume
    
    # Environment overrides
    if args.inflow_scenario is not None:
        overrides["environment.inflow_scenario"] = args.inflow_scenario
    if args.device is not None:
        overrides["environment.device"] = args.device
    elif args.no_cuda:
        overrides["environment.device"] = "cpu"
    
    # Training overrides
    if args.total_frames is not None:
        overrides["training.total_frames"] = args.total_frames
    if args.frames_per_batch is not None:
        overrides["training.frames_per_batch"] = args.frames_per_batch
    if args.batch_size is not None:
        overrides["training.batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["training.learning_rate"] = args.lr
    if args.gamma is not None:
        overrides["training.gamma"] = args.gamma
    if args.epsilon_start is not None:
        overrides["training.epsilon_start"] = args.epsilon_start
    if args.epsilon_end is not None:
        overrides["training.epsilon_end"] = args.epsilon_end
    if args.epsilon_frames is not None:
        overrides["training.epsilon_frames"] = args.epsilon_frames
    if args.target_update_freq is not None:
        overrides["training.target_update_freq"] = args.target_update_freq
    
    # Debug mode adjustments
    if args.debug:
        overrides["logging.log_interval"] = 10
        overrides["logging.eval_interval"] = 100
        overrides["logging.checkpoint_interval"] = 1000
    
    # Apply overrides
    config.update_from_args(overrides)
    
    # Auto-generate experiment ID and get git hash
    config.experiment.exp_id = f"{config.experiment.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    config.experiment.git_hash = get_git_hash()
    
    # Update log directory with experiment ID
    config.logging.log_dir = f"{config.logging.log_dir}/{config.experiment.exp_id}"
    config.logging.checkpoint_dir = f"{config.logging.checkpoint_dir}/{config.experiment.exp_id}"
    
    # Validate configuration
    config.validate()
    
    # Save the final config
    config_save_path = Path(config.logging.checkpoint_dir) / "config.yaml"
    config.to_yaml(config_save_path)
    print(f"Saved configuration to {config_save_path}")
    
    return config


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


def save_checkpoint(q_net, target_net_params, optimizer, frame_count, checkpoint_dir, config=None):
    """Save a training checkpoint."""
    checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{frame_count}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # target_net_params is a TensorDictParams object, convert to state dict
    target_state_dict = {k: v for k, v in target_net_params.items()} if hasattr(target_net_params, 'items') else target_net_params
    
    checkpoint_data = {
        'frame_count': frame_count,
        'q_net_state_dict': q_net.state_dict(),
        'target_net_params': target_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save config if provided
    if config is not None:
        checkpoint_data['config'] = config.to_dict()
        checkpoint_data['experiment_id'] = config.experiment.exp_id
    
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path, q_net, optimizer, loss_module):
    """Load a training checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model states
    q_net.load_state_dict(checkpoint['q_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load target network parameters
    if 'target_net_params' in checkpoint:
        target_params = checkpoint['target_net_params']
        # Update target network parameters
        for key, value in target_params.items():
            if hasattr(loss_module.target_value_network_params, key):
                getattr(loss_module.target_value_network_params, key).data = value
    
    frame_count = checkpoint.get('frame_count', 0)
    
    print(f"Resumed from checkpoint at frame {frame_count}")
    if 'experiment_id' in checkpoint:
        print(f"Original experiment ID: {checkpoint['experiment_id']}")
    
    return frame_count


def main():
    """Main training loop."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args)
    
    # Set random seed for reproducibility
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    
    # Print experiment info
    print("\n" + "=" * 60)
    print(f"Experiment: {config.experiment.exp_id}")
    print(f"Git hash: {config.experiment.git_hash}")
    print(f"Config: {args.config if args.config else 'default'}")
    print("=" * 60 + "\n")
    
    # Create environment
    print(f"Creating environment with {config.environment.inflow_scenario} scenario...")
    env = create_env(config)
    
    # Create Q-network
    print("Creating Q-network...")
    q_net = create_qnet(
        state_dim=config.network.state_dim,
        action_dim=config.network.action_dim,
        hidden_dim=config.network.hidden_dim,
        device=config.environment.device
    )
    
    # Create DQN training components
    print("Setting up DQN components...")
    components = create_dqn_training_components(
        qnet=q_net,
        action_spec=env.action_spec,
        gamma=config.training.gamma,
        eps_start=config.training.epsilon_start,
        eps_end=config.training.epsilon_end,
        eps_decay_steps=config.training.epsilon_frames,
        device=config.environment.device
    )
    
    actor = components["actor"]
    loss_module = components["loss_module"]
    exploration_module = components["exploration_module"]
    target_updater = components["target_updater"]
    
    # Create exploration policy by wrapping actor with exploration module
    from tensordict.nn import TensorDictSequential
    exploration_policy = TensorDictSequential(actor, exploration_module)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        q_net.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_frame = 0
    if config.experiment.resume:
        start_frame = load_checkpoint(
            config.experiment.resume,
            q_net,
            optimizer,
            loss_module
        )
        # Update exploration module to correct epsilon value
        exploration_module.step(start_frame)
    
    # Create replay buffer
    print(f"Creating replay buffer with capacity {config.training.buffer_size}...")
    replay_buffer = create_replay_buffer(
        buffer_size=config.training.buffer_size,
        device=config.environment.device
    )
    
    # Create logger
    print(f"Setting up TensorBoard logger at {config.logging.log_dir}...")
    logger = ReservoirLogger(config.logging.log_dir)
    
    # Log hyperparameters (convert config to flat dict)
    logger.log_hyperparameters(config.to_dict())
    
    # Create collector with exploration policy
    print("Setting up data collector...")
    collector = SyncDataCollector(
        env,
        exploration_policy,
        frames_per_batch=config.training.frames_per_batch,
        total_frames=config.training.total_frames - start_frame,  # Adjust for resumed training
        device=config.environment.device,
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
    frame_count = start_frame
    episodes_completed = 0
    best_reward = float('-inf')
    
    print(f"\nStarting training for {config.training.total_frames} frames...")
    print("=" * 60)
    
    # Main training loop
    for i, batch in enumerate(collector):
        frame_count += config.training.frames_per_batch
        
        # Update exploration based on total frames (including resumed)
        exploration_module.step(frame_count)
        
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
                    if total_reward > best_reward and config.logging.save_best:
                        best_reward = total_reward
                        save_checkpoint(
                            q_net,
                            loss_module.target_value_network_params,
                            optimizer,
                            frame_count,
                            config.logging.checkpoint_dir + "/best",
                            config
                        )
                    
                    # Reset episode tracking
                    episode_storages = []
                    episode_actions = []
                    current_episode_rewards = []
                    current_violations = {"flood": 0, "drought": 0, "env_flow": 0}
        
        # Train if buffer has enough samples
        if len(replay_buffer) >= config.training.min_replay_size:
            # Sample batch and train
            train_batch = replay_buffer.sample(config.training.batch_size)
            
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
                learning_rate=config.training.learning_rate,
                buffer_size=len(replay_buffer),
                q_values=q_values
            )
            logger.step()
            
            # Apply gradient clipping if configured
            if config.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), config.training.grad_clip)
        
        # Logging
        if frame_count % config.logging.log_interval == 0:
            avg_loss = np.mean(training_losses[-10:]) if training_losses else 0
            current_epsilon = exploration_module.eps
            
            print(f"Frame {frame_count}/{config.training.total_frames} | "
                  f"Episodes: {episodes_completed} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {current_epsilon:.3f} | "
                  f"Buffer: {len(replay_buffer)}")
            
            if episode_rewards:
                recent_rewards = episode_rewards[-10:]
                print(f"  Recent episode rewards: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}")
        
        # Evaluation
        if frame_count % config.logging.eval_interval == 0:
            print("\nRunning evaluation...")
            eval_mean, eval_std, eval_length, eval_violations = evaluate(
                actor,  # Use the base actor without exploration
                env,
                num_episodes=config.logging.eval_episodes,
                logger=logger
            )
            print(f"Evaluation: {eval_mean:.2f} ± {eval_std:.2f} (avg length: {eval_length:.1f})")
            
            # Log evaluation metrics
            logger.log_evaluation(
                eval_rewards=[eval_mean] * config.logging.eval_episodes,  # Simplified - in practice track each episode
                eval_lengths=[eval_length] * config.logging.eval_episodes,
                eval_violations=eval_violations
            )
            print()
        
        # Checkpointing
        if frame_count % config.logging.checkpoint_interval == 0:
            save_checkpoint(
                q_net,
                loss_module.target_value_network_params,
                optimizer,
                frame_count,
                config.logging.checkpoint_dir,
                config
            )
        
        # Note: exploration update moved earlier in the loop
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Training completed! Running final evaluation...")
    final_mean, final_std, final_length, final_violations = evaluate(
        actor,  # Use the base actor without exploration
        env,
        num_episodes=config.logging.eval_episodes * 2,  # Double episodes for final eval
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
        config.logging.checkpoint_dir,
        config
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
    print(f"\nExperiment ID: {config.experiment.exp_id}")
    print(f"Config saved to: {config.logging.checkpoint_dir}/config.yaml")


if __name__ == "__main__":
    main()