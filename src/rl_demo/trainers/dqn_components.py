"""DQN training components using TorchRL."""

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.data import (
    LazyMemmapStorage,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate


def create_dqn_loss(
    value_network: nn.Module | QValueActor,
    gamma: float = 0.99,
    loss_function: str = "smooth_l1",
    delay_value: bool = True,
    device: torch.device | str = "cpu",
) -> DQNLoss:
    """Create DQN loss module with specified configuration.

    Args:
        value_network: Q-network module or QValueActor that outputs action values
        gamma: Discount factor for future rewards (default: 0.99)
        loss_function: Loss function to use, either "smooth_l1" or "l2" (default: "smooth_l1")
        delay_value: Whether to use a target network (default: True)
        device: Device to place the loss module on (default: "cpu")

    Returns:
        DQNLoss: Configured DQN loss module ready for training

    Example:
        >>> qnet = create_qnet()
        >>> actor = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)
        >>> loss_module = create_dqn_loss(actor)
    """
    loss_module = DQNLoss(
        value_network=value_network,
        loss_function=loss_function,
        delay_value=delay_value,
    )

    # Set gamma using make_value_estimator
    loss_module.make_value_estimator(gamma=gamma)

    # Move to specified device
    loss_module = loss_module.to(device)

    return loss_module


def create_exploration_module(
    action_spec,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    annealing_num_steps: int = 10000,
) -> EGreedyModule:
    """Create epsilon-greedy exploration module with linear decay.

    Args:
        action_spec: Action specification from the environment
        eps_start: Initial epsilon value for exploration (default: 1.0)
        eps_end: Final epsilon value after decay (default: 0.01)
        annealing_num_steps: Number of steps to decay epsilon over (default: 10000)

    Returns:
        EGreedyModule: Configured epsilon-greedy exploration module

    Example:
        >>> from torchrl.data import OneHot
        >>> action_spec = OneHot(11)  # 11 discrete actions
        >>> exploration = create_exploration_module(action_spec)
    """
    exploration_module = EGreedyModule(
        spec=action_spec,
        eps_init=eps_start,
        eps_end=eps_end,
        annealing_num_steps=annealing_num_steps,
    )

    return exploration_module


def create_target_updater(
    loss_module: DQNLoss,
    update_type: str = "soft",
    tau: float = 0.995,
) -> SoftUpdate | HardUpdate:
    """Create target network updater for DQN.

    Args:
        loss_module: DQN loss module containing the value network
        update_type: Type of update - "soft" or "hard" (default: "soft")
        tau: Soft update interpolation parameter (default: 0.995)
            For soft updates: target = tau * target + (1 - tau) * main
            Note: TorchRL uses tau differently - closer to 1 means slower updates

    Returns:
        Target network updater module (SoftUpdate or HardUpdate)

    Example:
        >>> loss_module = create_dqn_loss(actor)
        >>> updater = create_target_updater(loss_module)
        >>> # In training loop:
        >>> if step % 100 == 0:
        ...     updater.step()
    """
    if update_type == "soft":
        updater = SoftUpdate(
            loss_module,
            eps=tau,  # In TorchRL, eps is used instead of tau
        )
    elif update_type == "hard":
        updater = HardUpdate(
            loss_module,
        )
    else:
        raise ValueError(f"Unknown update type: {update_type}. Choose 'soft' or 'hard'.")

    return updater


def create_replay_buffer(
    buffer_size: int = 10000,
    batch_size: int = 32,
    device: torch.device | str = "cpu",
    prefetch: int | None = None,
) -> TensorDictReplayBuffer:
    """Create replay buffer for experience storage and sampling.

    Args:
        buffer_size: Maximum number of transitions to store (default: 10000)
        batch_size: Number of transitions to sample per batch (default: 32)
        device: Device to store replay buffer on (default: "cpu")
        prefetch: Number of batches to prefetch for efficiency (default: None)

    Returns:
        TensorDictReplayBuffer: Configured replay buffer ready for storing transitions

    Example:
        >>> buffer = create_replay_buffer(buffer_size=10000, batch_size=32)
        >>> # Store transition
        >>> buffer.add(transition_tensordict)
        >>> # Sample batch
        >>> batch = buffer.sample()
    """
    device = torch.device(device) if isinstance(device, str) else device

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(
            max_size=buffer_size,
            device=device,
        ),
        sampler=SamplerWithoutReplacement(),
        batch_size=batch_size,
        prefetch=prefetch,
    )

    return replay_buffer


def create_qvalue_actor(
    qnet: nn.Module,
    action_spec,
    in_keys: list[str] | None = None,
) -> QValueActor:
    """Create Q-value actor from Q-network.

    Args:
        qnet: Q-network module that outputs Q-values
        action_spec: Action specification from environment
        in_keys: Input keys for the network (default: ["observation"])

    Returns:
        QValueActor: Actor that selects actions based on Q-values

    Example:
        >>> qnet = create_qnet()
        >>> actor = create_qvalue_actor(qnet, env.action_spec)
    """
    if in_keys is None:
        in_keys = ["observation"]

    # Create Q-value actor directly
    actor = QValueActor(
        module=qnet,
        in_keys=in_keys,
        spec=action_spec,
    )

    return actor


def create_dqn_training_components(
    qnet: nn.Module,
    action_spec,
    gamma: float = 0.99,
    eps_start: float = 1.0,
    eps_end: float = 0.01,
    eps_decay_steps: int = 10000,
    buffer_size: int = 10000,
    batch_size: int = 32,
    tau: float = 0.995,
    device: torch.device | str = "cpu",
) -> dict:
    """Create all DQN training components in one call.

    This is a convenience function that creates all the necessary components
    for DQN training with consistent configuration.

    Args:
        qnet: Q-network module (raw neural network)
        action_spec: Action specification from the environment
        gamma: Discount factor (default: 0.99)
        eps_start: Initial exploration epsilon (default: 1.0)
        eps_end: Final exploration epsilon (default: 0.01)
        eps_decay_steps: Steps to decay epsilon over (default: 10000)
        buffer_size: Replay buffer capacity (default: 10000)
        batch_size: Batch size for training (default: 32)
        tau: Soft update parameter for target network (default: 0.995)
        device: Device to place components on (default: "cpu")

    Returns:
        Dictionary containing:
            - actor: QValueActor for action selection
            - loss_module: DQN loss module
            - exploration_module: Epsilon-greedy exploration
            - replay_buffer: Experience replay buffer
            - target_updater: Target network updater

    Example:
        >>> qnet = create_qnet()
        >>> components = create_dqn_training_components(
        ...     qnet=qnet,
        ...     action_spec=env.action_spec,
        ... )
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Create Q-value actor
    actor = create_qvalue_actor(
        qnet=qnet.to(device),
        action_spec=action_spec,
        in_keys=["observation"],
    )

    # Create exploration policy
    exploration_module = create_exploration_module(
        action_spec=action_spec,
        eps_start=eps_start,
        eps_end=eps_end,
        annealing_num_steps=eps_decay_steps,
    )

    components = {
        "actor": actor,
        "loss_module": create_dqn_loss(
            value_network=actor,
            gamma=gamma,
            device=device,
        ),
        "exploration_module": exploration_module,
        "replay_buffer": create_replay_buffer(
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
        ),
    }

    # Create target updater after loss module is created
    components["target_updater"] = create_target_updater(
        loss_module=components["loss_module"],
        update_type="soft",
        tau=tau,
    )

    return components
