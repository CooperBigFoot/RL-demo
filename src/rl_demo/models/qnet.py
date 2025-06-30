"""Q-network architecture for DQN agent."""

import torch
import torch.nn as nn
from torchrl.modules import MLP


def create_qnet(
    state_dim: int = 13,
    action_dim: int = 11,
    hidden_dim: int = 128,
    activation_class: type[nn.Module] = nn.ReLU,
    device: torch.device | str = "cpu",
) -> MLP:
    """Create Q-network for DQN agent.

    Creates a multi-layer perceptron (MLP) that maps states to Q-values
    for each possible action. The network uses two hidden layers with
    ReLU activations and outputs raw Q-values (no activation on output).

    Args:
        state_dim: Dimension of state space (default: 13 for reservoir state)
        action_dim: Number of discrete actions (default: 11 for 0-10% release)
        hidden_dim: Number of units in each hidden layer (default: 128)
        activation_class: Activation function class (default: nn.ReLU)
        device: Device to place the network on (default: "cpu")

    Returns:
        MLP: Q-network that maps states to Q-values

    Example:
        >>> qnet = create_qnet()
        >>> state = torch.randn(32, 13)  # batch of 32 states
        >>> q_values = qnet(state)  # shape: (32, 11)
    """
    qnet = MLP(
        in_features=state_dim,
        out_features=action_dim,
        num_cells=[hidden_dim, hidden_dim],
        activation_class=activation_class,
        activate_last_layer=False,  # No activation on Q-value outputs
        device=device,
    )

    # Apply Xavier initialization to improve training stability
    # TorchRL's MLP uses reasonable defaults, but we can be explicit
    for module in qnet.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    return qnet


class DQNValueNetwork(nn.Module):
    """Wrapper class for Q-network with additional functionality.

    This class provides a more feature-rich interface around the basic
    Q-network, including methods for computing Q-values, selecting actions,
    and other utilities needed for DQN training.
    """

    def __init__(
        self,
        state_dim: int = 13,
        action_dim: int = 11,
        hidden_dim: int = 128,
        device: torch.device | str = "cpu",
    ):
        """Initialize DQN value network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            hidden_dim: Number of units in hidden layers
            device: Device to place the network on
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device) if isinstance(device, str) else device

        # Create the Q-network
        self.qnet = create_qnet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given states.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        return self.qnet(state)

    def get_action_values(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        """Get Q-values for specific actions or all actions.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Optional action tensor of shape (batch_size,) or (batch_size, 1)
                   If None, returns Q-values for all actions

        Returns:
            If action is provided: Q-values for specific actions, shape (batch_size, 1)
            If action is None: Q-values for all actions, shape (batch_size, action_dim)
        """
        q_values = self.forward(state)

        if action is None:
            return q_values

        # Gather Q-values for specific actions
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        return q_values.gather(1, action.long())

    def to(self, device: torch.device | str) -> "DQNValueNetwork":
        """Move network to specified device."""
        super().to(device)
        self.device = torch.device(device) if isinstance(device, str) else device
        return self

