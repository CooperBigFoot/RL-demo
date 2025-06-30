"""Tests for DQN training components."""

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import OneHot, TensorDictReplayBuffer
from torchrl.modules import QValueActor
from torchrl.objectives import DQNLoss, SoftUpdate

from rl_demo.models.qnet import create_qnet
from rl_demo.trainers.dqn_components import (
    create_dqn_loss,
    create_dqn_training_components,
    create_exploration_module,
    create_qvalue_actor,
    create_replay_buffer,
    create_target_updater,
)


class TestDQNLoss:
    """Test the DQN loss module creation."""

    def test_create_dqn_loss_default(self):
        """Test DQN loss with default parameters."""
        # Create a simple Q-network and actor
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(qnet, action_spec)

        # Create loss module
        loss_module = create_dqn_loss(actor)

        assert isinstance(loss_module, DQNLoss)
        assert loss_module.delay_value is True

    def test_create_dqn_loss_custom_params(self):
        """Test DQN loss with custom parameters."""
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(qnet, action_spec)

        # Create loss with custom parameters
        loss_module = create_dqn_loss(
            value_network=actor,
            gamma=0.95,
            loss_function="l2",
            delay_value=False,
        )

        assert loss_module.delay_value is False

    def test_loss_computation(self):
        """Test that loss can be computed on batch data."""
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(qnet, action_spec)
        loss_module = create_dqn_loss(actor)

        # Create dummy batch
        batch = TensorDict(
            {
                "observation": torch.randn(32, 13),
                "action": action_spec.rand((32,)),
                ("next", "observation"): torch.randn(32, 13),
                ("next", "reward"): torch.randn(32, 1),
                ("next", "done"): torch.zeros(32, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(32, 1, dtype=torch.bool),
            },
            batch_size=[32],
        )

        # Compute loss
        loss_vals = loss_module(batch)
        assert "loss" in loss_vals.keys()
        assert loss_vals["loss"].shape == ()  # Scalar loss


class TestExplorationModule:
    """Test epsilon-greedy exploration module."""

    def test_create_exploration_default(self):
        """Test exploration module with default parameters."""
        action_spec = OneHot(11)
        exploration = create_exploration_module(action_spec)

        # Check initial epsilon
        assert exploration.eps_init == 1.0
        assert exploration.eps_end == 0.01
        assert exploration.annealing_num_steps == 10000

    def test_create_exploration_custom(self):
        """Test exploration module with custom parameters."""
        action_spec = OneHot(5)
        exploration = create_exploration_module(
            action_spec=action_spec,
            eps_start=0.5,
            eps_end=0.05,
            annealing_num_steps=5000,
        )

        assert exploration.eps_init == 0.5
        assert exploration.eps_end == 0.05
        assert exploration.annealing_num_steps == 5000

    def test_exploration_properties(self):
        """Test exploration module properties."""
        action_spec = OneHot(11)
        exploration = create_exploration_module(
            action_spec=action_spec,
            eps_start=0.5,
            eps_end=0.05,
            annealing_num_steps=5000,
        )

        # Test that epsilon is set correctly
        assert exploration.eps == 0.5


class TestQValueActor:
    """Test Q-value actor creation."""

    def test_create_qvalue_actor(self):
        """Test creating Q-value actor from Q-network."""
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(qnet, action_spec)

        # Check structure
        assert isinstance(actor, QValueActor)

        # Test forward pass
        td = TensorDict({"observation": torch.randn(32, 13)}, batch_size=[32])
        output = actor(td)

        assert "action" in output.keys()
        assert output["action"].shape == torch.Size([32, 11]) or output["action"].shape == torch.Size([32])

    def test_qvalue_actor_custom_keys(self):
        """Test Q-value actor with custom keys."""
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(
            qnet,
            action_spec,
            in_keys=["state"],
            )

        # Test with custom keys
        td = TensorDict({"state": torch.randn(16, 13)}, batch_size=[16])
        output = actor(td)

        assert "action" in output.keys()
        assert "action_value" in output.keys()


class TestTargetUpdater:
    """Test target network updater."""

    def test_soft_update(self):
        """Test soft target network updates."""
        # Create actor and loss module
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(qnet, action_spec)
        loss_module = create_dqn_loss(actor)

        # Create updater
        updater = create_target_updater(
            loss_module=loss_module,
            update_type="soft",
            tau=0.99,  # Slow updates
        )

        assert isinstance(updater, SoftUpdate)

        # Perform update
        updater.step()

    def test_hard_update(self):
        """Test hard target network updates."""
        qnet = create_qnet()
        action_spec = OneHot(11)
        actor = create_qvalue_actor(qnet, action_spec)
        loss_module = create_dqn_loss(actor)

        updater = create_target_updater(
            loss_module=loss_module,
            update_type="hard",
        )

        # Perform update
        updater.step()


class TestReplayBuffer:
    """Test replay buffer creation and functionality."""

    def test_create_replay_buffer_default(self):
        """Test replay buffer with default parameters."""
        buffer = create_replay_buffer()

        assert isinstance(buffer, TensorDictReplayBuffer)
        assert buffer._batch_size == 32
        assert len(buffer) == 0  # Empty initially

    def test_create_replay_buffer_custom(self):
        """Test replay buffer with custom parameters."""
        buffer = create_replay_buffer(
            buffer_size=5000,
            batch_size=64,
        )

        assert buffer._batch_size == 64

    def test_replay_buffer_operations(self):
        """Test adding and sampling from replay buffer."""
        buffer = create_replay_buffer(buffer_size=100, batch_size=4)

        # Add some transitions
        for i in range(10):
            transition = TensorDict(
                {
                    "observation": torch.randn(13),
                    "action": torch.tensor(i % 11),
                    "reward": torch.tensor([float(i)]),
                    ("next", "observation"): torch.randn(13),
                    ("next", "done"): torch.tensor([False]),
                },
                batch_size=[],
            )
            buffer.add(transition)

        assert len(buffer) == 10

        # Sample a batch
        batch = buffer.sample()
        assert batch.batch_size[0] == 4
        assert "observation" in batch.keys()
        assert "action" in batch.keys()


class TestIntegratedComponents:
    """Test the integrated component creation function."""

    def test_create_all_components(self):
        """Test creating all DQN components at once."""
        # Create Q-network
        qnet = create_qnet()

        # Create action spec
        action_spec = OneHot(11)

        # Create all components
        components = create_dqn_training_components(
            qnet=qnet,
            action_spec=action_spec,
        )

        # Check all components exist
        assert "actor" in components
        assert "loss_module" in components
        assert "exploration_module" in components
        assert "replay_buffer" in components
        assert "target_updater" in components

        # Check types
        assert isinstance(components["actor"], QValueActor)
        assert isinstance(components["loss_module"], DQNLoss)
        assert hasattr(components["exploration_module"], "eps")
        assert hasattr(components["replay_buffer"], "sample")
        assert hasattr(components["target_updater"], "step")

    def test_components_integration(self):
        """Test that components work together."""
        qnet = create_qnet()
        action_spec = OneHot(11)
        components = create_dqn_training_components(
            qnet=qnet,
            action_spec=action_spec,
            buffer_size=100,
            batch_size=4,
        )

        # Generate some experience
        td = TensorDict({"observation": torch.randn(13)}, batch_size=[])
        
        # Get action without exploration first
        td_actor = components["actor"](td.clone())
        assert "action" in td_actor.keys()
        
        # Apply exploration
        td_explore = components["exploration_module"](td_actor)

        # Add experience to buffer
        td_explore["reward"] = torch.tensor([1.0])
        td_explore[("next", "observation")] = torch.randn(13)
        td_explore[("next", "done")] = torch.tensor([False])
        td_explore[("next", "terminated")] = torch.tensor([False])
        td_explore[("next", "reward")] = torch.tensor([0.0])
        
        components["replay_buffer"].add(td_explore)

        # After adding enough samples, we can train
        for i in range(10):
            td_new = TensorDict({
                "observation": torch.randn(13),
                "action": torch.tensor(i % 11),
                "reward": torch.tensor([1.0]),
                ("next", "observation"): torch.randn(13),
                ("next", "done"): torch.tensor([False]),
                ("next", "terminated"): torch.tensor([False]),
                ("next", "reward"): torch.tensor([0.0]),
            }, batch_size=[])
            components["replay_buffer"].add(td_new)

        # Sample and compute loss
        batch = components["replay_buffer"].sample()
        loss_vals = components["loss_module"](batch)
        assert "loss" in loss_vals.keys()

    @pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    def test_device_placement(self, device):
        """Test that components are placed on correct device."""
        qnet = create_qnet()
        action_spec = OneHot(11)

        components = create_dqn_training_components(
            qnet=qnet,
            action_spec=action_spec,
            device=device,
        )

        # Check loss module device
        for param in components["loss_module"].parameters():
            assert str(param.device).startswith(device)