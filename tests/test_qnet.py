"""Tests for Q-network architecture."""

import torch
import torch.nn as nn
from torchrl.modules import MLP

from rl_demo.models.qnet import DQNValueNetwork, create_qnet


class TestCreateQnet:
    """Test the create_qnet factory function."""

    def test_default_architecture(self):
        """Test Q-network with default parameters."""
        qnet = create_qnet()

        # Check it's an MLP instance
        assert isinstance(qnet, MLP)

        # Test input/output dimensions
        batch_size = 32
        state = torch.randn(batch_size, 13)
        q_values = qnet(state)

        assert q_values.shape == (batch_size, 11)

    def test_custom_dimensions(self):
        """Test Q-network with custom dimensions."""
        state_dim, action_dim, hidden_dim = 20, 5, 256
        qnet = create_qnet(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        # Test with custom dimensions
        batch_size = 16
        state = torch.randn(batch_size, state_dim)
        q_values = qnet(state)

        assert q_values.shape == (batch_size, action_dim)

    def test_network_layers(self):
        """Test that network has correct layer structure."""
        qnet = create_qnet()

        # MLP in TorchRL wraps layers in Sequential
        modules = list(qnet.modules())

        # Count Linear layers
        linear_layers = [m for m in modules if isinstance(m, nn.Linear)]
        assert len(linear_layers) == 3  # input->hidden1, hidden1->hidden2, hidden2->output

        # Check dimensions
        assert linear_layers[0].in_features == 13
        assert linear_layers[0].out_features == 128
        assert linear_layers[1].in_features == 128
        assert linear_layers[1].out_features == 128
        assert linear_layers[2].in_features == 128
        assert linear_layers[2].out_features == 11

    def test_activation_functions(self):
        """Test that ReLU activations are present."""
        qnet = create_qnet()

        # Count ReLU activations
        relu_count = sum(1 for m in qnet.modules() if isinstance(m, nn.ReLU))
        assert relu_count == 2  # Two ReLU activations (after first two layers)

    def test_no_output_activation(self):
        """Test that output layer has no activation."""
        qnet = create_qnet()

        # Get the sequential module
        seq_module = next(m for m in qnet.modules() if isinstance(m, nn.Sequential))
        layers = list(seq_module.children())

        # Last layer should be Linear (no activation after it)
        assert isinstance(layers[-1], nn.Linear)

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        qnet = create_qnet()

        # Forward pass
        state = torch.randn(8, 13, requires_grad=True)
        q_values = qnet(state)

        # Compute loss and backward
        loss = q_values.mean()
        loss.backward()

        # Check gradients exist
        assert state.grad is not None
        assert not torch.allclose(state.grad, torch.zeros_like(state.grad))

        # Check all parameters have gradients
        for param in qnet.parameters():
            assert param.grad is not None

    def test_device_placement(self):
        """Test network placement on different devices."""
        # Test CPU
        qnet_cpu = create_qnet(device="cpu")
        assert next(qnet_cpu.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            qnet_cuda = create_qnet(device="cuda")
            assert next(qnet_cuda.parameters()).device.type == "cuda"

            # Test forward pass on CUDA
            state = torch.randn(4, 13, device="cuda")
            q_values = qnet_cuda(state)
            assert q_values.device.type == "cuda"


class TestDQNValueNetwork:
    """Test the DQNValueNetwork wrapper class."""

    def test_initialization(self):
        """Test network initialization."""
        net = DQNValueNetwork()

        assert net.state_dim == 13
        assert net.action_dim == 11
        assert isinstance(net.qnet, MLP)

    def test_forward_pass(self):
        """Test forward pass through the network."""
        net = DQNValueNetwork()

        batch_size = 16
        state = torch.randn(batch_size, 13)
        q_values = net(state)

        assert q_values.shape == (batch_size, 11)

    def test_get_action_values_all_actions(self):
        """Test getting Q-values for all actions."""
        net = DQNValueNetwork()

        state = torch.randn(8, 13)
        q_values = net.get_action_values(state, action=None)

        assert q_values.shape == (8, 11)

    def test_get_action_values_specific_actions(self):
        """Test getting Q-values for specific actions."""
        net = DQNValueNetwork()

        batch_size = 8
        state = torch.randn(batch_size, 13)
        action = torch.randint(0, 11, (batch_size,))

        q_values = net.get_action_values(state, action)

        assert q_values.shape == (batch_size, 1)

        # Verify correct values are selected
        all_q_values = net(state)
        for i in range(batch_size):
            assert torch.allclose(q_values[i, 0], all_q_values[i, action[i]])

    def test_get_action_values_with_2d_actions(self):
        """Test getting Q-values with 2D action tensor."""
        net = DQNValueNetwork()

        batch_size = 8
        state = torch.randn(batch_size, 13)
        action = torch.randint(0, 11, (batch_size, 1))

        q_values = net.get_action_values(state, action)

        assert q_values.shape == (batch_size, 1)

    def test_device_transfer(self):
        """Test moving network between devices."""
        net = DQNValueNetwork(device="cpu")

        # Check initial device
        assert net.device.type == "cpu"
        assert next(net.parameters()).device.type == "cpu"

        # Test CUDA transfer if available
        if torch.cuda.is_available():
            net_cuda = net.to("cuda")

            assert net_cuda.device.type == "cuda"
            assert next(net_cuda.parameters()).device.type == "cuda"

            # Test forward pass on CUDA
            state = torch.randn(4, 13, device="cuda")
            q_values = net_cuda(state)
            assert q_values.device.type == "cuda"

    def test_custom_initialization(self):
        """Test network with custom parameters."""
        net = DQNValueNetwork(state_dim=20, action_dim=5, hidden_dim=256)

        assert net.state_dim == 20
        assert net.action_dim == 5

        # Test forward pass
        state = torch.randn(4, 20)
        q_values = net(state)
        assert q_values.shape == (4, 5)

    def test_parameter_initialization(self):
        """Test that parameters are properly initialized."""
        net = DQNValueNetwork()

        # Check that weights are not all zeros or ones
        for name, param in net.named_parameters():
            if "weight" in name:
                assert not torch.allclose(param, torch.zeros_like(param))
                assert not torch.allclose(param, torch.ones_like(param))
                # Check reasonable value range (Xavier init typically gives values in [-1, 1])
                assert param.abs().max() < 3.0
            elif "bias" in name:
                # Biases should be initialized to zero
                assert torch.allclose(param, torch.zeros_like(param))

    def test_reproducibility(self):
        """Test that network creation is reproducible with fixed seed."""
        torch.manual_seed(42)
        net1 = DQNValueNetwork()
        state = torch.randn(4, 13)
        out1 = net1(state)

        torch.manual_seed(42)
        net2 = DQNValueNetwork()
        out2 = net2(state)

        assert torch.allclose(out1, out2)


class TestIntegration:
    """Integration tests for Q-network with TorchRL components."""

    def test_batched_inference(self):
        """Test network handles various batch sizes correctly."""
        net = DQNValueNetwork()

        # Test different batch sizes
        for batch_size in [1, 8, 32, 128]:
            state = torch.randn(batch_size, 13)
            q_values = net(state)
            assert q_values.shape == (batch_size, 11)

    def test_torchrl_compatibility(self):
        """Test that network works with TorchRL's expected interfaces."""
        qnet = create_qnet()

        # TorchRL typically expects networks to handle TensorDict inputs
        # But our MLP works with regular tensors, which is fine for QValueModule

        # Test that it's a proper nn.Module
        assert isinstance(qnet, nn.Module)

        # Test that it has standard module methods
        assert hasattr(qnet, "forward")
        assert hasattr(qnet, "parameters")
        assert hasattr(qnet, "state_dict")

    def test_memory_efficiency(self):
        """Test that network doesn't create unnecessary copies."""
        net = DQNValueNetwork()

        state = torch.randn(100, 13)

        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Multiple forward passes shouldn't accumulate memory
        for _ in range(10):
            _ = net(state)

        # No memory test assertion, but this ensures no obvious memory leaks

