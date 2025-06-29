"""Tests for DiscreteReservoirEnv TorchRL wrapper."""

import torch
import pytest
from tensordict import TensorDict

from rl_demo.envs.discrete_reservoir_env import DiscreteReservoirEnv


class TestDiscreteReservoirEnv:
    """Test suite for DiscreteReservoirEnv."""
    
    @pytest.fixture
    def env(self):
        """Create a test environment instance."""
        return DiscreteReservoirEnv(
            v_max=1000.0,
            v_min=100.0,
            v_dead=50.0,
            initial_volume=500.0,
        )
    
    def test_basic_functionality(self, env):
        """Test reset/step cycle works correctly."""
        # Test reset
        td = env.reset()
        assert td["observation"].shape == (13,)
        assert td["observation"].dtype == torch.float32
        assert not td["done"].item()
        
        # Test step with each action
        for action in range(11):
            td["action"] = torch.tensor(action, dtype=torch.int64)
            td = env.step(td)
            assert "next" in td
            assert "reward" in td["next"]
            assert td["next"]["observation"].shape == (13,)
            assert td["next"]["reward"].shape == (1,)
            assert td["next"]["done"].shape == (1,)
    
    def test_action_constraints(self, env):
        """Verify all discrete actions map to valid releases."""
        env.reset()
        
        # Test that actions respect constraints
        for action_idx in range(11):
            # Get initial state
            initial_volume = env.simulator.current_volume
            
            # Take action
            td = TensorDict({"action": torch.tensor(action_idx)}, batch_size=())
            td = env.step(td)
            
            # Verify volume stays within bounds
            new_volume = env.simulator.current_volume
            assert 0 <= new_volume <= env.v_max
            
            # Reset for next test
            env.reset()
    
    def test_termination_conditions(self, env):
        """Test episode termination conditions."""
        # Test 365-day episodes
        td = env.reset()
        steps = 0
        
        while steps < 400:
            td["action"] = torch.tensor(5, dtype=torch.int64)  # Middle action
            td = env.step(td)
            steps += 1
            if td["next"]["done"].item():
                break
        
        # Should terminate at or before 365 steps
        assert steps <= 365
        assert td["next"]["done"].item()
    
    def test_torchrl_specs(self, env):
        """Test TorchRL spec definitions."""
        # Check observation spec (CompositeSpec with "observation" key)
        assert "observation" in env.observation_spec.keys()
        assert env.observation_spec["observation"].shape == torch.Size([13])
        assert env.observation_spec["observation"].dtype == torch.float32
        
        # Check action spec
        assert env.action_spec.shape == torch.Size([])  # Empty batch_size
        assert env.action_spec.dtype == torch.int64
        assert env.action_spec.space.n == 11
        
        # Check reward spec
        assert env.reward_spec.shape == torch.Size([1])  # Empty batch_size + 1
        assert env.reward_spec.dtype == torch.float32
    
    def test_observation_normalization(self, env):
        """Test that observations are properly normalized."""
        td = env.reset()
        obs = td["observation"]
        
        # Volume percentage should be in [0, 1]
        assert 0 <= obs[0] <= 1
        
        # Sin and cos should be in [-1, 1]
        assert -1 <= obs[1] <= 1  # sin_t
        assert -1 <= obs[2] <= 1  # cos_t
        
        # Forecast values should be normalized (roughly in [0, 2] range)
        for i in range(3, 13):
            assert obs[i] >= 0  # Non-negative
            assert obs[i] < 10  # Reasonable upper bound
    
    def test_reward_calculation(self, env):
        """Test reward function components."""
        td = env.reset()
        
        # Test low release (should trigger environmental penalty)
        td["action"] = torch.tensor(0, dtype=torch.int64)  # 0% release
        td = env.step(td)
        reward_low = td["next"]["reward"].item()
        
        # Test moderate release (should be positive)
        td = env.reset()
        td["action"] = torch.tensor(5, dtype=torch.int64)  # 5% release
        td = env.step(td)
        reward_mid = td["next"]["reward"].item()
        
        # Moderate release should generally give better reward than no release
        # (unless volume is very low)
        if env.simulator.current_volume > env.v_min * 2:
            assert reward_mid > reward_low
    
    def test_discrete_action_mapping(self, env):
        """Test that discrete actions map correctly to release percentages."""
        env.reset()
        
        # Test each action maps to correct percentage
        for action_idx in range(11):
            expected_pct = action_idx * 0.01  # 0%, 1%, 2%, ..., 10%
            
            # Get release amount for this action
            release = env._map_discrete_action(action_idx)
            
            # Calculate actual percentage
            releasable = max(0, env.simulator.current_volume - env.v_min)
            if releasable > 0:
                actual_pct = release / releasable
                assert abs(actual_pct - expected_pct) < 1e-6
            else:
                # If no releasable water, release should be 0
                assert release == 0
    
    def test_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        # Create two environments with same seed
        env1 = DiscreteReservoirEnv()
        env2 = DiscreteReservoirEnv()
        
        env1.set_seed(42)
        env2.set_seed(42)
        
        # Reset both
        td1 = env1.reset()
        td2 = env2.reset()
        
        # Should have same initial observation
        assert torch.allclose(td1["observation"], td2["observation"])
        
        # Take same actions
        for _ in range(10):
            action = torch.tensor(5, dtype=torch.int64)
            td1["action"] = action
            td2["action"] = action
            
            td1 = env1.step(td1)
            td2 = env2.step(td2)
            
            # Should have same observations and rewards
            assert torch.allclose(td1["next"]["observation"], td2["next"]["observation"])
            assert torch.allclose(td1["next"]["reward"], td2["next"]["reward"])