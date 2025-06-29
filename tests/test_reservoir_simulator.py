"""Unit tests for ReservoirSimulator class."""

import numpy as np
import pytest

from rl_demo.envs.reservoir_simulator import ReservoirSimulator


class TestReservoirSimulator:
    """Test suite for ReservoirSimulator."""

    def test_initialization(self):
        """Test proper initialization of reservoir parameters."""
        sim = ReservoirSimulator(v_max=1000, v_min=100, v_dead=50)

        assert sim.v_max == 1000
        assert sim.v_min == 100
        assert sim.v_dead == 50
        assert sim.v_safe == 850  # 85% of v_max
        assert sim.initial_volume == 500  # 50% of v_max by default
        assert sim.current_volume == 500
        assert sim.current_step == 0

    def test_initialization_with_custom_volume(self):
        """Test initialization with custom initial volume."""
        sim = ReservoirSimulator(v_max=1000, initial_volume=750)
        assert sim.initial_volume == 750
        assert sim.current_volume == 750

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        # Dead storage >= min volume
        with pytest.raises(ValueError, match="Dead storage must be less than minimum operational volume"):
            ReservoirSimulator(v_max=1000, v_min=100, v_dead=100)

        # Min volume >= max capacity
        with pytest.raises(ValueError, match="Minimum volume must be less than maximum capacity"):
            ReservoirSimulator(v_max=1000, v_min=1000, v_dead=50)

    def test_reset(self):
        """Test reset functionality."""
        sim = ReservoirSimulator(v_max=1000, initial_volume=600, random_seed=42)

        # Change state
        sim.current_volume = 800
        sim.current_step = 100

        # Reset
        state = sim.reset()

        assert sim.current_volume == 600
        assert sim.current_step == 0
        assert len(sim.inflow_history) == 1
        assert state["volume"] == 600
        assert state["volume_pct"] == 0.6
        assert state["day_of_year"] == 0
        assert state["is_safe"] is True

    def test_water_balance(self):
        """Test basic water balance equation."""
        sim = ReservoirSimulator(v_max=1000, initial_volume=500, random_seed=42)
        sim.reset()

        # Get initial volume
        initial_volume = sim.current_volume

        # Step with zero release
        state, inflow, done, info = sim.step(0)

        # Check water balance: new_volume = old_volume + inflow - release
        expected_volume = initial_volume + inflow - 0
        assert abs(sim.current_volume - expected_volume) < 1e-6
        assert not done

    def test_release_constraints(self):
        """Test that release constraints are properly applied."""
        sim = ReservoirSimulator(v_max=1000, v_min=100, v_dead=50, initial_volume=200)
        sim.reset()

        # Test 1: Cannot release more than available above dead storage
        state, inflow, done, info = sim.step(200)  # Try to release all water
        assert info["actual_release"] <= 150  # Max 200 - 50 (dead storage)
        assert info["constraint_violated"] is True

        # Test 2: Cannot release more than 10% of current volume
        sim.current_volume = 1000
        state, inflow, done, info = sim.step(200)  # Try to release 20%
        assert info["actual_release"] <= 100  # Max 10% of 1000
        assert info["constraint_violated"] is True

        # Test 3: Valid release within constraints
        sim.current_volume = 500
        state, inflow, done, info = sim.step(30)  # 6% of volume
        assert abs(info["actual_release"] - 30) < 1e-6
        assert info["constraint_violated"] is False

    def test_inflow_generation(self):
        """Test inflow generation with seasonal pattern."""
        sim = ReservoirSimulator(v_max=1000, random_seed=42)
        sim.reset()

        # Collect inflows for a full year
        inflows = []
        for _ in range(365):
            state, inflow, done, info = sim.step(0)
            inflows.append(inflow)

        inflows = np.array(inflows)

        # Check basic properties
        assert len(inflows) == 365
        assert np.all(inflows >= 0)  # All non-negative
        assert np.mean(inflows) > 50  # Reasonable average (base is 0.1 * 1000 = 100)
        assert np.mean(inflows) < 150

        # Check seasonal variation exists
        summer_avg = np.mean(inflows[150:240])  # Summer months
        winter_avg = np.mean(inflows[0:90])  # Winter months
        assert abs(summer_avg - winter_avg) > 10  # Should see seasonal difference

    def test_termination_conditions(self):
        """Test episode termination conditions."""
        # Test 1: Terminate after 365 days
        sim = ReservoirSimulator(v_max=1000, random_seed=42)
        sim.reset()

        done = False
        for i in range(365):
            state, inflow, done, info = sim.step(0)
            if i < 364:
                assert not done

        assert done  # Should terminate after 365 steps

        # Test 2: Terminate on critical drought
        sim = ReservoirSimulator(v_max=1000, v_dead=50, initial_volume=60)
        sim.reset()

        # Release water to go below dead storage
        state, inflow, done, info = sim.step(50)
        if sim.current_volume <= sim.v_dead:
            assert done

    def test_is_safe(self):
        """Test safety check method."""
        sim = ReservoirSimulator(v_max=1000, v_dead=50)

        # Test various volumes
        sim.current_volume = 500
        assert sim.is_safe() is True

        sim.current_volume = 50  # At dead storage
        assert sim.is_safe() is False

        sim.current_volume = 851  # Above flood threshold
        assert sim.is_safe() is False

        sim.current_volume = 850  # At flood threshold
        assert sim.is_safe() is True

    def test_get_state(self):
        """Test state representation."""
        sim = ReservoirSimulator(v_max=1000, initial_volume=750)
        sim.reset()

        state = sim.get_state()

        assert "volume" in state
        assert "volume_pct" in state
        assert "day_of_year" in state
        assert "is_safe" in state

        assert state["volume"] == 750
        assert state["volume_pct"] == 0.75
        assert state["day_of_year"] == 0
        assert state["is_safe"] is True

    def test_forecast(self):
        """Test inflow forecast generation."""
        sim = ReservoirSimulator(v_max=1000, random_seed=42)
        sim.reset()

        # Get 10-day forecast
        forecast = sim.get_forecast(days_ahead=10)

        assert len(forecast) == 10
        assert np.all(forecast >= 0)  # All non-negative

        # Check uncertainty increases (variance should increase)
        # This is probabilistic, but with fixed seed should be consistent

    def test_full_year_simulation(self):
        """Test running a full year simulation without errors."""
        sim = ReservoirSimulator(v_max=1000, v_min=100, v_dead=50, random_seed=42)
        sim.reset()

        volumes = []
        releases = []

        # Run for 365 days with varying release strategy
        for day in range(365):
            state = sim.get_state()
            volumes.append(state["volume"])

            # Simple release strategy: release more when volume is high
            if state["volume_pct"] > 0.8:
                release = 0.08 * state["volume"]
            elif state["volume_pct"] > 0.6:
                release = 0.05 * state["volume"]
            else:
                release = 0.02 * state["volume"]

            state, inflow, done, info = sim.step(release)
            releases.append(info["actual_release"])

            if done and day < 364:
                # Should only terminate early if constraints violated
                assert state["volume"] <= sim.v_dead or state["volume"] >= sim.v_max
                break

        # Verify simulation completed reasonably
        assert len(volumes) == 365
        assert min(volumes) > sim.v_dead  # Never went below dead storage
        assert max(volumes) <= sim.v_max  # Never exceeded capacity (but can reach it)

    def test_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        # First run
        sim1 = ReservoirSimulator(v_max=1000, random_seed=123)
        sim1.reset()

        states1 = []
        for _ in range(10):
            state, inflow, done, info = sim1.step(50)
            states1.append(state["volume"])

        # Second run with same seed
        sim2 = ReservoirSimulator(v_max=1000, random_seed=123)
        sim2.reset()

        states2 = []
        for _ in range(10):
            state, inflow, done, info = sim2.step(50)
            states2.append(state["volume"])

        # Should be identical
        assert np.allclose(states1, states2)
