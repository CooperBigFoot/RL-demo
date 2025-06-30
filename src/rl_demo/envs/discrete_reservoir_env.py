import torch
from tensordict import TensorDict
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from .reservoir_simulator import ReservoirSimulator


class DiscreteReservoirEnv(EnvBase):
    """TorchRL wrapper for discrete-action reservoir control.

    This environment wraps the core ReservoirSimulator to provide:
    - Discrete action space (11 actions: 0%, 1%, ..., 10% release)
    - TensorDict-based state representation (13 dimensions)
    - Reward function balancing hydropower, flood control, and environmental flow
    - TorchRL-compatible specs and interfaces
    """

    def __init__(
        self,
        v_max: float = 1000.0,
        v_min: float = 100.0,
        v_dead: float = 50.0,
        initial_volume: float | None = None,
        device: torch.device | None = None,
        batch_size: torch.Size | None = None,
    ):
        """Initialize the discrete reservoir environment.

        Args:
            v_max: Maximum reservoir capacity (m³)
            v_min: Minimum operational volume (m³)
            v_dead: Dead storage volume that cannot be released (m³)
            initial_volume: Starting volume, defaults to 50% of v_max
            device: Device for tensors (cpu/cuda)
            batch_size: Batch size for vectorized environments
        """
        if batch_size is None:
            batch_size = torch.Size([])
        super().__init__(device=device, batch_size=batch_size)

        # Initialize reservoir simulator
        self.simulator = ReservoirSimulator(
            v_max=v_max,
            v_min=v_min,
            v_dead=v_dead,
            initial_volume=initial_volume,
        )

        # Store reservoir parameters
        self.v_max = v_max
        self.v_min = v_min
        self.v_dead = v_dead
        self.v_safe = 0.85 * v_max  # 85% for flood safety

        # Discrete action space: 11 actions (0%, 1%, ..., 10%)
        self.n_actions = 11
        self.action_percentages = torch.linspace(0, 0.10, self.n_actions)

        # For state normalization
        self.max_historical_inflow = 0.2 * v_max  # Estimate based on simulator params

        # Reward weights
        self.w_hydro = 1.0
        self.w_flood = 10.0
        self.w_env = 5.0

        # Define TorchRL specs
        self._make_specs()

    def _make_specs(self) -> None:
        """Define TorchRL specifications for observations, actions, and rewards."""
        # Observation spec: Use Composite with "observation" key
        self.observation_spec = Composite(
            observation=Bounded(
                low=-2.0,  # Allow for sin/cos and some margin
                high=2.0,  # Normalized values should be in this range
                shape=(13,),  # Just the observation dimensions
                dtype=torch.float32,
                device=self.device,
            ),
            shape=self.batch_size,  # Batch size for the composite spec
        )

        # Action spec: Categorical with 11 discrete actions
        self.action_spec = Categorical(
            n=self.n_actions,
            shape=(),  # Scalar action
            dtype=torch.int64,
            device=self.device,
        )

        # Reward spec: Unbounded scalar
        self.reward_spec = Unbounded(
            shape=(1,),  # Scalar reward
            dtype=torch.float32,
            device=self.device,
        )

        # Done spec
        self.done_spec = Bounded(
            low=False,
            high=True,
            shape=(1,),  # Scalar done
            dtype=torch.bool,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        """Reset the environment to initial state.

        Args:
            tensordict: Optional input tensordict (not used)

        Returns:
            TensorDict containing initial observation and done flag
        """
        # Reset simulator
        state = self.simulator.reset()

        # Get observation
        observation = self._get_observation(state)

        # Create output tensordict
        out = TensorDict(
            {
                "observation": observation,
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return out

    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute one environment step.

        Args:
            tensordict: Must contain "action" key with discrete action index

        Returns:
            TensorDict with new observation, reward, and done flag
        """
        # Extract action
        action_idx = tensordict["action"].item()

        # Map discrete action to release amount
        release_amount = self._map_discrete_action(action_idx)

        # Execute simulator step
        new_state, inflow, done, info = self.simulator.step(release_amount)

        # Calculate reward
        reward = self._calculate_reward(
            release=info["actual_release"],
            new_volume=new_state["volume"],
            old_volume=info["old_volume"],
        )

        # Get new observation
        observation = self._get_observation(new_state)

        # Create output tensordict
        out = TensorDict(
            {
                "observation": observation,
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "info": {"inflow": inflow, "actual_release": info["actual_release"]},
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        return out

    def _set_seed(self, seed: int | None) -> None:
        """Set random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        if seed is not None:
            import numpy as np

            self.simulator.rng = np.random.RandomState(seed)
            torch.manual_seed(seed)

    def _get_observation(self, state: dict) -> torch.Tensor:
        """Convert simulator state to 13-dimensional observation vector.

        Components:
        - v_pct: Current volume percentage (1D)
        - sin_t, cos_t: Cyclical time encoding (2D)
        - inflow_forecast: 10-day forecast (10D)

        Args:
            state: State dictionary from simulator

        Returns:
            13-dimensional observation tensor
        """
        # Cyclical time encoding
        day_of_year = state["day_of_year"]
        theta = torch.tensor(2 * torch.pi * day_of_year / 365, dtype=torch.float32)

        # Get forecast and normalize
        forecast = self.simulator.get_forecast(days_ahead=10)
        normalized_forecast = torch.tensor(forecast / self.max_historical_inflow, dtype=torch.float32)

        # Combine into a single tensor
        observation = torch.cat([
            torch.tensor([state["volume_pct"]], dtype=torch.float32),
            torch.stack([torch.sin(theta), torch.cos(theta)]),
            normalized_forecast
        ]).to(self.device)

        return observation

    def _map_discrete_action(self, action_idx: int) -> float:
        """Convert discrete action index to release amount.

        Args:
            action_idx: Discrete action in {0, 1, 2, ..., 10}

        Returns:
            Actual release amount in m³
        """
        # Get release percentage for this action
        release_pct = self.action_percentages[action_idx].item()

        # Calculate release as percentage of releasable volume
        releasable_volume = max(0, self.simulator.current_volume - self.v_min)
        desired_release = release_pct * releasable_volume

        return desired_release

    def _calculate_reward(self, release: float, new_volume: float, old_volume: float) -> float:
        """Calculate reward based on hydropower, flood control, and environmental flow.

        Args:
            release: Actual water released (m³)
            new_volume: Volume after step (m³)
            old_volume: Volume before step (m³)

        Returns:
            Scalar reward value
        """
        # --- Normalized Hydropower Reward ---
        # Max possible hydro reward: (10% of v_max release) * (head is 100% of v_max)
        max_hydro_reward = (0.10 * self.v_max) * 1.0
        avg_volume = (old_volume + new_volume) / 2
        r_hydro = (release * (avg_volume / self.v_max)) / max_hydro_reward

        # --- Normalized Flood Penalty ---
        p_flood = 0.0
        if new_volume > self.v_safe:
            violation_depth = new_volume - self.v_safe
            # Max possible violation is from v_safe up to v_max
            max_possible_violation = self.v_max - self.v_safe
            p_flood = -1.0 * (violation_depth / max_possible_violation)

        # --- Normalized Environmental Flow Penalty ---
        min_env_flow = 0.01 * self.v_max
        p_env = 0.0
        if release < min_env_flow:
            # Penalty is proportional to the deficit, from 0 to -1
            p_env = -1.0 * ((min_env_flow - release) / min_env_flow)

        # With normalized components, weights are easier to interpret and tune
        # You might start with all weights at 1.0 and adjust from there.
        reward = self.w_hydro * r_hydro + self.w_flood * p_flood + self.w_env * p_env

        return reward

    @staticmethod
    def parse_observation(obs: torch.Tensor) -> dict:
        """Parse observation tensor into named components.
        
        Args:
            obs: 13-dimensional observation tensor
            
        Returns:
            Dictionary with named components:
            - volume_pct: Current volume as fraction of max capacity
            - sin_t: Sine of seasonal time
            - cos_t: Cosine of seasonal time  
            - forecast: 10-day inflow forecast (normalized)
        """
        return {
            "volume_pct": obs[0].item(),
            "sin_t": obs[1].item(),
            "cos_t": obs[2].item(),
            "forecast": obs[3:13]
        }
