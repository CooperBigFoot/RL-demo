"""Core reservoir simulation logic without TorchRL integration."""

import numpy as np


class ReservoirSimulator:
    """Simulates a single reservoir with water balance dynamics and constraints.

    This class handles the core reservoir physics including:
    - Water balance equation: new_volume = current_volume + inflow - release
    - Inflow generation with seasonal patterns and noise
    - Safety constraints on releases
    - State tracking over time
    """

    def __init__(
        self,
        v_max: float = 1000.0,
        v_min: float = 100.0,
        v_dead: float = 50.0,
        initial_volume: float | None = None,
        random_seed: int | None = None,
    ):
        """Initialize reservoir with capacity parameters.

        Args:
            v_max: Maximum reservoir capacity (m³)
            v_min: Minimum operational volume (m³)
            v_dead: Dead storage volume that cannot be released (m³)
            initial_volume: Starting volume, defaults to 50% of v_max
            random_seed: Random seed for reproducible inflow generation
        """
        # Reservoir parameters
        self.v_max = v_max
        self.v_min = v_min
        self.v_dead = v_dead
        self.v_safe = 0.85 * v_max  # 85% for flood safety

        # Validate parameters
        if v_dead >= v_min:
            raise ValueError("Dead storage must be less than minimum operational volume")
        if v_min >= v_max:
            raise ValueError("Minimum volume must be less than maximum capacity")

        # Set initial volume
        self.initial_volume = initial_volume or (0.5 * v_max)

        # Random number generator
        self.rng = np.random.RandomState(random_seed)

        # State variables
        self.current_volume = self.initial_volume
        self.current_step = 0
        self.inflow_history = []

        # Inflow parameters
        self.base_inflow = 0.1 * v_max  # Average daily inflow
        self.inflow_amplitude = 0.05 * v_max  # Seasonal variation
        self.inflow_noise_std = 0.02 * v_max  # Daily noise

    def reset(self) -> dict[str, float]:
        """Reset reservoir to initial conditions.

        Returns:
            Dictionary containing initial state information
        """
        self.current_volume = self.initial_volume
        self.current_step = 0
        self.inflow_history = []

        # Generate initial inflow
        inflow = self._generate_inflow()
        self.inflow_history.append(inflow)

        return self.get_state()

    def step(self, release_amount: float) -> tuple[dict[str, float], float, bool, dict]:
        """Execute one timestep of reservoir simulation.

        Args:
            release_amount: Desired water release (m³)

        Returns:
            Tuple of (new_state, inflow, done, info)
            - new_state: Current reservoir state
            - inflow: Generated inflow for this timestep
            - done: Whether episode should terminate
            - info: Additional diagnostic information
        """
        # Generate inflow for current timestep
        inflow = self._generate_inflow()
        self.inflow_history.append(inflow)

        # Apply safety constraints to release
        safe_release = self._apply_release_constraints(release_amount)

        # Update water balance
        self.current_volume = self.current_volume + inflow - safe_release

        # Enforce capacity limits
        self.current_volume = np.clip(self.current_volume, 0, self.v_max)

        # Check termination conditions
        done = self._check_termination()

        # Increment timestep
        self.current_step += 1

        # Collect info
        info = {
            "requested_release": release_amount,
            "actual_release": safe_release,
            "constraint_violated": abs(release_amount - safe_release) > 1e-6,
        }

        return self.get_state(), inflow, done, info

    def get_state(self) -> dict[str, float]:
        """Get current reservoir state.

        Returns:
            Dictionary containing:
            - volume: Current water volume (m³)
            - volume_pct: Volume as percentage of max capacity
            - day_of_year: Current day in annual cycle
            - is_safe: Whether reservoir is below flood threshold
        """
        day_of_year = self.current_step % 365

        return {
            "volume": self.current_volume,
            "volume_pct": self.current_volume / self.v_max,
            "day_of_year": day_of_year,
            "is_safe": self.is_safe(),
        }

    def is_safe(self) -> bool:
        """Check if reservoir is within safe operating bounds.

        Returns:
            True if volume is between dead storage and flood threshold
        """
        return self.v_dead < self.current_volume <= self.v_safe

    def _generate_inflow(self) -> float:
        """Generate realistic inflow with seasonal pattern and noise.

        Uses sinusoidal base pattern for seasonal variation plus
        Gaussian noise for daily variability.

        Returns:
            Generated inflow value (m³)
        """
        # Seasonal pattern
        day_of_year = self.current_step % 365
        seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)

        # Base inflow with seasonal variation
        base = self.base_inflow + self.inflow_amplitude * seasonal_factor

        # Add noise
        noise = self.rng.normal(0, self.inflow_noise_std)

        # Ensure non-negative
        inflow = max(0, base + noise)

        return inflow

    def _apply_release_constraints(self, requested_release: float) -> float:
        """Apply safety constraints to requested release.

        Constraints:
        1. Cannot release more than available above dead storage
        2. Cannot release more than 10% of current volume per day
        3. Release must be non-negative

        Args:
            requested_release: Desired release amount (m³)

        Returns:
            Constrained release amount (m³)
        """
        # Available water above dead storage
        available = max(0, self.current_volume - self.v_dead)

        # Daily release limit (10% of current volume)
        daily_limit = 0.10 * self.current_volume

        # Apply all constraints
        safe_release = min(requested_release, available, daily_limit)

        # Ensure non-negative
        safe_release = max(0, safe_release)

        return safe_release

    def _check_termination(self) -> bool:
        """Check if simulation should terminate.

        Termination conditions:
        1. Catastrophic flood (volume > v_max)
        2. Critical drought (volume < v_dead)
        3. Reached 365 days (full year)

        Returns:
            True if any termination condition is met
        """
        # Note: Volume is already clipped to [0, v_max] so flood won't occur
        # But we check if it would have exceeded without clipping
        if self.current_volume <= self.v_dead:
            return True

        # Check if we've completed 365 steps (after incrementing in step())
        return self.current_step >= 364

    def get_forecast(self, days_ahead: int = 10) -> np.ndarray:
        """Generate inflow forecast with increasing uncertainty.

        Args:
            days_ahead: Number of days to forecast

        Returns:
            Array of forecasted inflows
        """
        forecast = []

        for day in range(days_ahead):
            # Base seasonal pattern
            future_day = (self.current_step + day) % 365
            seasonal_factor = np.sin(2 * np.pi * future_day / 365)
            base = self.base_inflow + self.inflow_amplitude * seasonal_factor

            # Increasing uncertainty with forecast horizon
            uncertainty_factor = 1.0 + 0.1 * day
            noise_std = self.inflow_noise_std * uncertainty_factor
            noise = self.rng.normal(0, noise_std)

            forecast.append(max(0, base + noise))

        return np.array(forecast)
