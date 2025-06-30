"""
Default configuration for DQN training on the Discrete Reservoir Environment.

This module defines the configuration dataclasses used throughout the training
process, including environment, network, training, and experiment settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import yaml
from pathlib import Path


@dataclass
class EnvironmentConfig:
    """Configuration for the reservoir environment."""
    name: str = "DiscreteReservoir"
    inflow_scenario: str = "mixed"  # Options: stable, seasonal, extreme, mixed
    v_max: float = 1000.0  # Maximum reservoir volume
    v_min: float = 100.0   # Minimum reservoir volume
    device: str = "cpu"    # Device for environment operations


@dataclass
class NetworkConfig:
    """Configuration for the Q-network architecture."""
    hidden_dim: int = 256
    n_layers: int = 2
    activation: str = "relu"
    
    # Input/output dimensions (set based on environment)
    state_dim: int = 13
    action_dim: int = 11


@dataclass
class TrainingConfig:
    """Configuration for DQN training hyperparameters."""
    # Basic training
    total_frames: int = 50000
    frames_per_batch: int = 256
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # DQN specific
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_frames: int = 10000
    target_update_freq: int = 500
    
    # Replay buffer
    buffer_size: int = 10000
    min_replay_size: int = 1000
    
    # Optimization
    optimizer: str = "adam"
    grad_clip: Optional[float] = None
    weight_decay: float = 0.0


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    log_interval: int = 100        # Log metrics every N frames
    eval_interval: int = 1000      # Run evaluation every N frames
    eval_episodes: int = 5         # Number of episodes for evaluation
    checkpoint_interval: int = 5000 # Save checkpoint every N frames
    
    # Directories
    log_dir: str = "runs/dqn_reservoir"
    checkpoint_dir: str = "checkpoints"
    
    # Options
    save_best: bool = True         # Save best model separately
    log_gradients: bool = False    # Log gradient statistics
    log_actions: bool = True       # Log action distributions
    log_q_values: bool = True      # Log Q-value statistics


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    exp_name: str = "dqn_reservoir"
    seed: int = 42
    resume: Optional[str] = None   # Path to checkpoint to resume from
    notes: str = ""                # Experiment notes/description
    tags: list[str] = field(default_factory=list)  # Tags for filtering
    
    # Auto-generated fields
    exp_id: Optional[str] = None   # Will be auto-generated
    git_hash: Optional[str] = None # Will be auto-populated


@dataclass
class DQNConfig:
    """Complete configuration for DQN training."""
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DQNConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DQNConfig":
        """Create configuration from a dictionary."""
        config = cls()
        
        # Update each sub-config if present
        if "environment" in config_dict:
            config.environment = EnvironmentConfig(**config_dict["environment"])
        if "network" in config_dict:
            config.network = NetworkConfig(**config_dict["network"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "logging" in config_dict:
            config.logging = LoggingConfig(**config_dict["logging"])
        if "experiment" in config_dict:
            config.experiment = ExperimentConfig(**config_dict["experiment"])
            
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to a YAML file."""
        config_dict = self.to_dict()
        
        # Create directory if needed
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "environment": asdict(self.environment),
            "network": asdict(self.network),
            "training": asdict(self.training),
            "logging": asdict(self.logging),
            "experiment": asdict(self.experiment)
        }
    
    def update_from_args(self, args: Dict[str, Any]):
        """Update configuration from command-line arguments.
        
        Args are expected to be flat with dots for nesting, e.g.:
        {"training.lr": 0.001, "network.hidden_dim": 512}
        """
        for key, value in args.items():
            if value is None:
                continue
                
            # Split nested keys
            parts = key.split('.')
            
            # Navigate to the correct sub-config
            obj = self
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    raise ValueError(f"Invalid config key: {key}")
            
            # Set the value
            final_key = parts[-1]
            if hasattr(obj, final_key):
                setattr(obj, final_key, value)
            else:
                raise ValueError(f"Invalid config key: {key}")
    
    def validate(self):
        """Validate configuration consistency."""
        # Check device availability
        import torch
        if self.environment.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            self.environment.device = "cpu"
        
        # Validate epsilon decay
        if self.training.epsilon_frames > self.training.total_frames:
            print("Warning: epsilon_frames > total_frames, adjusting epsilon_frames")
            self.training.epsilon_frames = self.training.total_frames
        
        # Ensure directories exist
        from pathlib import Path
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)


def get_default_config() -> DQNConfig:
    """Get the default configuration."""
    return DQNConfig()