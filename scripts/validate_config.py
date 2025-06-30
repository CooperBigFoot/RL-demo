#!/usr/bin/env python3
"""
Validate a DQN configuration file.

This script checks that a configuration file is valid and displays
the resolved configuration after applying any command-line overrides.
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from rl_demo.configs.default_config import DQNConfig, get_default_config


def main():
    parser = argparse.ArgumentParser(
        description="Validate and display a DQN configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("config", type=str, help="Path to config file to validate")
    parser.add_argument("--show-diff", action="store_true",
                        help="Show differences from default configuration")
    parser.add_argument("--output", type=str, default=None,
                        help="Save resolved config to file")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        print(f"Loading config from: {args.config}")
        config = DQNConfig.from_yaml(args.config)
        config.validate()
        print("✓ Configuration is valid!")
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    
    # Display configuration
    print("\nResolved Configuration:")
    print("-" * 50)
    config_dict = config.to_dict()
    print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
    
    # Show differences from default if requested
    if args.show_diff:
        print("\nDifferences from default:")
        print("-" * 50)
        default_config = get_default_config()
        default_dict = default_config.to_dict()
        
        differences = find_differences(default_dict, config_dict)
        if differences:
            for path, (default_val, config_val) in differences.items():
                print(f"{path}:")
                print(f"  default: {default_val}")
                print(f"  config:  {config_val}")
        else:
            print("No differences from default configuration.")
    
    # Save to file if requested
    if args.output:
        config.to_yaml(args.output)
        print(f"\nSaved resolved configuration to: {args.output}")


def find_differences(dict1, dict2, path=""):
    """Find differences between two nested dictionaries."""
    differences = {}
    
    # Check all keys in dict1
    for key in dict1:
        current_path = f"{path}.{key}" if path else key
        
        if key not in dict2:
            differences[current_path] = (dict1[key], None)
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # Recursively check nested dictionaries
            nested_diffs = find_differences(dict1[key], dict2[key], current_path)
            differences.update(nested_diffs)
        elif dict1[key] != dict2[key]:
            differences[current_path] = (dict1[key], dict2[key])
    
    # Check for keys in dict2 not in dict1
    for key in dict2:
        if key not in dict1:
            current_path = f"{path}.{key}" if path else key
            differences[current_path] = (None, dict2[key])
    
    return differences


if __name__ == "__main__":
    main()