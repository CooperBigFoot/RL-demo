"""Example script demonstrating the DiscreteReservoirEnv."""

import torch

from rl_demo.envs import DiscreteReservoirEnv


def main():
    # Create environment
    env = DiscreteReservoirEnv(
        v_max=1000.0,
        v_min=100.0,
        v_dead=50.0,
        initial_volume=500.0,
    )

    # Reset environment
    print("Resetting environment...")
    td = env.reset()
    print(f"Initial observation shape: {td['observation'].shape}")
    print(f"Initial volume percentage: {td['observation'][0]:.2f}")

    # Run a few steps with random actions
    print("\nRunning 10 steps with random actions:")
    for step in range(10):
        # Sample random action
        action = torch.randint(0, 11, (1,))[0]
        td["action"] = action

        # Take step
        td = env.step(td)

        # Extract info from nested structure
        obs = td["next"]["observation"]
        reward = td["next"]["reward"].item()
        done = td["next"]["done"].item()

        print(f"Step {step + 1}: Action={action.item():2d}, Volume%={obs[0]:.2f}, Reward={reward:6.2f}, Done={done}")

        if done:
            print("Episode terminated!")
            break

    print("\nEnvironment test completed successfully!")


if __name__ == "__main__":
    main()
