# Issue #2: Implement Basic Reservoir Simulator Class

## Approach

Based on the requirements in issue #2 and the planning document, I'll create a core reservoir simulator that:

1. **Manages reservoir state**: Track water volume with safety constraints
2. **Generates inflows**: Use sinusoidal pattern with noise as specified
3. **Enforces constraints**: Ensure releases respect safety limits
4. **Provides clean API**: Methods for reset, step, get_state, and is_safe

## Implementation Details

### Class Structure
- Initialize with reservoir parameters (v_max, v_min, v_dead, v_safe)
- Track current volume, timestep, and inflow history
- Generate realistic inflow patterns

### Water Balance Equation
```
new_volume = current_volume + inflow - release
```

### Safety Constraints
- Release cannot exceed available water above dead storage
- Release limited to 10% of current volume per day
- Volume must stay within operational bounds

### Inflow Generation
- Base sinusoidal pattern for seasonal variation
- Gaussian noise for day-to-day variability
- 10-day forecast with increasing uncertainty

## Test Strategy
1. Unit tests for each method
2. Integration test running 365-day simulation
3. Boundary condition tests (empty/full reservoir)
4. Safety constraint validation