# IDM Multi-Lane Traffic Simulator

**Model:** Intelligent Driver Model (IDM) + optional MOBIL lane changing  
**Application:** Microscopic traffic simulation and model calibration

## Overview

The **IDM Multi-Lane Traffic Simulator** is a Python-based microscopic traffic simulation application for studying vehicle-level traffic dynamics on a straight, multi-lane road.

Longitudinal vehicle motion is governed by **IDM**, while discretionary lane changes can be governed by **MOBIL**. The application supports configurable road, demand, vehicle, IDM, MOBIL, and simulation parameters, real-time visualization, trajectories, speed histories, and CSV export.

A separate **model calibration module** estimates IDM and MOBIL parameters from observed vehicle trajectory data.

---

## 1. Main Features

- User-defined road length, number of lanes, and lane width.
- Default configuration: **2 lanes × 3.5 m**.
- Stochastic vehicle arrivals.
- Multiple vehicle classes (e.g., Car, Truck, Auto-Rickshaw, Two-Wheeler, or user-defined classes).
- Class-specific desired speeds and vehicle dimensions.
- IDM car-following model.
- Optional MOBIL lane-changing model.
- Virtual queue for vehicles unable to enter safely.
- Physical no-overlap constraint for road entry and lane changing.
- Real-time multi-lane visualization.
- Vehicle trajectories and speed histories.
- CSV trajectory export.
- Reproducible simulations using a random seed.

---

# 2. IDM Car-Following Model

IDM determines the acceleration of vehicle \(i\):

\[
a_i =
a\left[
1-\left(\frac{v_i}{v_{0,i}}\right)^\delta
-\left(\frac{s_i^*}{s_i}\right)^2
\right].
\]

The desired dynamic gap is

\[s_i^*=s_0+v_iT+\frac{v_i\Delta v_i}{2\sqrt{ab}},\]

where

\[\Delta v_i=v_i-v_{\mathrm{leader}}.\]

Parameters:

- \(v_0\): desired speed
- \(s_0\): minimum gap
- \(T\): desired time headway
- \(a\): acceleration
- \(b\): comfortable deceleration
- \(\delta\): acceleration exponent

### Why IDM?

IDM provides smooth, interpretable car-following behavior. Its parameters have direct behavioral meanings and can be calibrated or varied independently, making it suitable for microscopic traffic studies and sensitivity analysis.

---

# 3. MOBIL Lane-Changing Model

MOBIL evaluates whether a lane change provides sufficient acceleration benefit while considering its effect on surrounding vehicles.

The implemented incentive criterion is

\[\Delta a_{\mathrm{ego}}+p\left(\Delta a_{\mathrm{new\ follower}}+\Delta a_{\mathrm{old\ follower}}\right)>\Delta a_{\mathrm{th}},\]

where \(p\) is the politeness factor and \(\Delta a_{\mathrm{th}}\) is the incentive threshold.

IDM is used to calculate the relevant accelerations.

### Why MOBIL?

MOBIL provides a simple and interpretable mechanism for discretionary lane changing while accounting for both the subject vehicle's benefit and the effect on surrounding traffic.

---

# 4. Physical Safety and Vehicle Entry

The simulator uses the front-position convention:

\[x=\text{front position}.\]

A vehicle of length \(L\) occupies

\[[x-L,x].\]

A lane change is rejected if this interval overlaps the occupied interval of a vehicle in the target lane:

\[[x-L,x]\cap[x_i-L_i,x_i]\neq\varnothing.\]

The configured minimum clearance is also checked.

Vehicles that cannot safely enter the road are placed in a **virtual queue**. Queued vehicles are not part of the on-road traffic list and are therefore not visualized until successful entry.

---

# 5. Model Calibration

The application provides a separate **Model Calibration** mode at startup:

```text
Start
 ├── Default / Simulation
 └── Model Calibration
```

Calibration uses observed vehicle trajectory data and estimates model parameters before running the simulator.

### Required trajectory fields

The calibration CSV should contain:

```text
time
ID
position
vehicle_class
speed
Lane
```

Recommended additional fields:

```text
length
width
```

Common alternative column names can also be mapped by the calibration module.

### Calibration procedure

1. Load and validate the trajectory CSV.
2. Determine the simulation time step.
3. Identify vehicle classes.
4. Construct leader–follower observations.
5. Calculate observed gaps and speeds.
6. Estimate IDM parameters for each vehicle class.
7. Estimate MOBIL parameters when lane-changing observations are available.
8. Evaluate calibration error.
9. Save the calibrated parameter profile.
10. Apply the profile to the simulator.

### Calibrated IDM parameters

For each vehicle class:

\[\boxed{v_0,\;s_0,\;T,\;a,\;b,\;\delta}\]

### Calibrated MOBIL parameters

\[\boxed{p,\;\Delta a_{\mathrm{th}},\;b_{\mathrm{safe}}}\]

MOBIL calibration is optional and requires lane-changing information in the trajectory data.

The calibration module is intended to provide class-specific behavioral parameters for heterogeneous traffic simulation.

---

# 6. Multiple Vehicle Classes

The simulator is not restricted to Car and Truck.

Users can define multiple classes, for example:

```text
Car
Truck
Auto-Rickshaw
Two-Wheeler
Bus
LCV
HCV
```

Each class can have its own:

- composition;
- desired-speed distribution;
- vehicle length;
- vehicle width;
- IDM parameters.

This allows the simulator to represent heterogeneous traffic and to apply class-specific calibrated parameters.

---

# 7. Simulation Workflow

At each time step:

1. Process vehicle arrivals.
2. Attempt safe road entry.
3. Place blocked vehicles in the virtual queue.
4. Evaluate MOBIL lane changes if enabled.
5. Apply geometric no-overlap checks.
6. Calculate IDM acceleration.
7. Update vehicle speed and position.
8. Attempt queued-vehicle entry.
9. Record trajectories.
10. Remove vehicles leaving the road.
11. Update visualization.

---

# 8. User-Configurable Parameters

| Category | Parameters |
|---|---|
| Road | Length, number of lanes, lane width |
| Demand | Volume, vehicle-class composition |
| Vehicle | Class, desired speed, length, width |
| IDM | \(s_0, T, a, b, \delta\) |
| MOBIL | Enable/disable, \(p\), incentive threshold, safe deceleration |
| Simulation | Time step, simulation time |
| Randomness | Random seed |

---

# 9. Output

The simulator provides:

- real-time road visualization;
- vehicle trajectories;
- speed histories;
- simulation status;
- CSV trajectory export.

Typical trajectory variables include:

```text
time
vehicle ID
vehicle class
lane
position
speed
acceleration
gap
vehicle dimensions
```

Calibration results can be saved as a parameter profile and used for subsequent simulations.

---

# 10. Intended Applications

- Microscopic traffic-flow research
- IDM parameter calibration
- Heterogeneous traffic modelling
- Lane-changing studies
- Congestion and queue analysis
- Model sensitivity analysis
- Traffic-model teaching
- Vehicle trajectory generation

---

# 11. Current Scope

The current version models a **straight multi-lane road**.

It does not currently include intersections, traffic signals, curved roads, pedestrian interactions, route choice, or complex road networks.

---

# 12. Running the Application

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python main.py
```

At startup, select either:

- **Default / Simulation** — configure and run the simulator.
- **Model Calibration** — estimate model parameters from trajectory data.

---

# 13. Windows EXE

Build the recommended Windows application using:

```bat
build_exe.bat
```

The generated application is located at:

```text
dist/
└── IDM_Traffic_Simulator/
    └── IDM_Traffic_Simulator.exe
```

Distribute the complete application folder.

For a single-file executable, use:

```bat
build_onefile.bat
```

---

## Summary

\[\boxed{\text{IDM}+\text{MOBIL}+\text{Calibration}+\text{Physical Safety Constraints}}\]

The simulator combines interpretable microscopic car-following, discretionary lane changing, class-specific heterogeneous vehicle behavior, trajectory-based model calibration, and physical vehicle-occupancy constraints in a single configurable application.
