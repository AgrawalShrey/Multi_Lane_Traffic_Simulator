# IDM Multi-Lane Traffic Simulator

**Model:** Intelligent Driver Model (IDM) + optional MOBIL lane changing  
**Application type:** Microscopic traffic simulation

## Overview

The **IDM Multi-Lane Traffic Simulator** is a Python-based microscopic traffic simulation application for studying vehicle-level traffic dynamics on a straight, multi-lane road.

Each vehicle has its own position, speed, acceleration, desired speed, dimensions, vehicle type, and lane. Longitudinal motion is governed by the **Intelligent Driver Model (IDM)**. When enabled, discretionary lane-changing decisions are governed by **MOBIL (Minimizing Overall Braking Induced by Lane changes)**.

The application provides a graphical user interface, real-time road visualization, configurable parameters, vehicle trajectories, speed histories, and CSV trajectory export.

---

## 1. Main Features

- User-defined road length, number of lanes, and lane width.
- Default configuration: **2 lanes × 3.5 m**.
- Stochastic vehicle arrivals.
- Car and truck vehicle classes.
- User-defined desired-speed distributions and vehicle dimensions.
- IDM car-following model.
- Optional MOBIL lane-changing model.
- Virtual queue for vehicles unable to enter safely.
- Physical no-overlap constraint for road entry and lane changing.
- Real-time multi-lane visualization.
- Vehicle trajectories and speed histories.
- CSV export of vehicle-level trajectory data.
- Random seed for reproducible experiments.

---


# 2. Intelligent Driver Model (IDM)

## 2.1 Purpose

The **Intelligent Driver Model (IDM)** is a continuous microscopic car-following model. It determines how a vehicle accelerates or decelerates according to its current speed, desired speed, gap to the leader, and relative speed.

A major advantage of IDM is that its parameters have direct behavioral interpretations.

## 2.2 IDM acceleration

The model used in the simulator is

\[
a_i =a\left[1-\left(\frac{v_i}{v_{0,i}}\right)^\delta-\left(\frac{s_i^*}{s_i}\right)^2\right].
\]

Here:

- \(a_i\) = acceleration of vehicle \(i\);
- \(a\) = acceleration parameter;
- \(v_i\) = current speed;
- \(v_{0,i}\) = desired speed;
- \(\delta\) = acceleration exponent;
- \(s_i\) = current net gap to the leader;
- \(s_i^*\) = desired dynamic gap.

The net gap is

\[
s_i =x_{\mathrm{leader}}-L_{\mathrm{leader}}-x_i,
\]

where \(x_i\) is the **front position** of the subject vehicle.

## 2.3 Desired dynamic gap

The desired gap is

\[
s_i^*=s_0+v_iT+\frac{v_i\Delta v_i}{2\sqrt{ab}},
\]

where

\[
\Delta v_i=v_i-v_{\mathrm{leader}}.
\]

Thus,

\[
s_i^*=\underbrace{s_0}_{\text{minimum gap}}+\underbrace{v_iT}_{\text{time-headway term}}+\underbrace{\frac{v_i\Delta v_i}{2\sqrt{ab}}}_{\text{closing-speed term}}.
\]

The desired gap therefore increases with speed and desired time headway, and it increases when the subject vehicle is approaching the leader.

## 2.4 Advantages of IDM

### Interpretable parameters
Parameters such as \(s_0\), \(T\), \(a\), \(b\), and \(v_0\) have clear behavioral meanings.

### Smooth car-following dynamics
IDM provides continuous acceleration and deceleration rather than simple discrete rules.

### Free-road behavior
When no leader constrains the vehicle, it tends toward its desired speed.

### Gap adaptation
Acceleration automatically responds to the available gap and closing speed.

### Easy calibration and sensitivity analysis
Parameters can be changed independently for experiments and teaching.

### Modular longitudinal behavior
IDM can be combined naturally with a separate lane-changing model such as MOBIL.

---

# 3. MOBIL Lane-Changing Model

## 3.1 Purpose

IDM determines how a vehicle behaves **within its current lane**. It does not decide whether the vehicle should move to another lane.

The simulator therefore uses **MOBIL** as an optional lane-changing decision model.

MOBIL evaluates whether a lane change provides sufficient acceleration benefit while considering its effect on surrounding vehicles.

## 3.2 Incentive criterion

The implementation uses the MOBIL acceleration-incentive form

\[
\Delta a_{\mathrm{ego}}+p\left(\Delta a_{\mathrm{new\ follower}}+\Delta a_{\mathrm{old\ follower}}\right)>\Delta a_{\mathrm{th}},
\]

where:

- \(\Delta a_{\mathrm{ego}}\) = change in the subject vehicle's acceleration;
- \(\Delta a_{\mathrm{new\ follower}}\) = acceleration change of the follower in the target lane;
- \(\Delta a_{\mathrm{old\ follower}}\) = acceleration change of the follower in the original lane;
- \(p\) = politeness factor;
- \(\Delta a_{\mathrm{th}}\) = minimum incentive threshold.

The accelerations used in this criterion are calculated using IDM.

## 3.3 Politeness factor

The parameter \(p\) represents how much the subject vehicle considers the effect of its lane change on surrounding vehicles.

For

\[
p=0,
\]

the decision is based primarily on the subject vehicle's own acceleration advantage.

Increasing \(p\) makes the lane-changing driver more considerate of acceleration changes imposed on other vehicles.

---

# 4. Physical Safety Constraint

The simulator adds a hard geometric safety constraint to MOBIL.

The simulator defines \(x\) as the **front position** of a vehicle. A vehicle of length \(L\) therefore occupies

\[
[x-L,\;x].
\]

Another vehicle \(i\) occupies

\[
[x_i-L_i,\;x_i].
\]

A lane change is rejected if

\[
[x-L,x]\cap[x_i-L_i,x_i]\neq\varnothing.
\]

Therefore:

> **MOBIL can never override a physical vehicle-overlap condition.**

The simulator also checks the configured minimum longitudinal clearance.

This prevents vehicles from appearing to collide during lane changes.

---

# 5. Safe Vehicle Entry and Virtual Queue

A generated vehicle is first treated as an arrival candidate. It is inserted into a lane only if the physical entrance position is safe.

If a safe lane is available:

```text
Generated vehicle
       ↓
Physical entry check
       ↓
Successful
       ↓
On-road traffic
       ↓
Visualization
```

If no safe lane is available:

```text
Generated vehicle
       ↓
Physical entry check
       ↓
Unsafe
       ↓
Virtual queue
       ↓
Wait and retry
```

Vehicles in the virtual queue are **not part of the on-road traffic list** and therefore are **not visualized**.

This prevents unrealistic overlapping vehicles at the road entrance under high traffic demand.

---

# 6. IDM + MOBIL Framework

The simulator separates longitudinal and lane-changing behavior:

```text
                  VEHICLE
                     |
          +----------+----------+
          |                     |
          v                     v
         IDM                  MOBIL
          |                     |
          v                     v
  Longitudinal motion     Lane-change decision
          |                     |
          +----------+----------+
                     |
                     v
              Updated state
          x, speed, acceleration,
                  lane
```

### IDM asks

> How should the vehicle accelerate or decelerate in its current lane?

### MOBIL asks

> Is changing to an adjacent lane beneficial and safe?

This separation makes the framework modular and extensible.

---

# 7. Simulation Workflow

At each microscopic time step, the simulator:

1. Processes new vehicle arrivals.
2. Attempts physical road entry.
3. Places blocked vehicles in the virtual queue.
4. Evaluates MOBIL lane-changing decisions if enabled.
5. Applies the hard geometric no-overlap condition.
6. Computes longitudinal acceleration using IDM.
7. Updates speed and position.
8. Attempts to release queued vehicles.
9. Records on-road vehicle trajectories.
10. Removes vehicles that leave the road.
11. Updates the visualization.

---

# 8. Position Convention

The simulator consistently uses

\[
x=\text{front position of vehicle}.
\]

Therefore:

\[
\boxed{\text{occupied interval}=[x-L,x]}
\]

This convention is used for:

- IDM gap calculation;
- vehicle entry;
- lane-changing safety;
- physical overlap detection;
- visualization;
- trajectory output.

---

# 9. User-Configurable Parameters

| Category | Parameter | Meaning |
|---|---|---|
| Road | Road length | Simulated road length |
| Road | Number of lanes | Number of parallel lanes |
| Road | Lane width | Width of each lane |
| Demand | Volume | Arrival demand (veh/h) |
| Demand | Car composition | Fraction of cars |
| Vehicle | Desired speed | Mean and variation by type |
| Vehicle | Length / width | Physical vehicle dimensions |
| IDM | \(s_0\) | Minimum gap |
| IDM | \(T\) | Desired time headway |
| IDM | \(a\) | Acceleration parameter |
| IDM | \(b\) | Comfortable deceleration |
| IDM | \(\delta\) | Acceleration exponent |
| MOBIL | Enable/disable | Activates lane changing |
| MOBIL | \(p\) | Politeness factor |
| MOBIL | Threshold | Minimum lane-change incentive |
| MOBIL | Safe deceleration | Target-follower safety criterion |
| Simulation | Time step | Microscopic simulation interval |
| Simulation | Simulation time | Total simulation duration |
| Randomness | Seed | Reproducibility |

---

# 10. Advantages of the Combined IDM–MOBIL Approach

### IDM provides
- longitudinal car-following;
- speed adaptation;
- gap keeping;
- acceleration and deceleration;
- interpretable behavioral parameters.

### MOBIL provides
- discretionary lane-changing;
- acceleration-based lane-change incentives;
- driver politeness;
- consideration of surrounding vehicles.

### Physical constraints provide
- safe road entry;
- no overlap during lane changes;
- realistic virtual queue formation under high demand.

Together,

\[
\boxed{
\text{Microscopic Traffic Dynamics}
=
\text{IDM Longitudinal Model}
+
\text{MOBIL Lane-Changing Model}
+
\text{Physical Safety Constraints}
}
\]

---

# 11. Output

The simulator can export vehicle-level trajectory data to CSV.

Typical variables include:

- time;
- vehicle ID;
- vehicle type;
- lane;
- arrival time;
- desired speed;
- length;
- width;
- front position \(x\);
- speed;
- acceleration;
- gap.

The output can be analyzed using Python, MATLAB, Excel, or other data-analysis tools.

---

# 12. Intended Applications

The simulator is suitable for:

- microscopic traffic-flow demonstrations;
- IDM parameter sensitivity studies;
- lane-changing experiments;
- congestion and queue formation studies;
- traffic-model teaching;
- comparison of IDM-only and IDM+MOBIL behavior;
- trajectory-data generation;
- preliminary research experiments.

---

# 13. Current Scope and Limitations

The current simulator represents a **straight road with parallel lanes**.

It does not currently include:

- intersections;
- traffic signals;
- curved roads;
- pedestrian interactions;
- complex road networks;
- route choice;
- traffic-light control;
- detailed connected-vehicle communication;
- advanced driver-assistance systems.

These can be incorporated in future versions.

---

# 14. Running the Python Application

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

# 15. Building the Windows EXE

The project includes a PyInstaller build script.

Run:

```bat
build_exe.bat
```

The recommended distribution is the generated:

```text
dist/
└── IDM_Traffic_Simulator/
    └── IDM_Traffic_Simulator.exe
```

Distribute the complete application folder.

A single-file build can also be created using:

```bat
build_onefile.bat
```

---

# 16. Summary

The simulator combines:

\[
\boxed{
\text{IDM}
+
\text{MOBIL}
+
\text{Physical Safety Constraints}
}
\]

IDM governs longitudinal vehicle dynamics, MOBIL governs discretionary lane-changing decisions, and explicit geometric checks prevent physically impossible vehicle overlap during road entry and lane changing.

The result is a configurable and visually accessible microscopic traffic simulation platform for studying car-following, lane changing, congestion, queue formation, lane utilization, and vehicle trajectories.
