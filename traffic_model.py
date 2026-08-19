"""IDM multi-lane microscopic traffic model.

Important physical convention:
    x = FRONT position of a vehicle.
Therefore the longitudinal occupied interval is:
    [x - length, x]

A vehicle in the virtual queue has x=0 and is NOT part of `traffic`.
Only vehicles in `traffic` are simulated, recorded, and visualized.

The model therefore never visualizes a queued vehicle until it has passed
the physical entrance-placement checks and has actually entered a lane.
"""

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

from lane_changing import MOBILLaneChanger


@dataclass
class Vehicle:
    vehicle_id: int
    arrival_time: float
    desired_speed: float
    length: float
    width: float
    x: float = 0.0
    speed: float = 0.0
    lane: int = 0
    vehicle_type: str = "Car"
    acceleration: float = 0.0


class IDMModel:
    def __init__(self, params):
        self.p = params
        self.rng = np.random.RandomState(params.get("seed", 0))
        self.lane_changer = MOBILLaneChanger(self.p)
        self.reset()

    # ================================================================
    # Initialization
    # ================================================================
    def reset(self):
        self.time = 0.0
        self.vehicle_index = 1
        self.cum_headway = 0.0

        # IMPORTANT:
        # traffic = vehicles physically on the road only.
        # virtual_queue = vehicles waiting outside the road.
        self.traffic = []
        self.virtual_queue = []

        self.history = []
        self.generated_count = 0
        self.departed_count = 0

        self.next_vehicle = self._generate_vehicle()

    # ================================================================
    # Vehicle generation
    # ================================================================
    def generate_headway(self):
        rate = self.p["volume"] / 3600.0

        if rate <= 0:
            return float("inf")

        u = max(self.rng.rand(), np.finfo(float).tiny)
        return -math.log(u) / rate

    def generate_vehicle_type(self):
        return (
            "Car"
            if self.rng.rand() <= self.p["car_composition"]
            else "Truck"
        )

    def generate_desired_speed(self, vehicle_type):
        if vehicle_type == "Car":
            mean = self.p["car_ds_mean"]
            sigma = self.p["car_ds_sigma"]
        else:
            mean = self.p["truck_ds_mean"]
            sigma = self.p["truck_ds_sigma"]

        while True:
            u1 = max(self.rng.rand(), np.finfo(float).tiny)
            u2 = self.rng.rand()

            z = (
                math.sqrt(-2.0 * math.log(u1))
                * math.cos(2.0 * math.pi * u2)
            )

            ds = mean + sigma * z

            if mean - sigma <= ds <= mean + sigma:
                return ds

    def generate_dimensions(self, vehicle_type):
        if vehicle_type == "Car":
            return self.p["car_length"], self.p["car_width"]

        return self.p["truck_length"], self.p["truck_width"]

    def _generate_vehicle(self):
        self.cum_headway += self.generate_headway()

        vehicle_type = self.generate_vehicle_type()
        desired_speed = self.generate_desired_speed(vehicle_type)
        length, width = self.generate_dimensions(vehicle_type)

        vehicle = Vehicle(
            vehicle_id=self.vehicle_index,
            arrival_time=self.cum_headway,
            desired_speed=desired_speed,
            length=length,
            width=width,
            x=0.0,
            speed=0.0,
            lane=0,
            vehicle_type=vehicle_type,
        )

        self.generated_count = max(
            self.generated_count,
            self.vehicle_index
        )

        return vehicle

    # ================================================================
    # Geometry / physical occupancy
    # ================================================================
    @staticmethod
    def longitudinal_interval(vehicle):
        """Return [rear, front] using x as the front position."""
        return (
            vehicle.x - vehicle.length,
            vehicle.x
        )

    @staticmethod
    def intervals_overlap(v1, v2):
        """
        True only when the physical occupied intervals have a positive
        overlap.

        Touching at exactly one point is not treated as overlap.
        """
        rear1, front1 = IDMModel.longitudinal_interval(v1)
        rear2, front2 = IDMModel.longitudinal_interval(v2)

        return (
            rear1 < front2
            and rear2 < front1
        )

    def target_lane_has_overlap(
        self,
        vehicle,
        target_lane,
        include_minimum_gap=False
    ):
        """
        Check whether the subject vehicle [x-L, x] physically overlaps
        any vehicle [xi-li, xi] already occupying the target lane.

        This is the mandatory hard geometric constraint for lane changing.

        If include_minimum_gap=True, also require at least s0 clearance.
        """
        for other in self.traffic:
            if other is vehicle or other.lane != target_lane:
                continue

            if self.intervals_overlap(vehicle, other):
                return True

            if include_minimum_gap:
                rear1, front1 = self.longitudinal_interval(vehicle)
                rear2, front2 = self.longitudinal_interval(other)

                if front1 <= rear2:
                    gap = rear2 - front1
                elif front2 <= rear1:
                    gap = rear1 - front2
                else:
                    gap = -1.0

                if gap < self.p["minimum_gap"]:
                    return True

        return False

    def lane_has_entry_gap(self, vehicle, lane):
        """
        Physical entry check.

        A newly generated vehicle is placed with its front at x=L.
        Its occupied interval is [0, L].

        It may enter only if it does not overlap any vehicle already in
        that lane and has at least the model minimum gap to the nearest
        downstream vehicle.
        """
        vehicle.x = vehicle.length

        for other in self.traffic:
            if other.lane != lane:
                continue

            # Explicit physical overlap test.
            if self.intervals_overlap(vehicle, other):
                return False

            # Calculate physical clearance between the two intervals.
            rear_v, front_v = self.longitudinal_interval(vehicle)
            rear_o, front_o = self.longitudinal_interval(other)

            if front_v <= rear_o:
                gap = rear_o - front_v
            elif front_o <= rear_v:
                gap = rear_v - front_o
            else:
                return False

            if gap < self.p["minimum_gap"]:
                return False

        return True

    # ================================================================
    # Neighbours
    # ================================================================
    def neighbors(self, vehicle, lane=None):
        if lane is None:
            lane = vehicle.lane

        ahead = [
            v for v in self.traffic
            if v is not vehicle
            and v.lane == lane
            and v.x > vehicle.x
        ]

        behind = [
            v for v in self.traffic
            if v is not vehicle
            and v.lane == lane
            and v.x <= vehicle.x
        ]

        leader = min(ahead, key=lambda v: v.x) if ahead else None
        follower = max(behind, key=lambda v: v.x) if behind else None

        return leader, follower

    def gap(self, follower, leader):
        if leader is None:
            return 10000.0 - follower.x

        return leader.x - leader.length - follower.x

    # ================================================================
    # IDM
    # ================================================================
    def idm_acceleration(self, vehicle, leader=None):
        a = self.p["acceleration"]
        b = self.p["comfortable_deceleration"]
        T = self.p["desired_time_headway"]
        s0 = self.p["minimum_gap"]
        delta = self.p["acc_exponent"]

        if leader is None:
            relative_speed = vehicle.speed
            actual_gap = 10000.0 - vehicle.x
        else:
            relative_speed = vehicle.speed - leader.speed
            actual_gap = self.gap(vehicle, leader)

        if actual_gap <= 0:
            return -b

        dynamic_term = (
            vehicle.speed * relative_speed
            / (2.0 * math.sqrt(a * b))
        )

        desired_gap = (
            s0
            + vehicle.speed * T
            + dynamic_term
        )

        return a * (
            1.0
            - (vehicle.speed / max(vehicle.desired_speed, 1e-9)) ** delta
            - (desired_gap / actual_gap) ** 2
        )

    # ================================================================
    # Entrance placement
    # ================================================================
    def placement_safe(self, vehicle, lane):
        """
        BOTH conditions are required:

        1. Hard geometry:
           [x-L, x] cannot overlap any vehicle in the lane.

        2. IDM insertion condition:
           acceleration must not require more than the comfortable
           deceleration.
        """
        if not self.lane_has_entry_gap(vehicle, lane):
            return False

        leader, _ = self.neighbors(vehicle, lane)

        if leader is None:
            return True

        acc = self.idm_acceleration(vehicle, leader)

        return acc >= -self.p["comfortable_deceleration"]

    def vehicle_placement(self, vehicle):
        """
        Attempt physical entry into the road.

        A vehicle is NOT added to self.traffic until placement succeeds.

        Consequently:
            virtual_queue -> invisible/off-road
            successful placement -> visible/on-road
        """
        vehicle.x = vehicle.length
        vehicle.speed = vehicle.desired_speed

        n_lanes = int(self.p["num_lanes"])

        if not self.traffic:
            vehicle.lane = 0
            self.traffic.append(vehicle)
            return True

        # Try every lane.
        candidates = []

        for lane in range(n_lanes):
            if self.placement_safe(vehicle, lane):
                count = sum(
                    v.lane == lane
                    for v in self.traffic
                )

                nearest = min(
                    (
                        v.x
                        for v in self.traffic
                        if v.lane == lane
                        and v.x >= vehicle.x
                    ),
                    default=float("inf")
                )

                candidates.append(
                    (count, -nearest, lane)
                )

        if candidates:
            candidates.sort()
            vehicle.lane = candidates[0][2]
            self.traffic.append(vehicle)
            return True

        # Preserve the original MATLAB two-attempt speed reduction.
        for _ in range(2):
            vehicle.speed *= self.p["speed_reduction_factor"]

            candidates = []

            for lane in range(n_lanes):
                if self.placement_safe(vehicle, lane):
                    count = sum(
                        v.lane == lane
                        for v in self.traffic
                    )

                    nearest = min(
                        (
                            v.x
                            for v in self.traffic
                            if v.lane == lane
                            and v.x >= vehicle.x
                        ),
                        default=float("inf")
                    )

                    candidates.append(
                        (count, -nearest, lane)
                    )

            if candidates:
                candidates.sort()
                vehicle.lane = candidates[0][2]
                self.traffic.append(vehicle)
                return True

        # It remains queued. Reset its position so it is explicitly
        # outside the simulated road.
        vehicle.x = 0.0
        vehicle.speed = 0.0

        return False

    # ================================================================
    # MOBIL
    # ================================================================
    def apply_lane_changes(self):
        if (
            not self.p["enable_mobil"]
            or int(self.p["num_lanes"]) <= 1
        ):
            return

        # Evaluate vehicles one at a time. Before actually changing lane,
        # perform the hard geometric check again because an earlier
        # accepted change may have occupied the target lane.
        for vehicle in list(self.traffic):
            target = self.lane_changer.choose_lane(
                vehicle,
                self
            )

            if target is None:
                continue

            # Mandatory final geometric check.
            if self.target_lane_has_overlap(
                vehicle,
                target,
                include_minimum_gap=True
            ):
                continue

            vehicle.lane = target

    # ================================================================
    # Longitudinal update
    # ================================================================
    def idm_step(self):
        for vehicle in list(self.traffic):
            leader, _ = self.neighbors(vehicle)

            acc = self.idm_acceleration(
                vehicle,
                leader
            )

            new_speed = (
                vehicle.speed
                + acc * self.p["time_step"]
            )

            if new_speed < 0:
                new_speed = 0.0
                acc = (
                    -vehicle.speed
                    / self.p["time_step"]
                )

            vehicle.x += (
                0.5
                * (vehicle.speed + new_speed)
                * self.p["time_step"]
            )

            vehicle.speed = new_speed
            vehicle.acceleration = acc

    # ================================================================
    # Queue
    # ================================================================
    def vehicle_q_placement(self):
        """
        Only the first queued vehicle is tested, matching the original
        MATLAB vehicleQPlacement logic.

        The queue itself is NEVER appended to self.traffic unless entry
        succeeds. Therefore queued vehicles cannot be visualized.
        """
        if not self.virtual_queue:
            return

        vehicle = self.virtual_queue[0]

        if self.vehicle_placement(vehicle):
            self.virtual_queue.pop(0)

    # ================================================================
    # Exit
    # ================================================================
    def delete_vehicles(self):
        remaining = []

        for vehicle in self.traffic:
            if vehicle.x >= self.p["road_length"]:
                self.departed_count += 1
            else:
                remaining.append(vehicle)

        self.traffic = remaining

    # ================================================================
    # Recording
    # ================================================================
    def record_state(self):
        for v in self.traffic:
            leader, _ = self.neighbors(v)

            self.history.append({
                "time": self.time,
                "vehicle_id": v.vehicle_id,
                "vehicle_type": v.vehicle_type,
                "lane": v.lane + 1,
                "arrival_time": v.arrival_time,
                "desired_speed": v.desired_speed,
                "length": v.length,
                "width": v.width,
                "x": v.x,
                "speed": v.speed,
                "acceleration": v.acceleration,
                "gap": self.gap(v, leader),
            })

    # ================================================================
    # Simulation step
    # ================================================================
    def step(self):
        # A generated vehicle is only an arrival candidate.
        # It is NOT visualized until successfully inserted.
        if self.time >= self.next_vehicle.arrival_time:
            vehicle = self.next_vehicle

            if self.virtual_queue:
                self.virtual_queue.append(vehicle)
            elif not self.vehicle_placement(vehicle):
                self.virtual_queue.append(vehicle)

            self.vehicle_index += 1
            self.next_vehicle = self._generate_vehicle()

        # MOBIL is evaluated before longitudinal movement.
        self.apply_lane_changes()

        # IDM longitudinal movement.
        self.idm_step()

        # Try to release only the first queued vehicle.
        self.vehicle_q_placement()

        # Record only vehicles physically on the road.
        self.record_state()

        # Remove vehicles that have left.
        self.delete_vehicles()

        self.time += self.p["time_step"]

    def run(self, duration, callback=None):
        self.reset()

        while self.time <= duration:
            self.step()

            if callback:
                callback(self)

        return self.to_dataframe()

    def to_dataframe(self):
        return pd.DataFrame(self.history)

    def summary(self):
        lane_counts = {
            i + 1: sum(
                v.lane == i
                for v in self.traffic
            )
            for i in range(int(self.p["num_lanes"]))
        }

        return {
            "simulation_time": self.time,
            "generated": self.generated_count,
            "departed": self.departed_count,
            "on_road": len(self.traffic),
            "queue": len(self.virtual_queue),
            "lane_counts": lane_counts,
        }
