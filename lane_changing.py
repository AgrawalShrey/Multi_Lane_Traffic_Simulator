"""MOBIL lane-changing model with a hard physical-overlap constraint.

Vehicle position convention:
    x = front position
    occupied interval = [x-L, x]

MOBIL may provide the incentive to change lane, but it can NEVER override
the geometric rule that the subject vehicle must not overlap a vehicle
already occupying the target lane.
"""


class MOBILLaneChanger:
    def __init__(self, params):
        self.p = params

    def neighbors(self, vehicle, sim, lane):
        ahead = [
            v for v in sim.traffic
            if v is not vehicle
            and v.lane == lane
            and v.x > vehicle.x
        ]

        behind = [
            v for v in sim.traffic
            if v is not vehicle
            and v.lane == lane
            and v.x <= vehicle.x
        ]

        leader = min(
            ahead,
            key=lambda v: v.x
        ) if ahead else None

        follower = max(
            behind,
            key=lambda v: v.x
        ) if behind else None

        return leader, follower

    def safe(self, ego, target_leader, target_follower, sim):
        # ------------------------------------------------------------
        # HARD GEOMETRIC SAFETY RULE
        #
        # Before even considering acceleration safety, the ego interval
        # [x-L, x] must not overlap any vehicle interval [xi-li, xi]
        # in the target lane.
        # ------------------------------------------------------------
        if sim.target_lane_has_overlap(
            ego,
            ego.lane if target_follower is None else None
        ):
            pass

        # The actual target lane is checked in choose_lane(), where it is
        # known. This method therefore performs the dynamic IDM safety test.
        if target_follower is None:
            return True

        acc = sim.idm_acceleration(
            target_follower,
            ego
        )

        return (
            acc
            >= -sim.p["mobil_safe_deceleration"]
        )

    def incentive(
        self,
        ego,
        current_lane,
        target_lane,
        sim
    ):
        old_leader, old_follower = self.neighbors(
            ego, sim, current_lane
        )

        new_leader, new_follower = self.neighbors(
            ego, sim, target_lane
        )

        ego_old = sim.idm_acceleration(
            ego,
            old_leader
        )

        ego_new = sim.idm_acceleration(
            ego,
            new_leader
        )

        new_follower_before = (
            sim.idm_acceleration(
                new_follower,
                new_leader
            )
            if new_follower
            else 0.0
        )

        new_follower_after = (
            sim.idm_acceleration(
                new_follower,
                ego
            )
            if new_follower
            else 0.0
        )

        old_follower_before = (
            sim.idm_acceleration(
                old_follower,
                ego
            )
            if old_follower
            else 0.0
        )

        old_follower_after = (
            sim.idm_acceleration(
                old_follower,
                old_leader
            )
            if old_follower
            else 0.0
        )

        p = sim.p["mobil_politeness"]

        return (
            (ego_new - ego_old)
            + p * (
                (new_follower_after - new_follower_before)
                + (old_follower_after - old_follower_before)
            )
        )

    def choose_lane(self, ego, sim):
        current = ego.lane

        candidates = []

        if current > 0:
            candidates.append(current - 1)

        if current < int(sim.p["num_lanes"]) - 1:
            candidates.append(current + 1)

        best_lane = None
        best_score = sim.p["mobil_threshold"]

        for target in candidates:

            # ========================================================
            # HARD PHYSICAL OCCUPANCY CHECK
            #
            # Subject vehicle:
            #       [x-L, x]
            #
            # Target vehicle i:
            #       [xi-li, xi]
            #
            # If these intervals overlap, lane change is IMPOSSIBLE.
            #
            # We also enforce the minimum longitudinal clearance s0
            # because a zero-gap lane change is physically undesirable
            # even when the intervals merely touch.
            # ========================================================
            if sim.target_lane_has_overlap(
                ego,
                target,
                include_minimum_gap=True
            ):
                continue

            new_leader, new_follower = self.neighbors(
                ego,
                sim,
                target
            )

            # Dynamic MOBIL safety for the new follower.
            if new_follower is not None:
                follower_acc = sim.idm_acceleration(
                    new_follower,
                    ego
                )

                if follower_acc < -sim.p["mobil_safe_deceleration"]:
                    continue

            score = self.incentive(
                ego,
                current,
                target,
                sim
            )

            if score > best_score:
                best_score = score
                best_lane = target

        return best_lane
