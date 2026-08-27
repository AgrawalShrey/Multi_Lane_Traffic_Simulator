"""IDM multi-lane microscopic traffic model with generic vehicle classes.

x is the FRONT position, so a vehicle occupies [x-length, x].
Vehicles in virtual_queue are outside the road and are never visualized.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from lane_changing import MOBILLaneChanger

DEFAULT_VEHICLE_CLASSES = [
    {"name":"Car", "enabled":True, "composition":0.80, "desired_speed":16.66, "speed_sigma":2.78, "length":5.0, "width":2.0,
     "minimum_gap":2.0, "acceleration":1.0, "comfortable_deceleration":1.5, "desired_time_headway":1.0, "acc_exponent":4.0},
    {"name":"Truck", "enabled":True, "composition":0.20, "desired_speed":13.89, "speed_sigma":2.78, "length":8.0, "width":2.5,
     "minimum_gap":2.0, "acceleration":1.0, "comfortable_deceleration":1.5, "desired_time_headway":1.0, "acc_exponent":4.0},
    {"name":"Auto-Rickshaw", "enabled":False, "composition":0.0, "desired_speed":11.0, "speed_sigma":2.0, "length":3.0, "width":1.5,
     "minimum_gap":1.5, "acceleration":1.0, "comfortable_deceleration":1.5, "desired_time_headway":1.0, "acc_exponent":4.0},
    {"name":"Two-Wheeler", "enabled":False, "composition":0.0, "desired_speed":13.0, "speed_sigma":2.5, "length":2.0, "width":0.8,
     "minimum_gap":1.0, "acceleration":1.5, "comfortable_deceleration":2.0, "desired_time_headway":0.8, "acc_exponent":4.0},
    {"name":"Bus", "enabled":False, "composition":0.0, "desired_speed":12.0, "speed_sigma":2.0, "length":12.0, "width":2.8,
     "minimum_gap":2.5, "acceleration":0.8, "comfortable_deceleration":1.5, "desired_time_headway":1.5, "acc_exponent":4.0},
    {"name":"LCV", "enabled":False, "composition":0.0, "desired_speed":14.0, "speed_sigma":2.5, "length":6.0, "width":2.2,
     "minimum_gap":2.0, "acceleration":1.0, "comfortable_deceleration":1.5, "desired_time_headway":1.0, "acc_exponent":4.0},
    {"name":"HCV", "enabled":False, "composition":0.0, "desired_speed":11.0, "speed_sigma":2.0, "length":14.0, "width":2.8,
     "minimum_gap":2.5, "acceleration":0.7, "comfortable_deceleration":1.4, "desired_time_headway":1.5, "acc_exponent":4.0},
]


def normalize_vehicle_classes(classes):
    result = []
    for c in classes or []:
        q = dict(c)
        q["name"] = str(q.get("name", "Custom"))
        q["enabled"] = bool(q.get("enabled", True))
        q["composition"] = max(0.0, float(q.get("composition", 0.0)))
        for k, default in [("desired_speed",15.0),("speed_sigma",2.0),("length",5.0),("width",2.0),("minimum_gap",2.0),("acceleration",1.0),("comfortable_deceleration",1.5),("desired_time_headway",1.0),("acc_exponent",4.0)]:
            q[k] = float(q.get(k, default))
        result.append(q)
    enabled = [c for c in result if c["enabled"]]
    if not enabled:
        result[0]["enabled"] = True; result[0]["composition"] = 1.0; enabled = [result[0]]
    total = sum(c["composition"] for c in enabled)
    if total <= 0:
        for c in enabled: c["composition"] = 1.0 / len(enabled)
    else:
        for c in enabled: c["composition"] /= total
    return result


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
        self.p = dict(params)
        self.p["vehicle_classes"] = normalize_vehicle_classes(self.p.get("vehicle_classes", DEFAULT_VEHICLE_CLASSES))
        self.class_map = {c["name"]: c for c in self.p["vehicle_classes"]}
        self.rng = np.random.RandomState(int(self.p.get("seed", 0)))
        self.lane_changer = MOBILLaneChanger(self.p)
        self.reset()

    def reset(self):
        self.time = 0.0; self.vehicle_index = 1; self.cum_headway = 0.0
        self.traffic = []; self.virtual_queue = []; self.history = []
        self.generated_count = 0; self.departed_count = 0
        self.next_vehicle = self._generate_vehicle()

    def class_parameters(self, vehicle_type):
        return self.class_map.get(vehicle_type, self.class_map[next(iter(self.class_map))])

    def generate_headway(self):
        rate = float(self.p.get("volume", 1000.0)) / 3600.0
        if rate <= 0: return float("inf")
        return -math.log(max(self.rng.rand(), np.finfo(float).tiny)) / rate

    def generate_vehicle_type(self):
        enabled = [c for c in self.p["vehicle_classes"] if c["enabled"]]
        probs = np.array([c["composition"] for c in enabled], float); probs /= probs.sum()
        return enabled[int(self.rng.choice(len(enabled), p=probs))]["name"]

    def generate_desired_speed(self, vehicle_type):
        c = self.class_parameters(vehicle_type); mean = c["desired_speed"]; sigma = c["speed_sigma"]
        if sigma <= 0: return mean
        for _ in range(100):
            ds = mean + sigma * self.rng.normal()
            if mean - sigma <= ds <= mean + sigma: return max(0.1, ds)
        return max(0.1, mean)

    def _generate_vehicle(self):
        self.cum_headway += self.generate_headway(); typ = self.generate_vehicle_type(); c = self.class_parameters(typ)
        v = Vehicle(self.vehicle_index, self.cum_headway, self.generate_desired_speed(typ), c["length"], c["width"], vehicle_type=typ)
        self.generated_count = max(self.generated_count, self.vehicle_index); return v

    @staticmethod
    def longitudinal_interval(v): return (v.x - v.length, v.x)

    @staticmethod
    def intervals_overlap(v1, v2):
        r1, f1 = IDMModel.longitudinal_interval(v1); r2, f2 = IDMModel.longitudinal_interval(v2)
        return r1 < f2 and r2 < f1

    def minimum_gap_for(self, v): return self.class_parameters(v.vehicle_type)["minimum_gap"]

    def target_lane_has_overlap(self, vehicle, target_lane, include_minimum_gap=False):
        for other in self.traffic:
            if other is vehicle or other.lane != target_lane: continue
            if self.intervals_overlap(vehicle, other): return True
            if include_minimum_gap:
                r1, f1 = self.longitudinal_interval(vehicle); r2, f2 = self.longitudinal_interval(other)
                gap = r2 - f1 if f1 <= r2 else (r1 - f2 if f2 <= r1 else -1.0)
                if gap < max(self.minimum_gap_for(vehicle), self.minimum_gap_for(other)): return True
        return False

    def lane_has_entry_gap(self, vehicle, lane):
        vehicle.x = vehicle.length
        for other in self.traffic:
            if other.lane != lane: continue
            if self.intervals_overlap(vehicle, other): return False
            r1, f1 = self.longitudinal_interval(vehicle); r2, f2 = self.longitudinal_interval(other)
            gap = r2 - f1 if f1 <= r2 else (r1 - f2 if f2 <= r1 else -1.0)
            if gap < max(self.minimum_gap_for(vehicle), self.minimum_gap_for(other)): return False
        return True

    def neighbors(self, vehicle, lane=None):
        lane = vehicle.lane if lane is None else lane
        ahead = [v for v in self.traffic if v is not vehicle and v.lane == lane and v.x > vehicle.x]
        behind = [v for v in self.traffic if v is not vehicle and v.lane == lane and v.x <= vehicle.x]
        return (min(ahead, key=lambda v:v.x) if ahead else None, max(behind, key=lambda v:v.x) if behind else None)

    def gap(self, follower, leader):
        if leader is None: return 10000.0 - follower.x
        return leader.x - leader.length - follower.x

    def idm_acceleration(self, vehicle, leader=None):
        c = self.class_parameters(vehicle.vehicle_type)
        a = max(c["acceleration"], 1e-6); b = max(c["comfortable_deceleration"], 1e-6); T = c["desired_time_headway"]; s0 = c["minimum_gap"]; delta = c["acc_exponent"]
        if leader is None: dv = vehicle.speed; gap = 10000.0 - vehicle.x
        else: dv = vehicle.speed - leader.speed; gap = self.gap(vehicle, leader)
        if gap <= 0: return -b
        sstar = max(0.01, s0 + vehicle.speed*T + vehicle.speed*dv/(2*math.sqrt(a*b)))
        return a * (1 - (vehicle.speed/max(vehicle.desired_speed,1e-9))**delta - (sstar/gap)**2)

    def placement_safe(self, vehicle, lane):
        if not self.lane_has_entry_gap(vehicle, lane): return False
        leader, _ = self.neighbors(vehicle, lane)
        return leader is None or self.idm_acceleration(vehicle, leader) >= -self.class_parameters(vehicle.vehicle_type)["comfortable_deceleration"]

    def vehicle_placement(self, vehicle):
        vehicle.x = vehicle.length; vehicle.speed = vehicle.desired_speed
        candidates=[]
        for lane in range(int(self.p["num_lanes"])):
            if self.placement_safe(vehicle, lane):
                count=sum(v.lane==lane for v in self.traffic)
                nearest=min((v.x for v in self.traffic if v.lane==lane and v.x>=vehicle.x), default=float("inf"))
                candidates.append((count,-nearest,lane))
        if not candidates:
            for _ in range(2):
                vehicle.speed *= float(self.p.get("speed_reduction_factor",0.5))
                for lane in range(int(self.p["num_lanes"])):
                    if self.placement_safe(vehicle,lane):
                        candidates.append((sum(v.lane==lane for v in self.traffic), -min((v.x for v in self.traffic if v.lane==lane and v.x>=vehicle.x),default=float("inf")), lane))
                if candidates: break
        if candidates:
            candidates.sort(); vehicle.lane=candidates[0][2]; self.traffic.append(vehicle); return True
        vehicle.x=0.0; vehicle.speed=0.0; return False

    def apply_lane_changes(self):
        if not self.p.get("enable_mobil", True) or int(self.p["num_lanes"])<=1: return
        for vehicle in list(self.traffic):
            target=self.lane_changer.choose_lane(vehicle,self)
            if target is not None and not self.target_lane_has_overlap(vehicle,target,True): vehicle.lane=target

    def idm_step(self):
        for vehicle in list(self.traffic):
            leader,_=self.neighbors(vehicle); acc=float(np.clip(self.idm_acceleration(vehicle,leader),-10,8))
            dt=float(self.p["time_step"]); new_speed=max(0.0,vehicle.speed+acc*dt)
            if new_speed==0 and vehicle.speed>0: acc=-vehicle.speed/dt
            vehicle.x += 0.5*(vehicle.speed+new_speed)*dt; vehicle.speed=new_speed; vehicle.acceleration=acc

    def vehicle_q_placement(self):
        if self.virtual_queue and self.vehicle_placement(self.virtual_queue[0]): self.virtual_queue.pop(0)

    def delete_vehicles(self):
        keep=[]
        for v in self.traffic:
            if v.x>=self.p["road_length"]: self.departed_count+=1
            else: keep.append(v)
        self.traffic=keep

    def record_state(self):
        for v in self.traffic:
            leader,_=self.neighbors(v)
            self.history.append({"time":self.time,"vehicle_id":v.vehicle_id,"vehicle_type":v.vehicle_type,"lane":v.lane+1,"arrival_time":v.arrival_time,"desired_speed":v.desired_speed,"length":v.length,"width":v.width,"x":v.x,"speed":v.speed,"acceleration":v.acceleration,"gap":self.gap(v,leader)})

    def step(self):
        if self.time>=self.next_vehicle.arrival_time:
            vehicle=self.next_vehicle
            if self.virtual_queue: self.virtual_queue.append(vehicle)
            elif not self.vehicle_placement(vehicle): self.virtual_queue.append(vehicle)
            self.vehicle_index+=1; self.next_vehicle=self._generate_vehicle()
        self.apply_lane_changes(); self.idm_step(); self.vehicle_q_placement(); self.record_state(); self.delete_vehicles(); self.time+=self.p["time_step"]

    def run(self,duration,callback=None):
        self.reset()
        while self.time<=duration:
            self.step()
            if callback: callback(self)
        return self.to_dataframe()

    def to_dataframe(self): return pd.DataFrame(self.history)

    def summary(self):
        counts={i+1:sum(v.lane==i for v in self.traffic) for i in range(int(self.p["num_lanes"]))}
        return {"simulation_time":self.time,"generated":self.generated_count,"departed":self.departed_count,"on_road":len(self.traffic),"queue":len(self.virtual_queue),"lane_counts":counts}
