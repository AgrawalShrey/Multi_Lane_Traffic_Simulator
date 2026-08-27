"""IDM + MOBIL calibration module and calibration GUI.

Calibration methodology follows Punzo, Zheng & Montanino (2021):
- trajectory data are used directly;
- spacing is retained as a primary measure of performance;
- NRMSE(s,v) is the default because acceleration derived from trajectory
  speed can be noisy; NRMSE(s,v,a) is available when acceleration is reliable;
- a population-based genetic algorithm is used for parameter search;
- the model and data use the same fixed-step ballistic integration scheme.

The calibration is intentionally separated from the simulation engine.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from PySide6.QtCore import Qt, QThread, Signal
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
        QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
        QVBoxLayout, QWidget, QTextEdit
    )
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


IDM_BOUNDS = {
    "desired_speed": (5.0, 40.0),
    "minimum_gap": (0.2, 10.0),
    "desired_time_headway": (0.3, 3.0),
    "acceleration": (0.2, 4.0),
    "comfortable_deceleration": (0.2, 5.0),
    "acc_exponent": (1.0, 8.0),
}

MOBIL_BOUNDS = {
    "mobil_politeness": (0.0, 1.0),
    "mobil_threshold": (0.0, 2.0),
    "mobil_safe_deceleration": (0.5, 5.0),
}

ALIASES = {
    "time": {"time", "t", "timestamp", "simtime", "timestep", "time_s"},
    "position": {"position", "x", "pos", "frontposition", "front_position", "x_m"},
    "vehicle_id": {"id", "vehicleid", "vehicle_id", "vehid", "vehicle"},
    "vehicle_class": {"vehicleclass", "vehicle_class", "class", "type", "vehicletype", "vehicle_type"},
    "speed": {"speed", "v", "velocity", "speed_ms", "speed_mps"},
    "lane": {"lane", "laneno", "lane_id", "laneid"},
    "length": {"length", "vehiclelength", "vehicle_length", "length_m"},
    "width": {"width", "vehiclewidth", "vehicle_width", "width_m"},
}


def _norm_name(x: str) -> str:
    return "".join(ch for ch in str(x).strip().lower() if ch.isalnum())


def _resolve_columns(df: pd.DataFrame):
    normalized = {_norm_name(c): c for c in df.columns}
    out = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if _norm_name(alias) in normalized:
                out[canonical] = normalized[_norm_name(alias)]
                break
    required = ["time", "position", "vehicle_id", "vehicle_class", "speed", "lane"]
    missing = [x for x in required if x not in out]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing) +
            ". Required: time, position, ID, vehicle class, speed, Lane."
        )
    rename = {v: k for k, v in out.items()}
    return df.rename(columns=rename).copy(), out


def load_trajectory_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df, _ = _resolve_columns(df)
    for c in ["time", "position", "vehicle_id", "speed", "lane"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["vehicle_class"] = df["vehicle_class"].astype(str).str.strip()
    df = df.dropna(subset=["time", "position", "vehicle_id", "speed", "lane", "vehicle_class"])
    df = df.sort_values(["vehicle_id", "time"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("The CSV contains no usable trajectory rows.")
    return df


def infer_dt(df: pd.DataFrame) -> float:
    d = df.groupby("vehicle_id")["time"].diff().dropna()
    d = d[d > 0]
    return float(d.median()) if not d.empty else 0.1


def derive_acceleration(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    df = df.copy()
    df["acceleration_obs"] = np.nan
    for _, idx in df.groupby("vehicle_id").groups.items():
        ix = np.asarray(list(idx), dtype=int)
        ix = ix[np.argsort(df.loc[ix, "time"].to_numpy())]
        v = df.loc[ix, "speed"].to_numpy(float)
        if len(ix) >= 3:
            a = np.gradient(v, dt)
        elif len(ix) == 2:
            a = np.array([(v[1] - v[0]) / dt] * 2)
        else:
            a = np.zeros(len(ix))
        df.loc[ix, "acceleration_obs"] = a
    return df


def build_following_records(df: pd.DataFrame, class_lengths: dict[str, float]) -> dict[str, list[dict]]:
    """Create follower/leader trajectory pairs using same-time lane ordering."""
    records: dict[str, list[dict]] = {}
    work = df.sort_values(["time", "lane", "position"]).copy()
    for _, frame in work.groupby("time", sort=False):
        for _, lane_frame in frame.groupby("lane", sort=False):
            lane_frame = lane_frame.sort_values("position")
            rows = list(lane_frame.itertuples(index=False))
            for j in range(len(rows) - 1):
                follower = rows[j]
                leader = rows[j + 1]
                if follower.vehicle_id == leader.vehicle_id:
                    continue
                fclass = str(follower.vehicle_class)
                lclass = str(leader.vehicle_class)
                leader_length = float(
                    getattr(leader, "length", np.nan)
                    if hasattr(leader, "length") else np.nan
                )
                if not np.isfinite(leader_length):
                    leader_length = float(class_lengths.get(lclass, 5.0))
                spacing = float(leader.position - leader_length - follower.position)
                if spacing <= 0:
                    continue
                rec = records.setdefault(fclass, [])
                rec.append({
                    "follower_id": follower.vehicle_id,
                    "leader_id": leader.vehicle_id,
                    "time": float(follower.time),
                    "spacing_obs": spacing,
                    "speed_obs": float(follower.speed),
                    "acceleration_obs": float(getattr(follower, "acceleration_obs", np.nan)),
                    "follower_x": float(follower.position),
                    "leader_x": float(leader.position),
                    "leader_speed": float(leader.speed),
                    "leader_length": leader_length,
                    "follower_lane": float(follower.lane),
                })
    return records


def _nrmse(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if not np.any(mask):
        return 1e6
    o = obs[mask]
    e = sim[mask] - o
    denom = math.sqrt(max(float(np.mean(o ** 2)), 1e-12))
    return math.sqrt(float(np.mean(e ** 2))) / denom


def _simulate_pair(times, follower_x0, follower_v0, leader_x, leader_v, leader_length, params, dt):
    n = len(times)
    xs = np.empty(n)
    vs = np.empty(n)
    accs = np.empty(n)
    xs[0] = follower_x0
    vs[0] = max(0.0, follower_v0)
    accs[0] = 0.0

    a = max(params["acceleration"], 1e-4)
    b = max(params["comfortable_deceleration"], 1e-4)
    T = params["desired_time_headway"]
    s0 = params["minimum_gap"]
    delta = params["acc_exponent"]
    v0 = max(params["desired_speed"], 0.1)

    for k in range(1, n):
        lx = float(leader_x[k - 1])
        lv = float(leader_v[k - 1])
        gap = lx - leader_length - xs[k - 1]
        gap = max(gap, 0.05)
        dv = vs[k - 1] - lv
        s_star = s0 + vs[k - 1] * T + vs[k - 1] * dv / (2.0 * math.sqrt(a * b))
        # Standard IDM can yield a negative desired dynamic gap during a strong
        # closing/opening transition; a small floor keeps the numerical model stable.
        s_star = max(0.01, s_star)
        acc = a * (1.0 - (vs[k - 1] / v0) ** delta - (s_star / gap) ** 2)
        # Prevent optimizer candidates from generating numerical explosions.
        acc = float(np.clip(acc, -10.0, 8.0))
        vnew = max(0.0, vs[k - 1] + acc * dt)
        xs[k] = xs[k - 1] + 0.5 * (vs[k - 1] + vnew) * dt
        vs[k] = vnew
        accs[k] = acc
    return xs, vs, accs


@dataclass
class CalibrationResult:
    vehicle_class: str
    params: dict
    objective: float
    n_points: int
    n_trajectories: int


def _prepare_class_sequences(class_records, df_class, class_lengths):
    sequences = []
    for vid, follower in df_class.groupby("vehicle_id"):
        follower = follower.sort_values("time")
        # Keep periods for which a leader exists. For every timestamp, find
        # nearest leader ahead in the same lane.
        t = follower["time"].to_numpy(float)
        x = follower["position"].to_numpy(float)
        v = follower["speed"].to_numpy(float)
        lane = follower["lane"].to_numpy()
        if len(t) < 5:
            continue
        leader_rows = []
        for tt, xx, ll in zip(t, x, lane):
            candidates = df_class[(np.isclose(df_class["time"], tt)) & (df_class["lane"] == ll) & (df_class["position"] > xx)]
            # df_class only has same class; instead use all data outside here later.
            leader_rows.append(candidates.iloc[0] if not candidates.empty else None)
    return sequences


class IDMCalibrator:
    def __init__(self, df: pd.DataFrame, dt: float, objective: str = "NRMSE(s,v)",
                 population: int = 50, generations: int = 30, seed: int = 0,
                 max_trajectories_per_class: int = 30):
        df, _ = _resolve_columns(df.copy())
        self.df = derive_acceleration(df, dt)
        self.dt = dt
        self.objective = objective
        self.population = max(10, int(population))
        self.generations = max(5, int(generations))
        self.rng = np.random.RandomState(seed)
        self.max_trajectories = max(1, int(max_trajectories_per_class))
        self.class_lengths = self._class_lengths()
        self.sequences = self._make_sequences()

    def _class_lengths(self):
        out = {}
        if "length" in self.df.columns:
            for cls, g in self.df.groupby("vehicle_class"):
                vals = pd.to_numeric(g["length"], errors="coerce").dropna()
                if not vals.empty:
                    out[str(cls)] = float(vals.median())
        return out

    def _make_sequences(self):
        """Build complete follower/leader series, interpolating leader states."""
        data = self.df
        by_id = {vid: g.sort_values("time") for vid, g in data.groupby("vehicle_id")}
        # For each timestamp and lane, identify immediate leader by position.
        leader_map = {}
        for _, frame in data.groupby("time", sort=False):
            for _, lf in frame.groupby("lane", sort=False):
                lf = lf.sort_values("position")
                rows = list(lf.itertuples(index=False))
                for j in range(len(rows) - 1):
                    follower, leader = rows[j], rows[j + 1]
                    leader_map[(follower.vehicle_id, float(follower.time))] = leader.vehicle_id

        seq_by_class = {}
        for vid, follower in by_id.items():
            cls = str(follower.iloc[0].vehicle_class)
            leader_ids = [leader_map.get((vid, float(t))) for t in follower["time"]]
            valid = [lid is not None and lid != vid for lid in leader_ids]
            if sum(valid) < 5:
                continue
            t_all = follower["time"].to_numpy(float)
            x_all = follower["position"].to_numpy(float)
            v_all = follower["speed"].to_numpy(float)
            a_all = follower["acceleration_obs"].to_numpy(float)
            # Extract contiguous following segments where the same leader exists.
            start = None
            for k, lid in enumerate(leader_ids + [None]):
                if lid is not None and start is None:
                    start = k
                if (lid is None or k == len(leader_ids)) and start is not None:
                    end = k
                    if end - start >= 5:
                        seg_t = t_all[start:end]
                        seg_x = x_all[start:end]
                        seg_v = v_all[start:end]
                        seg_a = a_all[start:end]
                        leader_id = leader_ids[start]
                        lg = by_id.get(leader_id)
                        if lg is not None:
                            lt = lg["time"].to_numpy(float)
                            lx = lg["position"].to_numpy(float)
                            lv = lg["speed"].to_numpy(float)
                            if len(lt) >= 2:
                                lxi = np.interp(seg_t, lt, lx)
                                lvi = np.interp(seg_t, lt, lv)
                                llength = float(self.class_lengths.get(str(lg.iloc[0].vehicle_class), 5.0))
                                if "length" in lg.columns:
                                    llv = pd.to_numeric(lg["length"], errors="coerce").dropna()
                                    if not llv.empty:
                                        llength = float(llv.median())
                                spacing = lxi - llength - seg_x
                                good = np.isfinite(spacing) & (spacing > 0.05) & np.isfinite(seg_v)
                                if np.sum(good) >= 5:
                                    seq_by_class.setdefault(cls, []).append({
                                        "vehicle_id": vid,
                                        "time": seg_t[good],
                                        "x": seg_x[good],
                                        "v": seg_v[good],
                                        "a": seg_a[good],
                                        "leader_x": lxi[good],
                                        "leader_v": lvi[good],
                                        "leader_length": llength,
                                    })
                    start = None
        # Long sequences are useful, but cap the number of trajectories for GUI responsiveness.
        for cls in seq_by_class:
            seq_by_class[cls].sort(key=lambda s: len(s["time"]), reverse=True)
            seq_by_class[cls] = seq_by_class[cls][:self.max_trajectories]
        return seq_by_class

    def _decode(self, z):
        names = list(IDM_BOUNDS)
        return {name: float(value) for name, value in zip(names, z)}

    def objective_value(self, z, sequences):
        p = self._decode(z)
        s_obs, s_sim, v_obs, v_sim, a_obs, a_sim = [], [], [], [], [], []
        for seq in sequences:
            xs, vs, accs = _simulate_pair(
                seq["time"], seq["x"][0], seq["v"][0],
                seq["leader_x"], seq["leader_v"], seq["leader_length"], p, self.dt
            )
            s_sim.extend(seq["leader_x"] - seq["leader_length"] - xs)
            s_obs.extend(seq["leader_x"] - seq["leader_length"] - seq["x"])
            v_sim.extend(vs)
            v_obs.extend(seq["v"])
            a_sim.extend(accs)
            a_obs.extend(seq["a"])
        if len(s_obs) < 5:
            return 1e6
        ns = _nrmse(s_obs, s_sim)
        nv = _nrmse(v_obs, v_sim)
        if self.objective == "NRMSE(s,v,a)":
            na = _nrmse(a_obs, a_sim)
            if not np.isfinite(na) or na > 100:
                return 0.5 * ns + 0.5 * nv
            return (ns + nv + na) / 3.0
        return 0.5 * ns + 0.5 * nv

    def genetic_optimize(self, sequences, progress: Callable[[int, int, float], None] | None = None):
        names = list(IDM_BOUNDS)
        lo = np.array([IDM_BOUNDS[n][0] for n in names], float)
        hi = np.array([IDM_BOUNDS[n][1] for n in names], float)
        pop = self.rng.uniform(lo, hi, size=(self.population, len(names)))
        scores = np.array([self.objective_value(x, sequences) for x in pop])
        elite_n = max(2, self.population // 5)
        for gen in range(self.generations):
            order = np.argsort(scores)
            pop, scores = pop[order], scores[order]
            new_pop = [pop[i].copy() for i in range(elite_n)]
            while len(new_pop) < self.population:
                # Tournament selection
                ia, ib = self.rng.randint(0, elite_n, size=2)
                pa = pop[ia] if scores[ia] <= scores[ib] else pop[ib]
                ia, ib = self.rng.randint(0, elite_n, size=2)
                pb = pop[ia] if scores[ia] <= scores[ib] else pop[ib]
                mask = self.rng.rand(len(names)) < 0.5
                child = np.where(mask, pa, pb).copy()
                mutation_mask = self.rng.rand(len(names)) < (1.0 / len(names))
                child += mutation_mask * self.rng.normal(0, 0.12, len(names)) * (hi - lo)
                # Occasional stronger mutation improves global exploration.
                if self.rng.rand() < 0.10:
                    j = self.rng.randint(0, len(names))
                    child[j] = self.rng.uniform(lo[j], hi[j])
                child = np.clip(child, lo, hi)
                new_pop.append(child)
            pop = np.asarray(new_pop)
            scores = np.array([self.objective_value(x, sequences) for x in pop])
            best = float(np.min(scores))
            if progress:
                progress(gen + 1, self.generations, best)
        idx = int(np.argmin(scores))
        return self._decode(pop[idx]), float(scores[idx])

    def calibrate(self, progress=None):
        results = []
        classes = list(self.sequences)
        for ci, cls in enumerate(classes):
            seqs = self.sequences[cls]
            def pgen(g, total, best, ci=ci, cls=cls):
                if progress:
                    overall = int(((ci + g / total) / max(len(classes), 1)) * 100)
                    progress(overall, f"Calibrating {cls}: generation {g}/{total}; objective={best:.5f}")
            params, obj = self.genetic_optimize(seqs, pgen)
            results.append(CalibrationResult(cls, params, obj, sum(len(s["time"]) for s in seqs), len(seqs)))
        return results

    # ------------------------------------------------------------------
    # MOBIL calibration
    # ------------------------------------------------------------------
    def _idm_acc_row(self, row, leader, params):
        v = float(row.speed)
        v0 = max(params["desired_speed"], 0.1)
        a = max(params["acceleration"], 1e-6)
        b = max(params["comfortable_deceleration"], 1e-6)
        T = params["desired_time_headway"]
        s0 = params["minimum_gap"]
        delta = params["acc_exponent"]
        if leader is None:
            gap = 10000.0 - float(row.position); dv = v
        else:
            llen = float(leader.get("length", self.class_lengths.get(str(leader.vehicle_class), 5.0)))
            gap = float(leader.position) - llen - float(row.position)
            dv = v - float(leader.speed)
        if gap <= 0: return -b
        sstar = max(0.01, s0 + v*T + v*dv/(2*math.sqrt(a*b)))
        return a*(1-(v/v0)**delta-(sstar/gap)**2)

    @staticmethod
    def _lane_neighbors(frame, x, lane):
        lf = frame[frame["lane"] == lane]
        ahead = lf[lf["position"] > x].sort_values("position")
        behind = lf[lf["position"] <= x].sort_values("position", ascending=False)
        return (ahead.iloc[0] if not ahead.empty else None,
                behind.iloc[0] if not behind.empty else None)

    def _mobil_score_at_snapshot(self, frame, ego, current_lane, target_lane, idm_params, p, threshold, safe_dec):
        x = float(ego.position)
        old_leader, old_follower = self._lane_neighbors(frame, x, current_lane)
        new_leader, new_follower = self._lane_neighbors(frame, x, target_lane)
        # Hard geometric constraint: [x-L, x] may not overlap a target-lane vehicle.
        ego_len = float(ego.get("length", self.class_lengths.get(str(ego.vehicle_class), 5.0)))
        ego_rear, ego_front = x-ego_len, x
        for _, other in frame[frame["lane"] == target_lane].iterrows():
            olen = float(other.get("length", self.class_lengths.get(str(other.vehicle_class), 5.0)))
            orr, off = float(other.position)-olen, float(other.position)
            if ego_rear < off and orr < ego_front:
                return None
            gap = (orr-ego_front) if ego_front <= orr else ((ego_rear-off) if off <= ego_rear else -1)
            if gap < max(idm_params["minimum_gap"], self.class_lengths.get(str(other.vehicle_class), 2.0)*0 + idm_params["minimum_gap"]):
                return None
        ego_old = self._idm_acc_row(ego, old_leader, idm_params)
        ego_new = self._idm_acc_row(ego, new_leader, idm_params)
        # Dynamic safety for target follower.
        if new_follower is not None:
            fcls = str(new_follower.vehicle_class)
            fp = self.calibrated_params.get(fcls, idm_params)
            follower_after = self._idm_acc_row(new_follower, ego, fp)
            if follower_after < -safe_dec:
                return None
            follower_before = self._idm_acc_row(new_follower, new_leader, fp)
        else:
            follower_after = follower_before = 0.0
        if old_follower is not None:
            fcls = str(old_follower.vehicle_class)
            fp = self.calibrated_params.get(fcls, idm_params)
            old_before = self._idm_acc_row(old_follower, ego, fp)
            old_after = self._idm_acc_row(old_follower, old_leader, fp)
        else:
            old_before = old_after = 0.0
        score = (ego_new-ego_old) + p*((follower_after-follower_before)+(old_after-old_before))
        return score - threshold

    def mobil_objective(self, z, max_events=2500):
        """Objective for observed lane-change/no-change decisions.

        Each observation immediately before a lane transition is treated as a
        positive event when the next observed lane is different. A stratified
        sample of non-changing observations supplies negative examples. A
        logistic loss is used around the MOBIL incentive margin, while hard
        safety failures are assigned near-zero probability of changing.
        """
        p, threshold, safe_dec = map(float, z)
        df = self.df.sort_values(["vehicle_id", "time"])
        grouped = {vid:g.sort_values("time") for vid,g in df.groupby("vehicle_id")}
        events=[]; negatives=[]
        for vid,g in grouped.items():
            arr=g.to_numpy()
            for k in range(len(g)-1):
                r=g.iloc[k]; rn=g.iloc[k+1]
                lane=int(r.lane); next_lane=int(rn.lane)
                if next_lane != lane and abs(next_lane-lane)==1:
                    events.append((vid,float(r.time),lane,next_lane,1))
                elif k % 5 == 0:
                    # sample no-change observations to keep the event set manageable
                    negatives.append((vid,float(r.time),lane,None,0))
        if not events:
            return 0.0
        # Balance the classes rather than letting thousands of non-changes dominate.
        rng=np.random.RandomState(0); nneg=min(len(negatives), max(len(events),1)*3)
        if len(negatives)>nneg: negatives=list(rng.choice(negatives,nneg,replace=False))
        cases=(events+negatives)[:max_events]
        losses=[]
        for vid,t,current,target,label in cases:
            ego_g=grouped[vid]
            idx=int(np.argmin(np.abs(ego_g.time.to_numpy(float)-t))); ego=ego_g.iloc[idx]
            frame=df[np.isclose(df.time,t)]
            ip=self.calibrated_params.get(str(ego.vehicle_class), {
                "desired_speed":max(float(ego.speed)+1,15),"minimum_gap":2,"desired_time_headway":1,
                "acceleration":1,"comfortable_deceleration":1.5,"acc_exponent":4})
            targets=[target] if target is not None else [current-1,current+1]
            margins=[]
            for tar in targets:
                if tar < int(frame.lane.min())-1 or tar > int(frame.lane.max())+1: continue
                m=self._mobil_score_at_snapshot(frame,ego,current,tar,ip,p,threshold,safe_dec)
                if m is not None: margins.append(m)
            margin=max(margins) if margins else -10.0
            # Smooth deterministic probability of a lane change.
            prob=1.0/(1.0+math.exp(-np.clip(8.0*margin,-40,40)))
            eps=1e-9; losses.append(-(label*math.log(prob+eps)+(1-label)*math.log(1-prob+eps)))
        return float(np.mean(losses))

    def calibrate_mobil(self, calibrated_params, seed=0):
        self.calibrated_params = calibrated_params
        # Small GA over three MOBIL parameters.
        rng = np.random.RandomState(seed)
        names = list(MOBIL_BOUNDS)
        lo = np.array([MOBIL_BOUNDS[n][0] for n in names])
        hi = np.array([MOBIL_BOUNDS[n][1] for n in names])
        pop_n, gens = min(60, self.population), max(10, self.generations // 2)
        pop = rng.uniform(lo, hi, size=(pop_n, 3))
        scores = np.array([self.mobil_objective(x) for x in pop])
        for _ in range(gens):
            order = np.argsort(scores); pop, scores = pop[order], scores[order]
            elite = max(2, pop_n // 5); new = [pop[i].copy() for i in range(elite)]
            while len(new) < pop_n:
                a, b = pop[rng.randint(0, elite)], pop[rng.randint(0, elite)]
                child = np.where(rng.rand(3) < .5, a, b)
                child += rng.normal(0, .08, 3) * (hi - lo)
                new.append(np.clip(child, lo, hi))
            pop = np.asarray(new); scores = np.array([self.mobil_objective(x) for x in pop])
        best = pop[int(np.argmin(scores))]
        return {n: float(v) for n, v in zip(names, best)}, float(np.min(scores))


if QT_AVAILABLE:
    class CalibrationWorker(QThread):
        progress = Signal(int, str)
        finished_results = Signal(object)
        failed = Signal(str)

        def __init__(self, csv_path, dt, objective, population, generations, seed, max_traj, calibrate_mobil):
            super().__init__()
            self.csv_path = csv_path
            self.dt = dt
            self.objective = objective
            self.population = population
            self.generations = generations
            self.seed = seed
            self.max_traj = max_traj
            self.calibrate_mobil_flag = calibrate_mobil

        def run(self):
            try:
                df = load_trajectory_csv(self.csv_path)
                cal = IDMCalibrator(df, self.dt, self.objective, self.population, self.generations, self.seed, self.max_traj)
                self.progress.emit(0, f"Loaded {len(df):,} trajectory rows; dt={self.dt:g} s")
                results = cal.calibrate(self.progress.emit)
                profile = {r.vehicle_class: r.params for r in results}
                mobil = None
                if self.calibrate_mobil_flag:
                    self.progress.emit(90, "Calibrating MOBIL parameters from observed lane changes...")
                    mobil, _ = cal.calibrate_mobil(profile, self.seed)
                payload = {
                    "idms": profile,
                    "mobil": mobil,
                    "objective": self.objective,
                    "time_step": self.dt,
                    "results": [r.__dict__ for r in results],
                }
                self.progress.emit(100, "Calibration complete")
                self.finished_results.emit(payload)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")


    class CalibrationWindow(QMainWindow):
        """Standalone calibration interface launched by main.py."""
        calibration_completed = Signal(object)

        def __init__(self):
            super().__init__()
            self.setWindowTitle("IDM + MOBIL Model Calibration")
            self.resize(1150, 800)
            self.worker = None
            self.payload = None
            self.build_ui()

        def spin(self, value, lo, hi, step, decimals=2):
            w = QDoubleSpinBox(); w.setRange(lo, hi); w.setValue(value); w.setSingleStep(step); w.setDecimals(decimals); w.setMinimumWidth(130); return w

        def build_ui(self):
            root = QWidget(); self.setCentralWidget(root); layout = QVBoxLayout(root)
            intro = QLabel(
                "Calibrate IDM parameters from vehicle trajectories. "
                "Required CSV fields: time/timestep, position, ID, vehicle class, speed, and Lane. "
                "Optional: vehicle length and width. Position is assumed to be the vehicle front position."
            ); intro.setWordWrap(True); layout.addWidget(intro)

            data = QGroupBox("CALIBRATION DATA")
            f = QFormLayout(data)
            row = QHBoxLayout(); self.file_label = QLabel("No CSV selected"); b = QPushButton("SELECT CSV"); b.clicked.connect(self.select_csv); row.addWidget(self.file_label, 1); row.addWidget(b); f.addRow("Trajectory CSV:", row)
            self.dt = self.spin(0.10, 0.001, 2.0, 0.01, 3); f.addRow("Time step Δt (s):", self.dt)
            self.infer = QCheckBox("Infer timestep from CSV"); self.infer.setChecked(True); f.addRow("", self.infer)
            layout.addWidget(data)

            settings = QGroupBox("CALIBRATION SETTINGS")
            f = QFormLayout(settings)
            self.objective = QComboBox(); self.objective.addItems(["NRMSE(s,v)", "NRMSE(s,v,a)"]); self.objective.setToolTip("NRMSE(s,v) is recommended when acceleration is derived from noisy trajectory data."); f.addRow("Objective:", self.objective)
            self.population = QSpinBox(); self.population.setRange(20, 200); self.population.setValue(60); f.addRow("Population:", self.population)
            self.generations = QSpinBox(); self.generations.setRange(10, 300); self.generations.setValue(40); f.addRow("Generations:", self.generations)
            self.max_traj = QSpinBox(); self.max_traj.setRange(1, 2000); self.max_traj.setValue(30); f.addRow("Max trajectories / class:", self.max_traj)
            self.seed = QSpinBox(); self.seed.setRange(0, 999999); self.seed.setValue(0); f.addRow("Random seed:", self.seed)
            self.cal_mobil = QCheckBox("Also calibrate MOBIL from observed lane changes"); self.cal_mobil.setChecked(True); f.addRow("", self.cal_mobil)
            layout.addWidget(settings)

            self.run_button = QPushButton("▶  RUN MODEL CALIBRATION"); self.run_button.setMinimumHeight(38); self.run_button.clicked.connect(self.run_calibration); layout.addWidget(self.run_button)
            self.progress = QProgressBar(); layout.addWidget(self.progress)
            self.log = QTextEdit(); self.log.setReadOnly(True); layout.addWidget(self.log, 1)

            result_group = QGroupBox("CALIBRATED PARAMETERS")
            rf = QVBoxLayout(result_group)
            self.table = QTableWidget(0, 8); self.table.setHorizontalHeaderLabels(["Class", "v₀", "s₀", "T", "a", "b", "δ", "Objective"]); self.table.horizontalHeader().setStretchLastSection(True); rf.addWidget(self.table)
            buttons = QHBoxLayout(); self.save_button = QPushButton("SAVE PROFILE"); self.apply_button = QPushButton("APPLY TO SIMULATOR"); self.save_button.setEnabled(False); self.apply_button.setEnabled(False); self.save_button.clicked.connect(self.save_profile); self.apply_button.clicked.connect(self.apply_profile); buttons.addWidget(self.save_button); buttons.addWidget(self.apply_button); rf.addLayout(buttons)
            layout.addWidget(result_group, 2)

        def select_csv(self):
            path, _ = QFileDialog.getOpenFileName(self, "Select trajectory CSV", "", "CSV files (*.csv)")
            if path:
                self.csv_path = path; self.file_label.setText(Path(path).name)
                try:
                    df = load_trajectory_csv(path)
                    if self.infer.isChecked(): self.dt.setValue(infer_dt(df))
                    self.log.append(f"Loaded: {path}\nRows: {len(df):,}\nClasses: {', '.join(sorted(df.vehicle_class.unique()))}")
                except Exception as exc:
                    QMessageBox.critical(self, "CSV error", str(exc)); self.file_label.setText("Invalid CSV")

        def run_calibration(self):
            if not hasattr(self, "csv_path"):
                QMessageBox.warning(self, "Select CSV", "Please select a trajectory CSV first."); return
            self.run_button.setEnabled(False); self.save_button.setEnabled(False); self.apply_button.setEnabled(False); self.progress.setValue(0); self.log.append("Starting calibration...")
            self.worker = CalibrationWorker(self.csv_path, self.dt.value(), self.objective.currentText(), self.population.value(), self.generations.value(), self.seed.value(), self.max_traj.value(), self.cal_mobil.isChecked())
            self.worker.progress.connect(lambda v, s: (self.progress.setValue(v), self.log.append(s)))
            self.worker.finished_results.connect(self.show_results)
            self.worker.failed.connect(self.calibration_failed)
            self.worker.start()

        def show_results(self, payload):
            self.payload = payload; self.table.setRowCount(0)
            for item in payload["results"]:
                row = self.table.rowCount(); self.table.insertRow(row)
                vals = [item["vehicle_class"], item["params"]["desired_speed"], item["params"]["minimum_gap"], item["params"]["desired_time_headway"], item["params"]["acceleration"], item["params"]["comfortable_deceleration"], item["params"]["acc_exponent"], item["objective"]]
                for c, val in enumerate(vals): self.table.setItem(row, c, QTableWidgetItem(f"{val:.5f}" if isinstance(val, float) else str(val)))
            if payload.get("mobil"):
                self.log.append("MOBIL: " + json.dumps(payload["mobil"], indent=2))
            self.run_button.setEnabled(True); self.save_button.setEnabled(True); self.apply_button.setEnabled(True)
            self.log.append("Calibration finished successfully. Review the parameter table before applying it.")

        def calibration_failed(self, msg):
            self.run_button.setEnabled(True); QMessageBox.critical(self, "Calibration failed", msg); self.log.append(msg)

        def save_profile(self):
            if not self.payload: return
            path, _ = QFileDialog.getSaveFileName(self, "Save calibration profile", "calibrated_idm_mobil.json", "JSON files (*.json)")
            if path:
                Path(path).write_text(json.dumps(self.payload, indent=2), encoding="utf-8"); QMessageBox.information(self, "Saved", f"Calibration profile saved to:\n{path}")

        def apply_profile(self):
            if self.payload:
                self.calibration_completed.emit(self.payload)
                self.close()




if not QT_AVAILABLE:
    class CalibrationWorker:
        pass
    class CalibrationWindow:
        pass


__all__ = ["IDMCalibrator", "CalibrationWindow", "load_trajectory_csv", "infer_dt"]
