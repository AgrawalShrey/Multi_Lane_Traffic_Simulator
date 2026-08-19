import sys
import traceback

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout, QWidget
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from traffic_model import IDMModel


DEFAULTS = {
    "road_length": 1000.0,
    "road_width": 3.5,
    "num_lanes": 2,
    "volume": 1000.0,
    "car_composition": 0.80,
    "car_ds_mean": 16.66,
    "car_ds_sigma": 2.78,
    "truck_ds_mean": 13.89,
    "truck_ds_sigma": 2.78,
    "car_length": 5.0,
    "car_width": 2.0,
    "truck_length": 8.0,
    "truck_width": 2.5,
    "time_step": 0.1,
    "speed_reduction_factor": 0.5,
    "minimum_gap": 2.0,
    "acc_exponent": 4.0,
    "acceleration": 1.0,
    "comfortable_deceleration": 1.5,
    "desired_time_headway": 1.0,
    "simulation_time": 250.0,
    "seed": 0,
    "enable_mobil": True,
    "mobil_politeness": 0.25,
    "mobil_threshold": 0.20,
    "mobil_safe_deceleration": 2.0,
}


class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(10, 8))
        super().__init__(self.figure)

        self.road_ax = self.figure.add_subplot(311)
        self.st_ax = self.figure.add_subplot(312)
        self.speed_ax = self.figure.add_subplot(313)

        self.figure.subplots_adjust(
            left=0.07, right=0.98, top=0.95, bottom=0.07,
            hspace=0.58
        )

    def draw_simulation(self, sim):
        p = sim.p

        for ax in (self.road_ax, self.st_ax, self.speed_ax):
            ax.clear()

        n_lanes = int(p["num_lanes"])
        lane_width = float(p["road_width"])
        total_width = n_lanes * lane_width

        # ---------------- ROAD ----------------
        ax = self.road_ax
        ax.set_xlim(0, p["road_length"])
        ax.set_ylim(-total_width / 2 - 1, total_width / 2 + 1)

        ax.axhspan(
            -total_width / 2,
            total_width / 2,
            alpha=0.12
        )

        for i in range(n_lanes + 1):
            y = -total_width / 2 + i * lane_width
            if i in (0, n_lanes):
                ax.axhline(y, linewidth=1.8)
            else:
                ax.axhline(y, linestyle="--", linewidth=1.0)

        lane_centers = [
            -total_width / 2 + (i + 0.5) * lane_width
            for i in range(n_lanes)
        ]

        for v in sim.traffic:
            yc = lane_centers[v.lane]

            rect = Rectangle(
                (v.x - v.length, yc - v.width / 2),
                v.length,
                v.width,
                edgecolor="black",
                facecolor="yellow" if v.vehicle_type == "Car" else "gray",
                linewidth=0.8,
            )
            ax.add_patch(rect)

            ax.text(
                v.x - v.length / 2,
                yc,
                str(v.vehicle_id),
                ha="center",
                va="center",
                fontsize=7
            )

        ax.set_yticks(lane_centers)
        ax.set_yticklabels([f"Lane {i+1}" for i in range(n_lanes)])
        ax.set_ylabel("Lane")
        ax.set_title(
            f"IDM Multi-Lane Simulator | {n_lanes} Lanes | "
            f"MOBIL: {'ON' if p['enable_mobil'] else 'OFF'} | "
            f"t = {sim.time:.1f} s | On road = {len(sim.traffic)}"
        )

        # ---------------- TRAJECTORIES ----------------
        df = sim.to_dataframe()

        if not df.empty:
            # Keep the display reasonably light for long simulations.
            # Every vehicle is still retained in the CSV/history.
            plot_df = df
            if len(df) > 50000:
                plot_df = df.iloc[-50000:]

            for _, g in plot_df.groupby("vehicle_id"):
                self.st_ax.plot(
                    g["time"], g["x"], linewidth=0.7
                )

        self.st_ax.set_xlabel("Time (s)")
        self.st_ax.set_ylabel("Front position x (m)")
        self.st_ax.set_title("Vehicle trajectories")

        # ---------------- SPEED ----------------
        if not df.empty:
            plot_df = df
            if len(df) > 50000:
                plot_df = df.iloc[-50000:]

            for _, g in plot_df.groupby("vehicle_id"):
                self.speed_ax.plot(
                    g["time"], g["speed"], linewidth=0.7
                )

        self.speed_ax.set_xlabel("Time (s)")
        self.speed_ax.set_ylabel("Speed (m/s)")
        self.speed_ax.set_title("Vehicle speed histories")
        self.speed_ax.grid(alpha=0.25)

        self.draw_idle()


class SimulatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("IDM Multi-Lane Traffic Simulator")
        self.resize(1800, 1000)
        self.setMinimumSize(1250, 750)

        self.widgets = {}
        self.sim = None

        # GUI timer is deliberately independent from the model time step.
        # Each timer event advances several model steps. This avoids a
        # redraw-heavy GUI appearing frozen.
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)

        self.running = False
        self.steps_per_refresh = 2

        self.build_ui()
        self.restore_defaults()

    # ---------------------------------------------------------------
    # Widget helpers
    # ---------------------------------------------------------------
    def double_box(self, key, value, lo, hi, step=0.1, decimals=2):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(value)
        w.setMinimumWidth(120)
        self.widgets[key] = w
        return w

    def int_box(self, key, value, lo, hi, step=1):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(value)
        w.setMinimumWidth(120)
        self.widgets[key] = w
        return w

    def make_scroll_panel(self, content):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll.setWidget(content)
        return scroll

    # ---------------------------------------------------------------
    # GUI
    # ---------------------------------------------------------------
    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ============================================================
        # LEFT
        # ============================================================
        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)

        group = QGroupBox("ROAD NETWORK")
        form = QFormLayout(group)

        form.addRow(
            "Road length (m):",
            self.double_box("road_length", 1000, 50, 10000, 100, 0)
        )
        form.addRow(
            "Lane width (m):",
            self.double_box("road_width", 3.5, 2.5, 5, 0.25, 2)
        )

        lane_box = self.int_box("num_lanes", 2, 1, 8, 1)
        lane_box.setStyleSheet(
            "font-weight: bold; font-size: 13px;"
        )
        form.addRow("NUMBER OF LANES:", lane_box)

        self.lane_info = QLabel()
        form.addRow("", self.lane_info)

        lane_box.valueChanged.connect(self.update_lane_info)
        self.widgets["road_width"].valueChanged.connect(
            lambda _: self.update_lane_info(lane_box.value())
        )

        left_layout.addWidget(group)

        group = QGroupBox("TRAFFIC DEMAND")
        form = QFormLayout(group)

        form.addRow(
            "Volume (veh/h):",
            self.double_box("volume", 1000, 1, 20000, 100, 0)
        )
        form.addRow(
            "Car composition:",
            self.double_box("car_composition", 0.8, 0, 1, 0.05, 2)
        )
        form.addRow(
            "Simulation time (s):",
            self.double_box("simulation_time", 250, 1, 10000, 10, 0)
        )
        form.addRow(
            "Time step (s):",
            self.double_box("time_step", 0.1, 0.01, 1, 0.01, 2)
        )

        left_layout.addWidget(group)

        group = QGroupBox("SIMULATION CONTROL")
        box = QVBoxLayout(group)

        self.start_button = QPushButton("▶  START")
        self.pause_button = QPushButton("Ⅱ  PAUSE")
        self.reset_button = QPushButton("↻  RESET")
        self.full_button = QPushButton("RUN FULL SIMULATION")
        self.export_button = QPushButton("EXPORT CSV")
        self.defaults_button = QPushButton("RESTORE DEFAULTS")

        for button in [
            self.start_button, self.pause_button, self.reset_button,
            self.full_button, self.export_button, self.defaults_button
        ]:
            button.setMinimumHeight(34)
            box.addWidget(button)

        self.start_button.clicked.connect(self.start)
        self.pause_button.clicked.connect(self.pause)
        self.reset_button.clicked.connect(self.reset)
        self.full_button.clicked.connect(self.run_full)
        self.export_button.clicked.connect(self.export_csv)
        self.defaults_button.clicked.connect(self.restore_defaults)

        left_layout.addWidget(group)

        group = QGroupBox("SIMULATION STATUS")
        box = QVBoxLayout(group)
        self.status = QLabel()
        self.status.setWordWrap(True)
        box.addWidget(self.status)
        left_layout.addWidget(group)

        left_layout.addStretch()

        left_scroll = self.make_scroll_panel(left_content)
        left_scroll.setMinimumWidth(310)
        left_scroll.setMaximumWidth(400)

        # ============================================================
        # CENTER
        # ============================================================
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = PlotCanvas()
        center_layout.addWidget(self.canvas)

        # ============================================================
        # RIGHT
        # ============================================================
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)

        group = QGroupBox("VEHICLE PROPERTIES")
        form = QFormLayout(group)

        fields = [
            ("car_ds_mean", "Car desired speed μ:", 16.66, 1, 50, .5, 2),
            ("car_ds_sigma", "Car desired speed σ:", 2.78, .01, 20, .1, 2),
            ("truck_ds_mean", "Truck desired speed μ:", 13.89, 1, 50, .5, 2),
            ("truck_ds_sigma", "Truck desired speed σ:", 2.78, .01, 20, .1, 2),
            ("car_length", "Car length (m):", 5, 1, 20, .5, 2),
            ("car_width", "Car width (m):", 2, .5, 5, .1, 2),
            ("truck_length", "Truck length (m):", 8, 1, 30, .5, 2),
            ("truck_width", "Truck width (m):", 2.5, .5, 5, .1, 2),
        ]

        for key, label, val, lo, hi, step, dec in fields:
            form.addRow(
                label,
                self.double_box(key, val, lo, hi, step, dec)
            )

        right_layout.addWidget(group)

        group = QGroupBox("IDM CAR-FOLLOWING")
        form = QFormLayout(group)

        fields = [
            ("minimum_gap", "Minimum gap s₀ (m):", 2, .01, 30, .5, 2),
            ("acceleration", "Acceleration a (m/s²):", 1, .01, 10, .1, 2),
            ("comfortable_deceleration", "Comfort deceleration b:", 1.5, .01, 10, .1, 2),
            ("desired_time_headway", "Time headway T (s):", 1, .01, 10, .1, 2),
            ("acc_exponent", "Acceleration exponent δ:", 4, 1, 10, 1, 0),
            ("speed_reduction_factor", "Placement speed factor:", .5, 0, 1, .05, 2),
        ]

        for key, label, val, lo, hi, step, dec in fields:
            form.addRow(
                label,
                self.double_box(key, val, lo, hi, step, dec)
            )

        right_layout.addWidget(group)

        group = QGroupBox("MOBIL LANE CHANGING")
        form = QFormLayout(group)

        mobil = QCheckBox("ENABLE MOBIL LANE CHANGING")
        mobil.setChecked(True)
        mobil.setStyleSheet("font-weight: bold;")
        self.widgets["enable_mobil"] = mobil
        form.addRow(mobil)

        form.addRow(
            "Politeness factor p:",
            self.double_box("mobil_politeness", .25, 0, 1, .05, 2)
        )
        form.addRow(
            "Incentive threshold Δa:",
            self.double_box("mobil_threshold", .20, 0, 5, .05, 2)
        )
        form.addRow(
            "Safe deceleration:",
            self.double_box("mobil_safe_deceleration", 2, .1, 10, .1, 2)
        )

        right_layout.addWidget(group)

        group = QGroupBox("RANDOMNESS")
        form = QFormLayout(group)
        form.addRow(
            "Random seed:",
            self.int_box("seed", 0, 0, 999999, 1)
        )
        right_layout.addWidget(group)

        right_layout.addStretch()

        right_scroll = self.make_scroll_panel(right_content)
        right_scroll.setMinimumWidth(330)
        right_scroll.setMaximumWidth(420)

        main_layout.addWidget(left_scroll, 0)
        main_layout.addWidget(center, 1)
        main_layout.addWidget(right_scroll, 0)

    # ---------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------
    def update_lane_info(self, n):
        width = n * self.widgets["road_width"].value()
        self.lane_info.setText(
            f"Current: {n} lanes\n"
            f"Total road width: {width:.2f} m"
        )

    def parameters(self):
        result = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            else:
                result[key] = widget.value()
        return result

    # ---------------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------------
    def restore_defaults(self):
        self.pause()

        for key, value in DEFAULTS.items():
            if key not in self.widgets:
                continue

            widget = self.widgets[key]

            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            else:
                widget.setValue(value)

        self.update_lane_info(
            self.widgets["num_lanes"].value()
        )

        self.reset()

    def reset(self):
        self.pause()

        try:
            self.sim = IDMModel(self.parameters())
            self.canvas.draw_simulation(self.sim)
            self.update_status("Ready")
        except Exception as exc:
            self.sim = None
            self.update_status("ERROR")
            QMessageBox.critical(
                self,
                "Simulation initialization error",
                f"{type(exc).__name__}: {exc}\n\n"
                f"Check that main.py, gui.py, traffic_model.py and "
                f"lane_changing.py are from the same version."
            )

    def start(self):
        if self.sim is None:
            self.reset()

        if self.sim is None:
            return

        if self.sim.time >= self.sim.p["simulation_time"]:
            self.reset()

        self.running = True
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.update_status("RUNNING")

        # GUI refresh is 50 ms. The model advances several time steps
        # per refresh, depending on the selected model time step.
        self.timer.start(50)

    def pause(self):
        self.running = False
        self.timer.stop()

        if hasattr(self, "start_button"):
            self.start_button.setEnabled(True)

        if self.sim is not None:
            self.update_status("PAUSED")

    def advance(self):
        if not self.running or self.sim is None:
            return

        try:
            # Advance approximately 0.1 s of model time per GUI refresh.
            dt = float(self.sim.p["time_step"])
            n_steps = max(1, int(round(0.10 / dt)))

            for _ in range(n_steps):
                if self.sim.time > self.sim.p["simulation_time"]:
                    break
                self.sim.step()

            self.canvas.draw_simulation(self.sim)
            self.update_status("RUNNING")

            if self.sim.time > self.sim.p["simulation_time"]:
                self.pause()
                self.update_status("FINISHED")

        except Exception as exc:
            self.pause()

            error_text = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"The simulation was stopped to prevent the GUI from "
                f"silently failing."
            )

            self.status.setText(
                f"<b>ERROR</b><br>{type(exc).__name__}: {exc}"
            )

            QMessageBox.critical(
                self,
                "Simulation error",
                error_text
            )

    def run_full(self):
        self.pause()

        try:
            self.sim = IDMModel(self.parameters())
            self.update_status("RUNNING FULL SIMULATION")

            last_draw = -999.0

            while self.sim.time <= self.sim.p["simulation_time"]:
                self.sim.step()

                if self.sim.time - last_draw >= 2.0:
                    QApplication.processEvents()
                    self.canvas.draw_simulation(self.sim)
                    self.update_status("RUNNING FULL SIMULATION")
                    last_draw = self.sim.time

            self.canvas.draw_simulation(self.sim)
            self.update_status("FINISHED")

        except Exception as exc:
            self.update_status("ERROR")
            QMessageBox.critical(
                self,
                "Simulation error",
                f"{type(exc).__name__}: {exc}"
            )

    def update_status(self, state=None):
        if self.sim is None:
            self.status.setText(
                "<b>Status:</b> No simulation"
            )
            return

        s = self.sim.summary()

        lane_counts = [
            0 for _ in range(int(self.sim.p["num_lanes"]))
        ]

        for v in self.sim.traffic:
            lane_counts[v.lane] += 1

        occupancy = " | ".join(
            f"L{i + 1}: {n}"
            for i, n in enumerate(lane_counts)
        )

        status = state or (
            "RUNNING" if self.running else "PAUSED"
        )

        self.status.setText(
            f"<b>Status: {status}</b><br>"
            f"Lanes: {self.sim.p['num_lanes']}<br>"
            f"Lane occupancy: {occupancy}<br>"
            f"MOBIL: {'ON' if self.sim.p['enable_mobil'] else 'OFF'}<br>"
            f"Time: {s['simulation_time']:.1f} s / "
            f"{self.sim.p['simulation_time']:.1f} s<br>"
            f"Generated: {s['generated']}<br>"
            f"On road: {s['on_road']}<br>"
            f"Virtual queue: {s['queue']}<br>"
            f"Departed: {s['departed']}"
        )

    def export_csv(self):
        if not self.sim or not self.sim.history:
            QMessageBox.information(
                self,
                "No data",
                "Run the simulation before exporting."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save trajectory data",
            "idm_mobil_trajectories.csv",
            "CSV files (*.csv)"
        )

        if filename:
            self.sim.to_dataframe().to_csv(
                filename,
                index=False
            )

            QMessageBox.information(
                self,
                "Export complete",
                f"Saved:\n{filename}"
            )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = SimulatorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
