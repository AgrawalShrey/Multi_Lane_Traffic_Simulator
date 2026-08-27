"""GUI for IDM multi-lane simulator.

Supports normal simulation and calibrated class-specific IDM profiles.
"""
import sys
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from traffic_model import IDMModel, DEFAULT_VEHICLE_CLASSES

DEFAULTS = {
    "road_length":1000.0,"road_width":3.5,"num_lanes":2,"volume":1000.0,
    "simulation_time":250.0,"time_step":0.1,"speed_reduction_factor":0.5,"seed":0,
    "enable_mobil":True,"mobil_politeness":0.25,"mobil_threshold":0.20,"mobil_safe_deceleration":2.0,
}

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.figure=Figure(figsize=(10,8)); super().__init__(self.figure)
        self.road_ax=self.figure.add_subplot(311); self.st_ax=self.figure.add_subplot(312); self.speed_ax=self.figure.add_subplot(313)
        self.figure.subplots_adjust(left=.07,right=.98,top=.95,bottom=.07,hspace=.58)
    def draw_simulation(self,sim):
        p=sim.p
        for ax in (self.road_ax,self.st_ax,self.speed_ax): ax.clear()
        n=int(p['num_lanes']); w=float(p['road_width']); total=n*w
        ax=self.road_ax; ax.set_xlim(0,p['road_length']); ax.set_ylim(-total/2-1,total/2+1); ax.axhspan(-total/2,total/2,alpha=.12)
        for i in range(n+1):
            y=-total/2+i*w; ax.axhline(y,linewidth=1.8 if i in (0,n) else 1.0,linestyle='-' if i in (0,n) else '--')
        centers=[-total/2+(i+.5)*w for i in range(n)]
        classes=list(sim.class_map); class_index={c:i for i,c in enumerate(classes)}
        for v in sim.traffic:
            yc=centers[v.lane]; idx=class_index.get(v.vehicle_type,0)
            face='yellow' if idx==0 else ('gray' if idx==1 else 'lightgray')
            rect=Rectangle((v.x-v.length,yc-v.width/2),v.length,v.width,edgecolor='black',facecolor=face,linewidth=.8)
            ax.add_patch(rect); ax.text(v.x-v.length/2,yc,str(v.vehicle_id),ha='center',va='center',fontsize=7)
        ax.set_yticks(centers); ax.set_yticklabels([f'Lane {i+1}' for i in range(n)]); ax.set_ylabel('Lane')
        ax.set_title(f"IDM Multi-Lane Simulator | {n} Lanes | MOBIL: {'ON' if p['enable_mobil'] else 'OFF'} | t = {sim.time:.1f} s | On road = {len(sim.traffic)}")
        df=sim.to_dataframe()
        plot_df=df.iloc[-50000:] if len(df)>50000 else df
        if not plot_df.empty:
            for _,g in plot_df.groupby('vehicle_id'): self.st_ax.plot(g.time,g.x,linewidth=.7)
            for _,g in plot_df.groupby('vehicle_id'): self.speed_ax.plot(g.time,g.speed,linewidth=.7)
        self.st_ax.set_xlabel('Time (s)'); self.st_ax.set_ylabel('Front position x (m)'); self.st_ax.set_title('Vehicle trajectories')
        self.speed_ax.set_xlabel('Time (s)'); self.speed_ax.set_ylabel('Speed (m/s)'); self.speed_ax.set_title('Vehicle speed histories'); self.speed_ax.grid(alpha=.25)
        self.draw_idle()

class SimulatorWindow(QMainWindow):
    def __init__(self, calibrated_profile=None):
        super().__init__(); self.setWindowTitle('IDM Multi-Lane Traffic Simulator'); self.resize(1800,1000); self.setMinimumSize(1250,750)
        self.widgets={}; self.sim=None; self.running=False; self.current_class_index=0
        self.timer=QTimer(self); self.timer.timeout.connect(self.advance)
        self.calibrated_profile=calibrated_profile
        self.build_ui(); self.restore_defaults()
        if calibrated_profile: self.apply_calibration_profile(calibrated_profile)
    def double_box(self,key,value,lo,hi,step=.1,decimals=2):
        w=QDoubleSpinBox(); w.setRange(lo,hi); w.setSingleStep(step); w.setDecimals(decimals); w.setValue(value); w.setMinimumWidth(120); self.widgets[key]=w; return w
    def int_box(self,key,value,lo,hi,step=1):
        w=QSpinBox(); w.setRange(lo,hi); w.setSingleStep(step); w.setValue(value); w.setMinimumWidth(120); self.widgets[key]=w; return w
    def make_scroll_panel(self,content):
        s=QScrollArea(); s.setWidgetResizable(True); s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff); s.setWidget(content); return s
    def build_ui(self):
        root=QWidget(); self.setCentralWidget(root); main=QHBoxLayout(root); main.setContentsMargins(8,8,8,8); main.setSpacing(8)
        # LEFT
        lc=QWidget(); ll=QVBoxLayout(lc)
        g=QGroupBox('ROAD NETWORK'); f=QFormLayout(g); f.addRow('Road length (m):',self.double_box('road_length',1000,50,10000,100,0)); f.addRow('Lane width (m):',self.double_box('road_width',3.5,2.5,5,.25,2)); lb=self.int_box('num_lanes',2,1,8); lb.setStyleSheet('font-weight:bold;font-size:13px;'); f.addRow('NUMBER OF LANES:',lb); self.lane_info=QLabel(); f.addRow('',self.lane_info); lb.valueChanged.connect(self.update_lane_info); self.widgets['road_width'].valueChanged.connect(lambda _:self.update_lane_info(lb.value())); ll.addWidget(g)
        g=QGroupBox('TRAFFIC DEMAND'); f=QFormLayout(g); f.addRow('Volume (veh/h):',self.double_box('volume',1000,1,20000,100,0)); f.addRow('Simulation time (s):',self.double_box('simulation_time',250,1,10000,10,0)); f.addRow('Time step (s):',self.double_box('time_step',.1,.01,1,.01,3)); ll.addWidget(g)
        g=QGroupBox('VEHICLE CLASS SETTINGS'); v=QVBoxLayout(g); row=QHBoxLayout(); self.class_combo=QComboBox(); self.class_combo.currentIndexChanged.connect(self.class_changed); self.include_class=QCheckBox('Include'); self.add_class_btn=QPushButton('+ ADD'); self.remove_class_btn=QPushButton('REMOVE'); row.addWidget(self.class_combo,1); row.addWidget(self.include_class); row.addWidget(self.add_class_btn); row.addWidget(self.remove_class_btn); v.addLayout(row)
        self.class_name=QLineEdit(); self.class_name.setPlaceholderText('Class name'); v.addWidget(self.class_name)
        self.class_fields={}; cf=QFormLayout();
        for key,label,val,lo,hi,step,dec in [
            ('composition','Composition',.0,0,1,.01,3),('desired_speed','Desired speed mean (m/s)',15,1,50,.5,2),('speed_sigma','Desired speed sigma (m/s)',2,.0,20,.1,2),('length','Length (m)',5,0.5,30,.5,2),('width','Width (m)',2,.3,5,.1,2),
        ]:
            self.class_fields[key]=self.double_box('class_'+key,val,lo,hi,step,dec); cf.addRow(label+':',self.class_fields[key])
        v.addLayout(cf); self.add_class_btn.clicked.connect(self.add_class); self.remove_class_btn.clicked.connect(self.remove_class); self.include_class.stateChanged.connect(self.class_edited); self.class_name.editingFinished.connect(self.class_edited)
        ll.addWidget(g)
        g=QGroupBox('SIMULATION CONTROL'); b=QVBoxLayout(g); self.start_button=QPushButton('▶  START'); self.pause_button=QPushButton('Ⅱ  PAUSE'); self.reset_button=QPushButton('↻  RESET'); self.full_button=QPushButton('RUN FULL SIMULATION'); self.export_button=QPushButton('EXPORT CSV'); self.defaults_button=QPushButton('RESTORE DEFAULTS'); self.load_profile_btn=QPushButton('LOAD CALIBRATED PROFILE');
        for x in [self.start_button,self.pause_button,self.reset_button,self.full_button,self.export_button,self.load_profile_btn,self.defaults_button]: x.setMinimumHeight(34); b.addWidget(x)
        self.start_button.clicked.connect(self.start); self.pause_button.clicked.connect(self.pause); self.reset_button.clicked.connect(self.reset); self.full_button.clicked.connect(self.run_full); self.export_button.clicked.connect(self.export_csv); self.defaults_button.clicked.connect(self.restore_defaults); self.load_profile_btn.clicked.connect(self.load_profile); ll.addWidget(g)
        g=QGroupBox('SIMULATION STATUS'); b=QVBoxLayout(g); self.status=QLabel(); self.status.setWordWrap(True); b.addWidget(self.status); ll.addWidget(g); ll.addStretch(); ls=self.make_scroll_panel(lc); ls.setMinimumWidth(310); ls.setMaximumWidth(420)
        # CENTER
        c=QWidget(); cl=QVBoxLayout(c); cl.setContentsMargins(0,0,0,0); self.canvas=PlotCanvas(); cl.addWidget(self.canvas)
        # RIGHT
        rc=QWidget(); rl=QVBoxLayout(rc)
        g=QGroupBox('IDM CAR-FOLLOWING — SELECTED CLASS'); f=QFormLayout(g)
        for key,label,val,lo,hi,step,dec in [('minimum_gap','Minimum gap s₀ (m)',2,.01,30,.1,2),('acceleration','Acceleration a (m/s²)',1,.01,10,.1,2),('comfortable_deceleration','Comfort deceleration b (m/s²)',1.5,.01,10,.1,2),('desired_time_headway','Time headway T (s)',1,.01,10,.1,2),('acc_exponent','Acceleration exponent δ',4,1,10,1,0)]:
            self.class_fields[key]=self.double_box('class_'+key,val,lo,hi,step,dec); f.addRow(label+':',self.class_fields[key])
        self.apply_class_btn=QPushButton('APPLY CLASS PARAMETERS'); self.apply_class_btn.clicked.connect(self.class_edited); f.addRow('',self.apply_class_btn); rl.addWidget(g)
        g=QGroupBox('MOBIL LANE CHANGING'); f=QFormLayout(g); m=QCheckBox('ENABLE MOBIL LANE CHANGING'); m.setChecked(True); self.widgets['enable_mobil']=m; f.addRow(m); f.addRow('Politeness factor p:',self.double_box('mobil_politeness',.25,0,1,.05,2)); f.addRow('Incentive threshold Δa:',self.double_box('mobil_threshold',.2,0,5,.05,2)); f.addRow('Safe deceleration (m/s²):',self.double_box('mobil_safe_deceleration',2,.1,10,.1,2)); rl.addWidget(g)
        g=QGroupBox('RANDOMNESS'); f=QFormLayout(g); f.addRow('Random seed:',self.int_box('seed',0,0,999999)); rl.addWidget(g); rl.addStretch(); rs=self.make_scroll_panel(rc); rs.setMinimumWidth(350); rs.setMaximumWidth(450)
        main.addWidget(ls,0); main.addWidget(c,1); main.addWidget(rs,0)
        self.set_default_class_list(DEFAULT_VEHICLE_CLASSES)
    def set_default_class_list(self,classes):
        self.class_profiles=[dict(x) for x in classes]; self.class_combo.blockSignals(True); self.class_combo.clear(); self.class_combo.addItems([c['name'] for c in self.class_profiles]); self.class_combo.blockSignals(False); self.current_class_index=0; self.load_class_to_widgets()
    def update_lane_info(self,n): self.lane_info.setText(f'Current: {n} lanes\nTotal road width: {n*self.widgets["road_width"].value():.2f} m')
    def load_class_to_widgets(self):
        if not self.class_profiles: return
        c=self.class_profiles[self.current_class_index]; self.class_combo.blockSignals(True); self.class_combo.setCurrentIndex(self.current_class_index); self.class_combo.blockSignals(False); self.class_name.setText(c['name']); self.include_class.setChecked(bool(c.get('enabled',True)))
        for k,w in self.class_fields.items(): w.blockSignals(True); w.setValue(float(c.get(k,0))); w.blockSignals(False)
    def save_widgets_to_class(self):
        if not self.class_profiles:return
        c=self.class_profiles[self.current_class_index]; c['name']=self.class_name.text().strip() or f'Class {self.current_class_index+1}'; c['enabled']=self.include_class.isChecked()
        for k,w in self.class_fields.items(): c[k]=w.value()
        self.class_combo.blockSignals(True); self.class_combo.setItemText(self.current_class_index,c['name']); self.class_combo.blockSignals(False)
    def class_changed(self,i):
        if hasattr(self,'class_profiles') and self.class_profiles:
            self.save_widgets_to_class(); self.current_class_index=i; self.load_class_to_widgets()
    def class_edited(self,*args): self.save_widgets_to_class()
    def add_class(self):
        self.save_widgets_to_class(); n=len(self.class_profiles)+1; c={'name':f'Custom {n}','enabled':False,'composition':0.0,'desired_speed':15,'speed_sigma':2,'length':5,'width':2,'minimum_gap':2,'acceleration':1,'comfortable_deceleration':1.5,'desired_time_headway':1,'acc_exponent':4}; self.class_profiles.append(c); self.class_combo.addItem(c['name']); self.current_class_index=len(self.class_profiles)-1; self.load_class_to_widgets()
    def remove_class(self):
        if len(self.class_profiles)<=1:return
        self.class_profiles.pop(self.current_class_index); self.class_combo.removeItem(self.current_class_index); self.current_class_index=max(0,self.current_class_index-1); self.load_class_to_widgets()
    def parameters(self):
        self.save_widgets_to_class(); out={k:(w.isChecked() if isinstance(w,QCheckBox) else w.value()) for k,w in self.widgets.items()}; out['vehicle_classes']=[dict(c) for c in self.class_profiles]; return out
    def restore_defaults(self):
        self.pause(); self.set_default_class_list(DEFAULT_VEHICLE_CLASSES)
        for k,val in DEFAULTS.items():
            if k in self.widgets:
                w=self.widgets[k]; w.setChecked(bool(val)) if isinstance(w,QCheckBox) else w.setValue(val)
        self.update_lane_info(self.widgets['num_lanes'].value()); self.reset()
    def apply_calibration_profile(self,payload):
        classes={c['name']:c for c in self.class_profiles}
        for name,par in payload.get('idms',{}).items():
            if name not in classes:
                self.class_profiles.append({'name':name,'enabled':True,'composition':0,'desired_speed':par['desired_speed'],'speed_sigma':2,'length':5,'width':2,**par})
            else: classes[name].update(par); classes[name]['enabled']=True
        self.class_combo.blockSignals(True); self.class_combo.clear(); self.class_combo.addItems([c['name'] for c in self.class_profiles]); self.class_combo.blockSignals(False); self.current_class_index=0; self.load_class_to_widgets()
        if payload.get('mobil'):
            for k,v in payload['mobil'].items():
                if k in self.widgets:self.widgets[k].setValue(v)
        self.reset(); self.update_status('CALIBRATED PROFILE APPLIED')
    def reset(self):
        self.pause()
        try:self.sim=IDMModel(self.parameters()); self.canvas.draw_simulation(self.sim); self.update_status('Ready')
        except Exception as e:self.sim=None; self.update_status('ERROR'); QMessageBox.critical(self,'Initialization error',f'{type(e).__name__}: {e}')
    def start(self):
        if self.sim is None:self.reset()
        if self.sim is None:return
        if self.sim.time>=self.sim.p['simulation_time']:self.reset()
        self.running=True; self.start_button.setEnabled(False); self.timer.start(50); self.update_status('RUNNING')
    def pause(self):
        self.running=False; self.timer.stop()
        if hasattr(self,'start_button'):self.start_button.setEnabled(True)
        if self.sim is not None:self.update_status('PAUSED')
    def advance(self):
        if not self.running or self.sim is None:return
        try:
            dt=float(self.sim.p['time_step']); n=max(1,int(round(.10/dt)))
            for _ in range(n):
                if self.sim.time>self.sim.p['simulation_time']:break
                self.sim.step()
            self.canvas.draw_simulation(self.sim); self.update_status('RUNNING')
            if self.sim.time>self.sim.p['simulation_time']:self.pause(); self.update_status('FINISHED')
        except Exception as e:
            self.pause(); QMessageBox.critical(self,'Simulation error',f'{type(e).__name__}: {e}')
    def run_full(self):
        self.pause()
        try:
            self.sim=IDMModel(self.parameters()); last=-999
            while self.sim.time<=self.sim.p['simulation_time']:
                self.sim.step()
                if self.sim.time-last>=2: QApplication.processEvents(); self.canvas.draw_simulation(self.sim); self.update_status('RUNNING FULL SIMULATION'); last=self.sim.time
            self.canvas.draw_simulation(self.sim); self.update_status('FINISHED')
        except Exception as e:QMessageBox.critical(self,'Simulation error',f'{type(e).__name__}: {e}')
    def update_status(self,state=None):
        if self.sim is None:self.status.setText('<b>Status:</b> No simulation'); return
        s=self.sim.summary(); occ=' | '.join(f'L{i}: {n}' for i,n in s['lane_counts'].items()); st=state or ('RUNNING' if self.running else 'PAUSED')
        self.status.setText(f'<b>Status: {st}</b><br>Lanes: {self.sim.p["num_lanes"]}<br>Lane occupancy: {occ}<br>MOBIL: {"ON" if self.sim.p["enable_mobil"] else "OFF"}<br>Time: {s["simulation_time"]:.1f} / {self.sim.p["simulation_time"]:.1f} s<br>Generated: {s["generated"]}<br>On road: {s["on_road"]}<br>Virtual queue: {s["queue"]}<br>Departed: {s["departed"]}')
    def load_profile(self):
        fn,_=QFileDialog.getOpenFileName(self,'Load calibration profile','','JSON files (*.json)')
        if not fn:return
        try:
            import json
            payload=json.loads(open(fn,'r',encoding='utf-8').read())
            self.apply_calibration_profile(payload)
            QMessageBox.information(self,'Profile loaded',f'Calibration profile loaded from:\n{fn}')
        except Exception as e:
            QMessageBox.critical(self,'Profile error',f'{type(e).__name__}: {e}')

    def export_csv(self):
        if not self.sim or not self.sim.history:QMessageBox.information(self,'No data','Run the simulation before exporting.'); return
        fn,_=QFileDialog.getSaveFileName(self,'Save trajectory data','idm_mobil_trajectories.csv','CSV files (*.csv)')
        if fn:self.sim.to_dataframe().to_csv(fn,index=False); QMessageBox.information(self,'Export complete',f'Saved:\n{fn}')

def main(calibrated_profile=None):
    app=QApplication.instance() or QApplication(sys.argv); app.setStyle('Fusion'); w=SimulatorWindow(calibrated_profile); w.show(); return w
