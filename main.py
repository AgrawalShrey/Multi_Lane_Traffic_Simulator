"""IDM Multi-Lane Traffic Simulator launcher.

At startup the user chooses between:
1. Default / calibrated simulation
2. Model calibration from trajectory CSV
"""
import sys
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from gui import SimulatorWindow
from calibration import CalibrationWindow

class ModeDialog(QDialog):
    def __init__(self):
        super().__init__(); self.setWindowTitle('IDM Multi-Lane Traffic Simulator'); self.setFixedSize(620,280); self.choice=None
        layout=QVBoxLayout(self); title=QLabel('<h2>IDM Multi-Lane Traffic Simulator</h2>'); layout.addWidget(title)
        msg=QLabel('Choose how you want to start the application:'); msg.setWordWrap(True); layout.addWidget(msg)
        row=QHBoxLayout(); self.default=QPushButton('DEFAULT / SIMULATION\n\nRun the simulator with configurable vehicle classes, IDM and MOBIL.'); self.cal=QPushButton('MODEL CALIBRATION\n\nEstimate IDM and MOBIL parameters from trajectory CSV data.'); self.default.setMinimumHeight(110); self.cal.setMinimumHeight(110); row.addWidget(self.default); row.addWidget(self.cal); layout.addLayout(row)
        self.default.clicked.connect(lambda:self.accept_choice('default')); self.cal.clicked.connect(lambda:self.accept_choice('calibration'))
    def accept_choice(self,c):self.choice=c; self.accept()

def main():
    app=QApplication(sys.argv); app.setStyle('Fusion')
    dlg=ModeDialog()
    if dlg.exec()!=QDialog.DialogCode.Accepted:return
    if dlg.choice=='calibration':
        cw=CalibrationWindow(); holder={'sim':None}
        def open_sim(payload):
            holder['sim']=SimulatorWindow(payload); holder['sim'].show()
        cw.calibration_completed.connect(open_sim); cw.show(); app.exec()
    else:
        w=SimulatorWindow(); w.show(); app.exec()

if __name__=='__main__':main()
