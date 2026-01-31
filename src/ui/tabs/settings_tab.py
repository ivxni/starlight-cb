"""
Settings Tab - Minimal Design
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QScrollArea, QFrame, QTabWidget,
    QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QFileDialog
)
from PyQt6.QtCore import Qt

from ..widgets.section_widget import SectionWidget
from ..widgets.slider_widget import SliderWidget
from ..widgets.toggle_switch import LabeledToggle, ToggleWithStatus
from ...core.config import Config


class CaptureSettings(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._load_config()
    
    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 10, 0)
        
        # Capture
        cap = SectionWidget("Capture")
        
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        mode_row.addStretch()
        self.mode = QComboBox()
        self.mode.addItems(["DXGI (Recommended)", "MSS"])
        self.mode.setFixedWidth(140)
        mode_row.addWidget(self.mode)
        cap.addLayout(mode_row)
        
        self.crop = LabeledToggle("Center Crop")
        self.crop.setChecked(True)
        cap.addWidget(self.crop)
        
        layout.addWidget(cap)
        
        # Resolution
        res = SectionWidget("Resolution")
        
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Width"))
        self.width = QSpinBox()
        self.width.setRange(128, 1920)
        self.width.setValue(640)
        self.width.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.width.setFixedWidth(70)
        self.width.setKeyboardTracking(False)  # Only emit when editing finished
        self.width.valueChanged.connect(self._on_width_changed)
        res_row.addWidget(self.width)
        
        res_row.addSpacing(10)
        res_row.addWidget(QLabel("Height"))
        self.height = QSpinBox()
        self.height.setRange(128, 1080)
        self.height.setValue(640)
        self.height.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.height.setFixedWidth(70)
        self.height.setKeyboardTracking(False)  # Only emit when editing finished
        self.height.valueChanged.connect(self._on_height_changed)
        res_row.addWidget(self.height)
        res_row.addStretch()
        res.addLayout(res_row)
        
        self.max_fps = SliderWidget("Max FPS", 30, 360, 240)
        self.max_fps.valueChanged.connect(lambda v: setattr(self.config.capture, 'max_fps', int(v)))
        res.addWidget(self.max_fps)
        
        layout.addWidget(res)
        
        # Debug
        dbg = SectionWidget("Debug Display")
        
        self.dbg_scale = SliderWidget("Scale", 0.25, 2.0, 1.0, decimals=2)
        self.dbg_scale.valueChanged.connect(lambda v: setattr(self.config.capture, 'debug_scale', v))
        dbg.addWidget(self.dbg_scale)
        
        self.dbg_fps = SliderWidget("Max FPS", 15, 120, 60)
        self.dbg_fps.valueChanged.connect(lambda v: setattr(self.config.capture, 'debug_max_fps', int(v)))
        dbg.addWidget(self.dbg_fps)
        
        layout.addWidget(dbg)
        layout.addStretch()
        
        scroll.setWidget(content)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)
    
    def _load_config(self):
        # Block signals to prevent feedback loop
        self.width.blockSignals(True)
        self.height.blockSignals(True)
        self.max_fps.blockSignals(True)
        self.dbg_scale.blockSignals(True)
        self.dbg_fps.blockSignals(True)
        
        c = self.config.capture
        self.width.setValue(c.capture_width)
        self.height.setValue(c.capture_height)
        self.max_fps.setValue(c.max_fps)
        self.dbg_scale.setValue(c.debug_scale)
        self.dbg_fps.setValue(c.debug_max_fps)
        
        # Re-enable signals
        self.width.blockSignals(False)
        self.height.blockSignals(False)
        self.max_fps.blockSignals(False)
        self.dbg_scale.blockSignals(False)
        self.dbg_fps.blockSignals(False)
    
    def _on_width_changed(self, value: int):
        self.config.capture.capture_width = value
    
    def _on_height_changed(self, value: int):
        self.config.capture.capture_height = value


class DetectionSettings(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._load_config()
    
    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 10, 0)
        
        # Model
        model = SectionWidget("Model")
        
        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend"))
        backend_row.addStretch()
        self.backend = QComboBox()
        self.backend.addItems(["TensorRT", "ONNX Runtime"])
        self.backend.setFixedWidth(120)
        backend_row.addWidget(self.backend)
        model.addLayout(backend_row)
        
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("File"))
        self.model_path = QLineEdit()
        self.model_path.setPlaceholderText("models/model.onnx")
        self.model_path.textChanged.connect(lambda v: setattr(self.config.detection, 'model_path', v))
        model_row.addWidget(self.model_path)
        self.browse = QPushButton("...")
        self.browse.setFixedWidth(30)
        self.browse.clicked.connect(self._browse)
        model_row.addWidget(self.browse)
        model.addLayout(model_row)
        
        self.conf = SliderWidget("Confidence", 10, 99, 60, suffix="%")
        self.conf.valueChanged.connect(lambda v: setattr(self.config.detection, 'confidence_threshold', v / 100))
        model.addWidget(self.conf)
        
        layout.addWidget(model)
        
        # Aim Point
        aim = SectionWidget("Aim Point")
        
        bone_row = QHBoxLayout()
        bone_row.addWidget(QLabel("Target"))
        bone_row.addStretch()
        self.bone = QComboBox()
        self.bone.addItems(["Top Head", "Upper Head", "Head", "Neck", "Chest"])
        self.bone.setFixedWidth(100)
        self.bone.currentTextChanged.connect(self._on_bone)
        bone_row.addWidget(self.bone)
        aim.addLayout(bone_row)
        
        self.bone_scale = SliderWidget("Scale", 0.5, 2.0, 1.0, decimals=2)
        self.bone_scale.valueChanged.connect(lambda v: setattr(self.config.detection, 'bone_scale', v))
        aim.addWidget(self.bone_scale)
        
        layout.addWidget(aim)
        layout.addStretch()
        
        scroll.setWidget(content)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)
    
    def _load_config(self):
        # Block signals
        self.model_path.blockSignals(True)
        self.conf.blockSignals(True)
        self.bone_scale.blockSignals(True)
        self.bone.blockSignals(True)
        
        d = self.config.detection
        self.model_path.setText(d.model_path)
        self.conf.setValue(d.confidence_threshold * 100)
        self.bone_scale.setValue(d.bone_scale)
        
        bone_map = {"top_head": "Top Head", "upper_head": "Upper Head", "head": "Head",
                    "neck": "Neck", "chest": "Chest"}
        self.bone.setCurrentText(bone_map.get(d.aim_bone, "Upper Head"))
        
        # Unblock signals
        self.model_path.blockSignals(False)
        self.conf.blockSignals(False)
        self.bone_scale.blockSignals(False)
        self.bone.blockSignals(False)
    
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "models",
                                               "Model Files (*.onnx *.engine *.enc *.onnx.enc)")
        if path:
            self.model_path.setText(path)
    
    def _on_bone(self, text):
        bone_map = {"Top Head": "top_head", "Upper Head": "upper_head", "Head": "head",
                    "Neck": "neck", "Chest": "chest"}
        self.config.detection.aim_bone = bone_map.get(text, "upper_head")


class MouseSettings(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._load_config()
    
    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 10, 0)
        
        # Device
        dev = SectionWidget("Device")
        
        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Method"))
        dev_row.addStretch()
        self.device = QComboBox()
        self.device.addItems(["Internal (SendInput)"])
        self.device.setFixedWidth(140)
        dev_row.addWidget(self.device)
        dev.addLayout(dev_row)
        
        layout.addWidget(dev)
        
        # Sensitivity
        sens = SectionWidget("Sensitivity")
        
        self.sens_enabled = ToggleWithStatus("Normalization", True, "On", "Off")
        self.sens_enabled.toggled.connect(lambda v: setattr(self.config.mouse, 'sens_normalization_enabled', v))
        sens.addWidget(self.sens_enabled)
        
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI"))
        dpi_row.addStretch()
        self.dpi = QSpinBox()
        self.dpi.setRange(100, 16000)
        self.dpi.setValue(800)
        self.dpi.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.dpi.setFixedWidth(70)
        self.dpi.setKeyboardTracking(False)
        self.dpi.valueChanged.connect(self._on_sens)
        dpi_row.addWidget(self.dpi)
        sens.addLayout(dpi_row)
        
        ig_row = QHBoxLayout()
        ig_row.addWidget(QLabel("In-Game Sens"))
        ig_row.addStretch()
        self.ig_sens = QDoubleSpinBox()
        self.ig_sens.setRange(0.01, 10.0)
        self.ig_sens.setDecimals(2)
        self.ig_sens.setValue(0.41)
        self.ig_sens.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.ig_sens.setFixedWidth(70)
        self.ig_sens.setKeyboardTracking(False)
        self.ig_sens.valueChanged.connect(self._on_sens)
        ig_row.addWidget(self.ig_sens)
        sens.addLayout(ig_row)
        
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Reference Sens"))
        ref_row.addStretch()
        self.ref_sens = QDoubleSpinBox()
        self.ref_sens.setRange(0.01, 10.0)
        self.ref_sens.setDecimals(2)
        self.ref_sens.setValue(0.70)
        self.ref_sens.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.ref_sens.setFixedWidth(70)
        self.ref_sens.setKeyboardTracking(False)
        self.ref_sens.valueChanged.connect(self._on_sens)
        ref_row.addWidget(self.ref_sens)
        sens.addLayout(ref_row)
        
        self.edpi = QLabel("eDPI: 328.0")
        self.edpi.setStyleSheet("color: #64748b; font-size: 10px;")
        sens.addWidget(self.edpi)
        
        layout.addWidget(sens)
        
        # Delays
        delays = SectionWidget("Delays")
        
        self.aim_delay = SliderWidget("Aim", 0, 20, 3, suffix="ms")
        self.aim_delay.valueChanged.connect(lambda v: setattr(self.config.mouse, 'aim_delay', int(v)))
        delays.addWidget(self.aim_delay)
        
        self.flick_delay = SliderWidget("Flick", 0, 20, 3, suffix="ms")
        self.flick_delay.valueChanged.connect(lambda v: setattr(self.config.mouse, 'flick_delay', int(v)))
        delays.addWidget(self.flick_delay)
        
        self.sens_mult = SliderWidget("Sens Mult", 0.1, 3.0, 1.0, decimals=2)
        self.sens_mult.valueChanged.connect(lambda v: setattr(self.config.mouse, 'sens_multiplier', v))
        delays.addWidget(self.sens_mult)
        
        layout.addWidget(delays)
        layout.addStretch()
        
        scroll.setWidget(content)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)
    
    def _load_config(self):
        # Block signals
        self.sens_enabled.blockSignals(True)
        self.dpi.blockSignals(True)
        self.ig_sens.blockSignals(True)
        self.ref_sens.blockSignals(True)
        self.aim_delay.blockSignals(True)
        self.flick_delay.blockSignals(True)
        self.sens_mult.blockSignals(True)
        
        m = self.config.mouse
        self.sens_enabled.setChecked(m.sens_normalization_enabled)
        self.dpi.setValue(m.dpi_value)
        self.ig_sens.setValue(m.in_game_sens)
        self.ref_sens.setValue(m.reference_sens)
        self.aim_delay.setValue(m.aim_delay)
        self.flick_delay.setValue(m.flick_delay)
        self.sens_mult.setValue(m.sens_multiplier)
        self._update_edpi()
        
        # Unblock signals
        self.sens_enabled.blockSignals(False)
        self.dpi.blockSignals(False)
        self.ig_sens.blockSignals(False)
        self.ref_sens.blockSignals(False)
        self.aim_delay.blockSignals(False)
        self.flick_delay.blockSignals(False)
        self.sens_mult.blockSignals(False)
    
    def _on_sens(self, v):
        self.config.mouse.dpi_value = self.dpi.value()
        self.config.mouse.in_game_sens = self.ig_sens.value()
        self.config.mouse.reference_sens = self.ref_sens.value()
        self._update_edpi()
    
    def _update_edpi(self):
        edpi = self.dpi.value() * self.ig_sens.value()
        self.edpi.setText(f"eDPI: {edpi:.1f}")


class TrackingSettings(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
        self._load_config()
    
    def _setup_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background-color: transparent;")
        
        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(content)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 10, 0)
        
        # Basic
        basic = SectionWidget("Basic")
        
        self.min_smooth = SliderWidget("Min Smooth", 0, 20, 2, decimals=1)
        self.min_smooth.valueChanged.connect(lambda v: setattr(self.config.tracking, 'min_smoothing', v))
        basic.addWidget(self.min_smooth)
        
        self.deadzone = SliderWidget("Deadzone", 0, 20, 1)
        self.deadzone.valueChanged.connect(lambda v: setattr(self.config.tracking, 'tracking_deadzone', int(v)))
        basic.addWidget(self.deadzone)
        
        self.extra = SliderWidget("Extra Smooth", 0, 20, 1, decimals=1)
        self.extra.valueChanged.connect(lambda v: setattr(self.config.tracking, 'extra_smoothing', v))
        basic.addWidget(self.extra)
        
        layout.addWidget(basic)
        
        # Clamping
        clamp = SectionWidget("Movement Clamping")
        
        self.clamp_enabled = LabeledToggle("Enable")
        self.clamp_enabled.toggled.connect(lambda v: setattr(self.config.tracking, 'movement_clamping_enabled', v))
        clamp.addWidget(self.clamp_enabled)
        
        self.clamp_fov = SliderWidget("Clamp FOV", 10, 200, 50, suffix="px")
        self.clamp_fov.valueChanged.connect(lambda v: setattr(self.config.tracking, 'clamp_fov', int(v)))
        clamp.addWidget(self.clamp_fov)
        
        self.clamp_max = SliderWidget("Clamp Max", 1, 50, 14, suffix="px")
        self.clamp_max.valueChanged.connect(lambda v: setattr(self.config.tracking, 'clamp_max', int(v)))
        clamp.addWidget(self.clamp_max)
        
        layout.addWidget(clamp)
        layout.addStretch()
        
        scroll.setWidget(content)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)
    
    def _load_config(self):
        # Block signals
        self.min_smooth.blockSignals(True)
        self.deadzone.blockSignals(True)
        self.extra.blockSignals(True)
        self.clamp_enabled.blockSignals(True)
        self.clamp_fov.blockSignals(True)
        self.clamp_max.blockSignals(True)
        
        t = self.config.tracking
        self.min_smooth.setValue(t.min_smoothing)
        self.deadzone.setValue(t.tracking_deadzone)
        self.extra.setValue(t.extra_smoothing)
        self.clamp_enabled.setChecked(t.movement_clamping_enabled)
        self.clamp_fov.setValue(t.clamp_fov)
        self.clamp_max.setValue(t.clamp_max)
        
        # Unblock signals
        self.min_smooth.blockSignals(False)
        self.deadzone.blockSignals(False)
        self.extra.blockSignals(False)
        self.clamp_enabled.blockSignals(False)
        self.clamp_fov.blockSignals(False)
        self.clamp_max.blockSignals(False)


class SettingsTab(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
                margin-right: 4px;
                color: #64748b;
                font-size: 11px;
            }
            QTabBar::tab:hover { background: rgba(255,255,255,0.04); color: #94a3b8; }
            QTabBar::tab:selected { background: #8b5cf6; color: white; }
        """)
        
        self.tabs.addTab(CaptureSettings(self.config), "Capture")
        self.tabs.addTab(DetectionSettings(self.config), "Detection")
        self.tabs.addTab(MouseSettings(self.config), "Mouse")
        self.tabs.addTab(TrackingSettings(self.config), "Tracking")
        
        layout.addWidget(self.tabs)
