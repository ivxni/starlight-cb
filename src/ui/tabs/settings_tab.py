"""
Settings Tab - Minimal Design
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QScrollArea, QFrame, QTabWidget,
    QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QLineEdit, QPushButton,
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
        
        # Capture Source
        cap = SectionWidget("Capture Source")
        
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
        source_row.addStretch()
        self.source_mode = QComboBox()
        self.source_mode.addItems(["Screen"])
        self.source_mode.setFixedWidth(140)
        self.source_mode.currentTextChanged.connect(self._on_source_change)
        source_row.addWidget(self.source_mode)
        cap.addLayout(source_row)
        
        layout.addWidget(cap)
        
        # Resolution
        res = SectionWidget("Resolution")
        
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Width"))
        self.width = QSpinBox()
        # allow small crop sizes like 120x120
        self.width.setRange(32, 3840)
        self.width.setValue(640)
        self.width.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        # If user types out-of-range, clamp instead of reverting to previous value
        self.width.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        # Don't emit intermediate value changes while typing
        self.width.setKeyboardTracking(False)
        self.width.setFixedWidth(70)
        self.width.editingFinished.connect(self._on_width_changed)
        res_row.addWidget(self.width)
        
        res_row.addSpacing(10)
        res_row.addWidget(QLabel("Height"))
        self.height = QSpinBox()
        self.height.setRange(32, 2160)
        self.height.setValue(640)
        self.height.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.height.setCorrectionMode(QAbstractSpinBox.CorrectionMode.CorrectToNearestValue)
        self.height.setKeyboardTracking(False)
        self.height.setFixedWidth(70)
        self.height.editingFinished.connect(self._on_height_changed)
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
        # Block signals to prevent feedback loops while populating UI
        self.source_mode.blockSignals(True)
        self.width.blockSignals(True)
        self.height.blockSignals(True)
        self.max_fps.blockSignals(True)
        self.dbg_scale.blockSignals(True)
        self.dbg_fps.blockSignals(True)
        
        c = self.config.capture
        
        mode_map = {"screen": "Screen"}
        self.source_mode.setCurrentText(mode_map.get(c.capture_mode, "Screen"))
        
        self.width.setValue(int(c.capture_width))
        self.height.setValue(int(c.capture_height))
        self.max_fps.setValue(int(c.max_fps))
        self.dbg_scale.setValue(float(c.debug_scale))
        self.dbg_fps.setValue(int(c.debug_max_fps))
        
        # Re-enable signals
        self.source_mode.blockSignals(False)
        self.width.blockSignals(False)
        self.height.blockSignals(False)
        self.max_fps.blockSignals(False)
        self.dbg_scale.blockSignals(False)
        self.dbg_fps.blockSignals(False)
    
    def _on_source_change(self, text: str):
        mode_map = {"Screen": "screen", "OBS Virtual Camera": "obs"}
        self.config.capture.capture_mode = mode_map.get(text, "screen")

    def _on_width_changed(self):
        self.config.capture.capture_width = int(self.width.value())

    def _on_height_changed(self):
        self.config.capture.capture_height = int(self.height.value())


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
        
        # Color Detection Section
        self.color_section = SectionWidget("Color Settings")
        
        # Color Preset
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        preset_row.addStretch()
        self.color_preset = QComboBox()
        self.color_preset.addItems(["purple", "purple2", "purple3", "yellow", "yellow2", "red", "red2", "custom"])
        self.color_preset.setFixedWidth(100)
        self.color_preset.currentTextChanged.connect(self._on_preset_change)
        preset_row.addWidget(self.color_preset)
        self.color_section.addLayout(preset_row)

        color_device_row = QHBoxLayout()
        color_device_row.addWidget(QLabel("Device"))
        color_device_row.addStretch()
        self.color_device = QComboBox()
        self.color_device.addItems(["CPU", "GPU"])
        self.color_device.setFixedWidth(100)
        self.color_device.currentTextChanged.connect(lambda v: setattr(self.config.detection, 'color_device', v.lower()))
        color_device_row.addWidget(self.color_device)
        self.color_section.addLayout(color_device_row)
        
        # HSV Hue
        self.color_h_min = SliderWidget("Hue Min", 0, 180, 130)
        self.color_h_min.valueChanged.connect(lambda v: self._set_custom("color_h_min", int(v)))
        self.color_section.addWidget(self.color_h_min)
        
        self.color_h_max = SliderWidget("Hue Max", 0, 180, 160)
        self.color_h_max.valueChanged.connect(lambda v: self._set_custom("color_h_max", int(v)))
        self.color_section.addWidget(self.color_h_max)
        
        # HSV Saturation
        self.color_s_min = SliderWidget("Sat Min", 0, 255, 100)
        self.color_s_min.valueChanged.connect(lambda v: self._set_custom("color_s_min", int(v)))
        self.color_section.addWidget(self.color_s_min)
        
        self.color_s_max = SliderWidget("Sat Max", 0, 255, 255)
        self.color_s_max.valueChanged.connect(lambda v: self._set_custom("color_s_max", int(v)))
        self.color_section.addWidget(self.color_s_max)
        
        # HSV Value
        self.color_v_min = SliderWidget("Val Min", 0, 255, 100)
        self.color_v_min.valueChanged.connect(lambda v: self._set_custom("color_v_min", int(v)))
        self.color_section.addWidget(self.color_v_min)
        
        self.color_v_max = SliderWidget("Val Max", 0, 255, 255)
        self.color_v_max.valueChanged.connect(lambda v: self._set_custom("color_v_max", int(v)))
        self.color_section.addWidget(self.color_v_max)
        
        layout.addWidget(self.color_section)
        
        # Color Morphology Section
        self.morph_section = SectionWidget("Color Filtering")
        
        self.blur_enabled = LabeledToggle("Gaussian Blur")
        self.blur_enabled.setChecked(True)
        self.blur_enabled.toggled.connect(lambda v: setattr(self.config.detection, 'color_blur_enabled', v))
        self.morph_section.addWidget(self.blur_enabled)
        
        self.blur_size = SliderWidget("Blur Size", 1, 15, 3)
        self.blur_size.valueChanged.connect(lambda v: setattr(self.config.detection, 'color_blur_size', int(v)))
        self.morph_section.addWidget(self.blur_size)
        
        self.color_dilate = SliderWidget("Dilate", 0, 10, 2)
        self.color_dilate.valueChanged.connect(lambda v: setattr(self.config.detection, 'color_dilate', int(v)))
        self.morph_section.addWidget(self.color_dilate)
        
        self.color_erode = SliderWidget("Erode", 0, 10, 1)
        self.color_erode.valueChanged.connect(lambda v: setattr(self.config.detection, 'color_erode', int(v)))
        self.morph_section.addWidget(self.color_erode)
        
        self.color_min_area = SliderWidget("Min Area", 10, 1000, 50, suffix="px")
        self.color_min_area.valueChanged.connect(lambda v: setattr(self.config.detection, 'color_min_area', int(v)))
        self.morph_section.addWidget(self.color_min_area)
        
        self.color_max_area = SliderWidget("Max Area", 1000, 100000, 50000, suffix="px")
        self.color_max_area.valueChanged.connect(lambda v: setattr(self.config.detection, 'color_max_area', int(v)))
        self.morph_section.addWidget(self.color_max_area)
        
        layout.addWidget(self.morph_section)
        
        # Detection Smoothing Section (Anti-Wobble)
        self.smooth_section = SectionWidget("Anti-Wobble")
        
        self.smoothing_enabled = LabeledToggle("Enable Smoothing")
        self.smoothing_enabled.setChecked(True)
        self.smoothing_enabled.toggled.connect(lambda v: setattr(self.config.detection, 'smoothing_enabled', v))
        self.smooth_section.addWidget(self.smoothing_enabled)
        
        self.smoothing_window = SliderWidget("Frame Window", 2, 20, 5)
        self.smoothing_window.valueChanged.connect(lambda v: setattr(self.config.detection, 'smoothing_window', int(v)))
        self.smooth_section.addWidget(self.smoothing_window)
        
        self.smoothing_outlier = SliderWidget("Outlier Threshold", 10, 150, 50, suffix="px")
        self.smoothing_outlier.valueChanged.connect(lambda v: setattr(self.config.detection, 'smoothing_outlier', int(v)))
        self.smooth_section.addWidget(self.smoothing_outlier)
        
        self.bbox_smoothing = SliderWidget("BBox EMA", 0.5, 1.0, 0.95, decimals=2)
        self.bbox_smoothing.valueChanged.connect(lambda v: setattr(self.config.detection, 'bbox_smoothing', v))
        self.smooth_section.addWidget(self.bbox_smoothing)
        
        layout.addWidget(self.smooth_section)
        
        # Aim Target Settings
        aim_section = SectionWidget("Aim Target")
        
        bone_row = QHBoxLayout()
        bone_row.addWidget(QLabel("Target Point"))
        bone_row.addStretch()
        self.bone = QComboBox()
        self.bone.addItems(["Top Head", "Upper Head", "Head", "Neck", "Chest"])
        self.bone.setFixedWidth(100)
        self.bone.currentTextChanged.connect(self._on_bone)
        bone_row.addWidget(self.bone)
        aim_section.addLayout(bone_row)
        
        self.bone_scale = SliderWidget("Scale", 0.5, 2.0, 1.0, decimals=2)
        self.bone_scale.valueChanged.connect(lambda v: setattr(self.config.detection, 'bone_scale', v))
        aim_section.addWidget(self.bone_scale)
        
        self.color_head_offset = SliderWidget("Head Offset", 0.0, 1.0, 0.15, decimals=2)
        self.color_head_offset.valueChanged.connect(lambda v: setattr(self.config.detection, 'color_head_offset', v))
        aim_section.addWidget(self.color_head_offset)
        
        layout.addWidget(aim_section)
        layout.addStretch()
        
        scroll.setWidget(content)
        main = QVBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(scroll)
    
    def _load_config(self):
        # Block all signals
        widgets = [
            self.bone_scale, self.bone,
            self.color_preset, self.color_device, self.color_h_min, self.color_h_max, self.color_s_min, self.color_s_max,
            self.color_v_min, self.color_v_max, self.color_dilate, self.color_erode,
            self.color_min_area, self.color_max_area, self.color_head_offset,
            self.blur_enabled, self.blur_size, self.smoothing_enabled, self.smoothing_window,
            self.smoothing_outlier, self.bbox_smoothing
        ]
        for w in widgets:
            w.blockSignals(True)
        
        d = self.config.detection
        
        # Aim target settings
        self.bone_scale.setValue(d.bone_scale)
        bone_map = {"top_head": "Top Head", "upper_head": "Upper Head", "head": "Head",
                    "neck": "Neck", "chest": "Chest"}
        self.bone.setCurrentText(bone_map.get(d.aim_bone, "Upper Head"))
        
        # Color settings
        self.color_preset.setCurrentText(d.color_preset)
        self.color_device.setCurrentText("GPU" if d.color_device == "gpu" else "CPU")
        self.color_h_min.setValue(d.color_h_min)
        self.color_h_max.setValue(d.color_h_max)
        self.color_s_min.setValue(d.color_s_min)
        self.color_s_max.setValue(d.color_s_max)
        self.color_v_min.setValue(d.color_v_min)
        self.color_v_max.setValue(d.color_v_max)
        self.color_dilate.setValue(d.color_dilate)
        self.color_erode.setValue(d.color_erode)
        self.color_min_area.setValue(d.color_min_area)
        self.color_max_area.setValue(d.color_max_area)
        self.color_head_offset.setValue(d.color_head_offset)
        
        # Blur and smoothing
        self.blur_enabled.setChecked(d.color_blur_enabled)
        self.blur_size.setValue(d.color_blur_size)
        self.smoothing_enabled.setChecked(d.smoothing_enabled)
        self.smoothing_window.setValue(d.smoothing_window)
        self.smoothing_outlier.setValue(d.smoothing_outlier)
        self.bbox_smoothing.setValue(d.bbox_smoothing)
        
        # Unblock all signals
        for w in widgets:
            w.blockSignals(False)
    
    def _on_preset_change(self, text: str):
        """Handle color preset change"""
        self.config.detection.color_preset = text
        
        # Apply preset values to UI
        from ...detection.color_detector import COLOR_PRESETS
        if text in COLOR_PRESETS:
            preset = COLOR_PRESETS[text]
            self.color_h_min.setValue(preset["h_min"])
            self.color_h_max.setValue(preset["h_max"])
            self.color_s_min.setValue(preset["s_min"])
            self.color_s_max.setValue(preset["s_max"])
            self.color_v_min.setValue(preset["v_min"])
            self.color_v_max.setValue(preset["v_max"])
    
    def _set_custom(self, attr: str, value):
        """Set custom color value and switch to custom preset"""
        setattr(self.config.detection, attr, value)
        if self.color_preset.currentText() != "custom":
            self.color_preset.blockSignals(True)
            self.color_preset.setCurrentText("custom")
            self.config.detection.color_preset = "custom"
            self.color_preset.blockSignals(False)
    
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
        
        # Info label pointing to Mouse tab
        info_label = QLabel("Configure button blocking in Mouse tab")
        info_label.setStyleSheet("color: #64748b; font-size: 9px;")
        dev.addWidget(info_label)
        
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
        self.dpi.editingFinished.connect(self._on_sens)
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
        self.ig_sens.editingFinished.connect(self._on_sens)
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
        self.ref_sens.editingFinished.connect(self._on_sens)
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
    
    def _on_sens(self):
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
