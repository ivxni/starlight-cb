"""
Debug View Widget - Minimal Design
Square preview with stats
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


class DebugView(QWidget):
    """Minimal debug view - square preview"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._frame = None
        self._detections = []
        self._target = None
        self._aim_fov = 25
        
        self._capture_fps = 0.0
        self._detection_fps = 0.0
        self._latency_ms = 0.0
        self._program_latency_ms = 0.0
        
        self._setup_ui()
        
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_display)
        self._timer.start(16)
    
    def _setup_ui(self):
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 6px;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("Debug")
        header.setStyleSheet("""
            color: #64748b;
            font-size: 9px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: transparent;
            border: none;
        """)
        layout.addWidget(header)
        
        # Frame - square container
        self.frame_container = QFrame()
        self.frame_container.setStyleSheet("""
            QFrame {
                background-color: rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 255, 255, 0.04);
                border-radius: 4px;
            }
        """)
        # Make it square - match width
        self.frame_container.setMinimumSize(240, 240)
        self.frame_container.setMaximumSize(280, 280)
        
        frame_layout = QVBoxLayout(self.frame_container)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setStyleSheet("background: transparent; border: none;")
        frame_layout.addWidget(self.frame_label)
        
        layout.addWidget(self.frame_container)
        
        # Stats - compact grid
        stats = QFrame()
        stats.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.01);
                border: 1px solid rgba(255, 255, 255, 0.03);
                border-radius: 4px;
            }
        """)
        
        stats_layout = QGridLayout(stats)
        stats_layout.setContentsMargins(8, 6, 8, 6)
        stats_layout.setSpacing(4)
        
        # Row 1
        stats_layout.addWidget(self._stat_label("CAP"), 0, 0)
        self.cap_fps = self._value_label("#22c55e")
        stats_layout.addWidget(self.cap_fps, 0, 1)
        
        stats_layout.addWidget(self._stat_label("DET"), 0, 2)
        self.det_fps = self._value_label("#22c55e")
        stats_layout.addWidget(self.det_fps, 0, 3)
        
        # Row 2
        stats_layout.addWidget(self._stat_label("INF"), 1, 0)
        self.inf_ms = self._value_label("#eab308")
        stats_layout.addWidget(self.inf_ms, 1, 1)
        
        stats_layout.addWidget(self._stat_label("LAT"), 1, 2)
        self.lat_ms = self._value_label("#eab308")
        stats_layout.addWidget(self.lat_ms, 1, 3)
        
        layout.addWidget(stats)
        
        # Status
        self.status = QLabel("Idle")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("color: #64748b; font-size: 9px; background: transparent; border: none;")
        layout.addWidget(self.status)
    
    def _stat_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #475569; font-size: 9px; background: transparent; border: none;")
        return lbl
    
    def _value_label(self, color: str) -> QLabel:
        lbl = QLabel("--")
        lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-family: 'Consolas'; font-weight: 600; background: transparent; border: none;")
        return lbl
    
    def update_frame(self, frame: np.ndarray):
        if frame is not None:
            self._frame = frame.copy()
    
    def update_detections(self, detections: list, target=None):
        self._detections = detections
        self._target = target
    
    def update_metrics(self, capture_fps: float, detection_fps: float,
                       latency_ms: float, program_latency_ms: float = 0):
        self._capture_fps = capture_fps
        self._detection_fps = detection_fps
        self._latency_ms = latency_ms
        self._program_latency_ms = program_latency_ms
        
        self.cap_fps.setText(f"{capture_fps:.0f}")
        self.det_fps.setText(f"{detection_fps:.0f}")
        self.inf_ms.setText(f"{latency_ms:.1f}ms")
        self.lat_ms.setText(f"{program_latency_ms:.1f}ms")
        
        if capture_fps > 0:
            self.status.setText("Running")
            self.status.setStyleSheet("color: #22c55e; font-size: 9px; background: transparent; border: none;")
        else:
            self.status.setText("Idle")
            self.status.setStyleSheet("color: #64748b; font-size: 9px; background: transparent; border: none;")
    
    def set_aim_fov(self, fov: int):
        self._aim_fov = fov
    
    def _update_display(self):
        if self._frame is None:
            return
        
        display = self._frame.copy()
        h, w = display.shape[:2]
        cx, cy = w // 2, h // 2
        
        # FOV circle
        cv2.circle(display, (cx, cy), self._aim_fov, (139, 92, 246), 1, cv2.LINE_AA)
        
        # Crosshair
        cv2.line(display, (cx - 6, cy), (cx + 6, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(display, (cx, cy - 6), (cx, cy + 6), (255, 255, 255), 1, cv2.LINE_AA)
        
        # Detections - yellow
        for det in self._detections:
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            is_target = self._target and det == self._target.detection
            color = (0, 255, 0) if is_target else (0, 255, 255)
            thick = 2 if is_target else 1
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
            
            # Aim point
            aim_x, aim_y = det.get_aim_point("upper_head", 1.0)
            cv2.circle(display, (int(aim_x), int(aim_y)), 2, (0, 0, 255), -1, cv2.LINE_AA)
        
        # Target line
        if self._target:
            cv2.line(display, (cx, cy), (int(self._target.aim_x), int(self._target.aim_y)),
                     (0, 255, 0), 1, cv2.LINE_AA)
        
        # Convert to pixmap
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale to fit square
        container_size = min(self.frame_container.width(), self.frame_container.height()) - 4
        scaled = pixmap.scaled(container_size, container_size,
                               Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        
        self.frame_label.setPixmap(scaled)
    
    def start(self):
        if not self._timer.isActive():
            self._timer.start()
    
    def stop(self):
        self._timer.stop()
