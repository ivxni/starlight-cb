"""
Debug View Widget - Minimal Design
Square preview with stats
Optimized for performance - no freezing
"""

import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


class DebugView(QWidget):
    """Minimal debug view - square preview (optimized)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._frame = None
        self._frame_dirty = False  # Only redraw when new frame arrives
        self._detections = []
        self._aim_target = None
        self._flick_target = None
        self._aim_fov = 25
        self._flick_fov = 30
        self._aim_key_down = False
        self._flick_key_down = False
        self._trigger_key_down = False
        # Feature enabled states
        self._aim_enabled = True
        self._flick_enabled = True
        self._trigger_enabled = True
        
        self._capture_fps = 0.0
        self._detection_fps = 0.0
        self._latency_ms = 0.0
        self._program_latency_ms = 0.0
        self._frame_age_ms = 0.0
        self._lifecycle_state = "Idle"
        
        # Performance: only redraw when new frame available
        # Timer controls max fps, dirty flag prevents redundant draws
        
        self._setup_ui()
        
        # Timer at ~120fps for smooth debug display
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_display)
        self._timer.start(16)  # ~60fps (matches frame timer)
    
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

        # Row 3
        stats_layout.addWidget(self._stat_label("AGE"), 2, 0)
        self.age_ms = self._value_label("#eab308")
        stats_layout.addWidget(self.age_ms, 2, 1)

        stats_layout.addWidget(self._stat_label("STATE"), 2, 2)
        self.state_lbl = self._value_label("#22c55e")
        stats_layout.addWidget(self.state_lbl, 2, 3)

        # Row 4
        stats_layout.addWidget(self._stat_label("DEV"), 3, 0)
        self.dev_lbl = self._value_label("#22c55e")
        stats_layout.addWidget(self.dev_lbl, 3, 1)
        
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
        """Update frame - uses reference, copies only when displaying"""
        if frame is not None:
            # Store reference, don't copy here (copy in _update_display)
            self._frame = frame
            self._frame_dirty = True
    
    def update_detections(self, detections: list, target=None):
        self._detections = detections
        # Backward compatibility: old call passes aim target as `target`
        self._aim_target = target
    
    def update_targets(self, aim_target=None, flick_target=None):
        self._aim_target = aim_target
        self._flick_target = flick_target
    
    def update_metrics(self, capture_fps: float, detection_fps: float,
                       latency_ms: float, program_latency_ms: float = 0,
                       frame_age_ms: float = 0, lifecycle_state: str = "Idle",
                       detector_device: str = ""):
        self._capture_fps = capture_fps
        self._detection_fps = detection_fps
        self._latency_ms = latency_ms
        self._program_latency_ms = program_latency_ms
        self._frame_age_ms = frame_age_ms
        self._lifecycle_state = lifecycle_state
        
        self.cap_fps.setText(f"{capture_fps:.0f}")
        self.det_fps.setText(f"{detection_fps:.0f}")
        self.inf_ms.setText(f"{latency_ms:.1f}ms")
        self.lat_ms.setText(f"{program_latency_ms:.1f}ms")
        self.age_ms.setText(f"{frame_age_ms:.1f}ms")
        self.state_lbl.setText(lifecycle_state)
        self.dev_lbl.setText(detector_device or "--")
        
        if capture_fps > 0 or lifecycle_state != "Idle":
            # Only show key states for enabled features
            keys_parts = []
            if self._aim_enabled:
                keys_parts.append(f"A:{int(self._aim_key_down)}")
            if self._flick_enabled:
                keys_parts.append(f"F:{int(self._flick_key_down)}")
            if self._trigger_enabled:
                keys_parts.append(f"T:{int(self._trigger_key_down)}")
            keys = " ".join(keys_parts) if keys_parts else "No features"
            self.status.setText(f"{lifecycle_state} | {keys}")
            self.status.setStyleSheet("color: #22c55e; font-size: 9px; background: transparent; border: none;")
        else:
            self.status.setText("Idle")
            self.status.setStyleSheet("color: #64748b; font-size: 9px; background: transparent; border: none;")

    def update_key_states(self, aim_down: bool, flick_down: bool, trigger_down: bool):
        self._aim_key_down = aim_down
        self._flick_key_down = flick_down
        self._trigger_key_down = trigger_down
    
    def set_aim_fov(self, fov: int):
        self._aim_fov = fov
    
    def set_flick_fov(self, fov: int):
        self._flick_fov = fov
    
    def set_feature_enabled(self, aim: bool, flick: bool, trigger: bool):
        """Set which features are enabled (only show FOV for enabled features)"""
        self._aim_enabled = aim
        self._flick_enabled = flick
        self._trigger_enabled = trigger
    
    def _update_display(self):
        """Update display - heavily optimized to prevent UI freezing"""
        if self._frame is None:
            return
        
        # Skip if no new frame available
        if not self._frame_dirty:
            return
        
        self._frame_dirty = False
        
        try:
            # Get target display size (small!)
            target_size = 240
            
            # Get original frame dimensions
            orig_h, orig_w = self._frame.shape[:2]
            
            # Calculate scale factor
            scale = target_size / max(orig_h, orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # RESIZE FIRST - much faster to process small image
            # Use INTER_NEAREST for speed (INTER_LINEAR is slower but smoother)
            small = cv2.resize(self._frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            # Calculate center and scaled FOV for small image
            cx, cy = new_w // 2, new_h // 2
            scaled_aim_fov = int(self._aim_fov * scale)
            scaled_flick_fov = int(self._flick_fov * scale)
            
            # FOV circles - only show if feature is enabled
            if self._aim_enabled and scaled_aim_fov > 1:
                cv2.circle(small, (cx, cy), scaled_aim_fov, (139, 92, 246), 1, cv2.LINE_4)
            if self._flick_enabled and scaled_flick_fov > 1:
                cv2.circle(small, (cx, cy), scaled_flick_fov, (34, 211, 238), 1, cv2.LINE_4)
            
            # Detections - limit to first 5 for performance
            for det in self._detections[:5]:
                # Scale detection coordinates
                x1 = int(det.x1 * scale)
                y1 = int(det.y1 * scale)
                x2 = int(det.x2 * scale)
                y2 = int(det.y2 * scale)
                
                is_aim = bool(self._aim_enabled and self._aim_target and det == self._aim_target.detection)
                is_flick = bool(self._flick_enabled and self._flick_target and det == self._flick_target.detection)
                
                if is_aim:
                    color = (0, 255, 0)
                    thick = 2
                elif is_flick:
                    color = (255, 255, 0)
                    thick = 2
                else:
                    color = (0, 255, 255)
                    thick = 1
                cv2.rectangle(small, (x1, y1), (x2, y2), color, thick, cv2.LINE_4)
                
                # Show confidence score and class ID for AI detection
                conf = getattr(det, 'confidence', 0)
                class_id = getattr(det, 'class_id', 0)
                if conf > 0:
                    label = f"C{class_id}:{conf*100:.0f}%"
                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                    cv2.rectangle(small, (x1, y1-th-2), (x1+tw+2, y1), color, -1, cv2.LINE_4)
                    cv2.putText(small, label, (x1+1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Target line - scaled
            if self._aim_enabled and self._aim_target:
                tx = int(self._aim_target.aim_x * scale)
                ty = int(self._aim_target.aim_y * scale)
                cv2.line(small, (cx, cy), (tx, ty), (0, 255, 0), 1, cv2.LINE_4)
            if self._flick_enabled and self._flick_target:
                tx = int(self._flick_target.aim_x * scale)
                ty = int(self._flick_target.aim_y * scale)
                cv2.line(small, (cx, cy), (tx, ty), (255, 255, 0), 1, cv2.LINE_4)
            
            # Convert BGR to RGB - on small image, very fast
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            # Create QImage directly from numpy array
            qimg = QImage(rgb.data, new_w, new_h, 3 * new_w, QImage.Format.Format_RGB888)
            
            # Convert to pixmap - must copy since rgb array will go out of scope
            pixmap = QPixmap.fromImage(qimg.copy())
            
            self.frame_label.setPixmap(pixmap)
            
        except Exception:
            # Silently ignore display errors
            pass
    
    def start(self):
        if not self._timer.isActive():
            self._timer.start()
    
    def stop(self):
        self._timer.stop()
