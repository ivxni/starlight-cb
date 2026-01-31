"""
Starlight Main Window - Minimal Design
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor

from .styles import GLASSMORPHISM_STYLESHEET, COLORS
from .widgets.sidebar import Sidebar
from .widgets.debug_view import DebugView
from .tabs.aim_tab import AimTab
from .tabs.flick_trigger_tab import FlickTriggerTab
from .tabs.humanizer_tab import HumanizerTab
from .tabs.settings_tab import SettingsTab
from ..core.config import Config, save_config
from ..core.assistant import Assistant


class MainWindow(QMainWindow):
    """Main window - minimal design"""
    
    TITLES = ["Aim", "Flick & Trigger", "Humanizer", "Settings"]
    
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self.config = config
        self.assistant = None
        self._last_detections = []
        
        self._setup_window()
        self._setup_ui()
        
        self._save_timer = QTimer()
        self._save_timer.timeout.connect(lambda: save_config())
        self._save_timer.start(30000)
    
    def _setup_window(self):
        self.setWindowTitle("Starlight")
        self.setMinimumSize(1000, 650)
        self.resize(1100, 700)
        
        self.setStyleSheet(GLASSMORPHISM_STYLESHEET)
        
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(COLORS['bg_primary']))
        self.setPalette(palette)
    
    def _setup_ui(self):
        central = QWidget()
        central.setStyleSheet(f"background-color: {COLORS['bg_primary']};")
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        self.sidebar = Sidebar()
        self.sidebar.pageChanged.connect(self._on_page_change)
        main_layout.addWidget(self.sidebar)
        
        # Content
        content = QWidget()
        content.setStyleSheet("background-color: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 14, 16, 14)
        content_layout.setSpacing(12)
        
        # Header
        header = self._create_header()
        content_layout.addWidget(header)
        
        # Body
        body = QHBoxLayout()
        body.setSpacing(12)
        
        # Pages
        self.pages = QStackedWidget()
        self.pages.setStyleSheet("background-color: transparent;")
        
        self.pages.addWidget(AimTab(self.config))
        self.pages.addWidget(FlickTriggerTab(self.config))
        self.pages.addWidget(HumanizerTab(self.config))
        self.pages.addWidget(SettingsTab(self.config))
        
        body.addWidget(self.pages, stretch=1)
        
        # Debug
        self.debug_view = DebugView()
        self.debug_view.setFixedWidth(260)
        body.addWidget(self.debug_view)
        
        content_layout.addLayout(body)
        main_layout.addWidget(content, stretch=1)
    
    def _create_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(40)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Title
        self.title = QLabel("Aim")
        self.title.setStyleSheet("color: #e2e8f0; font-size: 16px; font-weight: 600;")
        layout.addWidget(self.title)
        
        layout.addStretch()
        
        # Status
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(6, 6)
        self.status_dot.setStyleSheet("background-color: #475569; border-radius: 3px;")
        layout.addWidget(self.status_dot)
        
        self.status_text = QLabel("Idle")
        self.status_text.setStyleSheet("color: #475569; font-size: 11px;")
        layout.addWidget(self.status_text)
        
        layout.addSpacing(8)
        
        # Save button
        self.save_btn = QPushButton("Save")
        self.save_btn.setFixedSize(60, 28)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                color: #e2e8f0;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3f4f63;
                border-color: #64748b;
            }
        """)
        self.save_btn.clicked.connect(self._save_config)
        layout.addWidget(self.save_btn)
        
        layout.addSpacing(4)
        
        # Start button
        self.start_btn = QPushButton("Start")
        self.start_btn.setFixedSize(70, 28)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                border: none;
                border-radius: 4px;
                color: white;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #9d6ff8;
            }
        """)
        self.start_btn.clicked.connect(self._toggle_assistant)
        layout.addWidget(self.start_btn)
        
        return header
    
    def _on_page_change(self, index: int):
        self.pages.setCurrentIndex(index)
        self.title.setText(self.TITLES[index])
    
    def _save_config(self):
        """Save config to file and show feedback"""
        save_config()
        
        # Visual feedback - briefly change button style
        self.save_btn.setText("Saved!")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                border: none;
                border-radius: 4px;
                color: white;
                font-size: 11px;
                font-weight: 500;
            }
        """)
        
        # Reset after 1.5 seconds
        QTimer.singleShot(1500, self._reset_save_btn)
    
    def _reset_save_btn(self):
        """Reset save button to normal state"""
        self.save_btn.setText("Save")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                border: 1px solid #475569;
                border-radius: 4px;
                color: #e2e8f0;
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3f4f63;
                border-color: #64748b;
            }
        """)
    
    def _toggle_assistant(self):
        if self.assistant is None or not self.assistant.is_running:
            self._start_assistant()
        else:
            self._stop_assistant()
    
    def _start_assistant(self):
        self.assistant = Assistant(self.config)
        self.assistant.on_state_change = self._on_state_change
        self.assistant.on_detection = self._on_detection
        
        if self.assistant.initialize() and self.assistant.start():
            self.start_btn.setText("Stop")
            self.start_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    border: none;
                    border-radius: 4px;
                    color: white;
                    font-size: 11px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #f87171;
                }
            """)
            
            self.status_dot.setStyleSheet("background-color: #22c55e; border-radius: 3px;")
            self.status_text.setText("Running")
            self.status_text.setStyleSheet("color: #22c55e; font-size: 11px;")
            
            self.debug_view.start()
            
            self._frame_timer = QTimer()
            self._frame_timer.timeout.connect(self._update_frame)
            self._frame_timer.start(16)
    
    def _stop_assistant(self):
        if hasattr(self, '_frame_timer') and self._frame_timer:
            self._frame_timer.stop()
            self._frame_timer = None
        
        self.debug_view.stop()
        
        if self.assistant:
            self.assistant.stop()
        
        self.start_btn.setText("Start")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b5cf6;
                border: none;
                border-radius: 4px;
                color: white;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #9d6ff8;
            }
        """)
        
        self.status_dot.setStyleSheet("background-color: #475569; border-radius: 3px;")
        self.status_text.setText("Idle")
        self.status_text.setStyleSheet("color: #475569; font-size: 11px;")
        
        self.debug_view.update_metrics(0, 0, 0, 0)
    
    def _on_state_change(self, state):
        inf_ms = 0
        if self.assistant and self.assistant.detector:
            inf_ms = self.assistant.detector.inference_time_ms
        
        self.debug_view.update_metrics(
            state.capture_fps, state.detection_fps,
            inf_ms, state.latency_ms
        )
        self.debug_view.update_detections(self._last_detections, state.current_target)
        self.debug_view.set_aim_fov(self.config.aim.aim_fov)
    
    def _on_detection(self, detections):
        self._last_detections = detections
    
    def _update_frame(self):
        if self.assistant and self.assistant.capture:
            frame = self.assistant.capture.get_frame()
            if frame is not None:
                self.debug_view.update_frame(frame)
    
    def closeEvent(self, event):
        if self.assistant and self.assistant.is_running:
            self.assistant.stop()
        save_config()
        event.accept()
