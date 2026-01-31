"""
Screen Capture Module
Uses DXGI Desktop Duplication for low-latency screen capture
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple
from dataclasses import dataclass
from threading import Thread, Lock
import ctypes

try:
    import dxcam
    DXCAM_AVAILABLE = True
except ImportError:
    DXCAM_AVAILABLE = False
    print("Warning: dxcam not available, falling back to mss")

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False


@dataclass
class CaptureRegion:
    """Defines the capture region (cropped area)"""
    left: int
    top: int
    width: int
    height: int
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Returns (left, top, right, bottom)"""
        return (self.left, self.top, self.right, self.bottom)


class ScreenCapture:
    """
    High-performance screen capture using DXGI Desktop Duplication
    Captures a cropped region around screen center for AI processing
    """
    
    def __init__(self, capture_width: int = 640, capture_height: int = 640, 
                 target_fps: int = 240, monitor: int = 0):
        """
        Initialize screen capture
        
        Args:
            capture_width: Width of capture region (centered on screen)
            capture_height: Height of capture region (centered on screen)
            target_fps: Target capture framerate
            monitor: Monitor index to capture
        """
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.target_fps = target_fps
        self.monitor_idx = monitor
        
        # Get screen dimensions
        user32 = ctypes.windll.user32
        self.screen_width = user32.GetSystemMetrics(0)
        self.screen_height = user32.GetSystemMetrics(1)
        
        # Calculate center crop region
        self.region = self._calculate_center_region()
        
        # Capture backend
        self.camera = None
        self._backend = None
        self._mss = None
        
        # Frame buffer
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = Lock()
        self._frame_time: float = 0
        
        # Performance tracking
        self._capture_count: int = 0
        self._capture_fps: float = 0
        self._last_fps_time: float = 0
        
        # Threading
        self._running: bool = False
        self._capture_thread: Optional[Thread] = None
        
    def _calculate_center_region(self) -> CaptureRegion:
        """Calculate the center crop region"""
        left = (self.screen_width - self.capture_width) // 2
        top = (self.screen_height - self.capture_height) // 2
        return CaptureRegion(left, top, self.capture_width, self.capture_height)
    
    def start(self) -> bool:
        """Start screen capture"""
        if self._running:
            return True
            
        # Try dxcam first (fastest)
        if DXCAM_AVAILABLE:
            try:
                self.camera = dxcam.create(
                    device_idx=0,
                    output_idx=self.monitor_idx,
                    output_color="BGR"
                )
                self.camera.start(
                    region=self.region.to_tuple(),
                    target_fps=self.target_fps,
                    video_mode=True
                )
                self._backend = "dxcam"
                self._running = True
                
                # Start capture thread
                self._capture_thread = Thread(target=self._capture_loop_dxcam, daemon=True)
                self._capture_thread.start()
                
                print(f"Screen capture started (dxcam) - Region: {self.region.to_tuple()}")
                return True
            except Exception as e:
                print(f"dxcam failed: {e}")
                self.camera = None
        
        # Fallback to mss
        if MSS_AVAILABLE:
            try:
                self._mss = mss.mss()
                self._backend = "mss"
                self._running = True
                
                # Start capture thread
                self._capture_thread = Thread(target=self._capture_loop_mss, daemon=True)
                self._capture_thread.start()
                
                print(f"Screen capture started (mss) - Region: {self.region.to_tuple()}")
                return True
            except Exception as e:
                print(f"mss failed: {e}")
                
        print("No capture backend available!")
        return False
    
    def stop(self):
        """Stop screen capture"""
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
            
        if self.camera:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None
            
        if self._mss:
            self._mss.close()
            self._mss = None
            
        print("Screen capture stopped")
    
    def _capture_loop_dxcam(self):
        """Capture loop using dxcam"""
        while self._running:
            try:
                frame = self.camera.get_latest_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._frame = frame
                        self._frame_time = time.perf_counter()
                    self._update_fps()
            except Exception as e:
                time.sleep(0.001)
    
    def _capture_loop_mss(self):
        """Capture loop using mss"""
        monitor = {
            "left": self.region.left,
            "top": self.region.top,
            "width": self.region.width,
            "height": self.region.height
        }
        
        target_interval = 1.0 / self.target_fps
        
        while self._running:
            start = time.perf_counter()
            try:
                screenshot = self._mss.grab(monitor)
                frame = np.array(screenshot)
                # Convert BGRA to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                with self._frame_lock:
                    self._frame = frame
                    self._frame_time = time.perf_counter()
                self._update_fps()
            except Exception as e:
                pass
                
            # Rate limiting
            elapsed = time.perf_counter() - start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
    
    def _update_fps(self):
        """Update FPS counter"""
        self._capture_count += 1
        now = time.perf_counter()
        
        if now - self._last_fps_time >= 1.0:
            self._capture_fps = self._capture_count / (now - self._last_fps_time)
            self._capture_count = 0
            self._last_fps_time = now
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest captured frame
        
        Returns:
            BGR numpy array of shape (height, width, 3) or None
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None
    
    def get_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Get the latest frame with its timestamp
        
        Returns:
            Tuple of (frame, timestamp)
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy(), self._frame_time
        return None, 0
    
    @property
    def fps(self) -> float:
        """Get current capture FPS"""
        return self._capture_fps
    
    @property
    def backend(self) -> str:
        """Get current capture backend name"""
        return self._backend or "none"
    
    @property
    def is_running(self) -> bool:
        """Check if capture is running"""
        return self._running
    
    def update_region(self, width: int, height: int):
        """Update capture region size"""
        self.capture_width = width
        self.capture_height = height
        self.region = self._calculate_center_region()
        
        # Restart capture with new region if running
        if self._running:
            was_running = True
            self.stop()
            self.start()
    
    @property
    def screen_center(self) -> Tuple[int, int]:
        """Get screen center coordinates"""
        return (self.screen_width // 2, self.screen_height // 2)
    
    def frame_to_screen_coords(self, frame_x: int, frame_y: int) -> Tuple[int, int]:
        """Convert frame coordinates to screen coordinates"""
        screen_x = self.region.left + frame_x
        screen_y = self.region.top + frame_y
        return (screen_x, screen_y)
    
    def screen_to_frame_coords(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to frame coordinates"""
        frame_x = screen_x - self.region.left
        frame_y = screen_y - self.region.top
        return (frame_x, frame_y)
