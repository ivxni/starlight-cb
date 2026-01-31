"""
Main Assistant Control Loop
Coordinates capture, detection, and mouse movement
"""

import time
import threading
import random
import ctypes
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

from ..capture.screen_capture import ScreenCapture
from ..detection.tensorrt_engine import TensorRTEngine, Detection
from ..detection.target_selector import TargetSelector, Target
from ..movement.mouse_controller import MouseController, get_mouse_controller
from ..movement.humanizer import Humanizer
from .config import Config, get_config

# Windows API for mouse button state
GetAsyncKeyState = ctypes.windll.user32.GetAsyncKeyState

# Virtual key codes
VK_LBUTTON = 0x01   # Left mouse
VK_RBUTTON = 0x02   # Right mouse
VK_MBUTTON = 0x04   # Middle mouse
VK_XBUTTON1 = 0x05  # X1 (Back)
VK_XBUTTON2 = 0x06  # X2 (Forward)


@dataclass
class AssistantState:
    """Current state of the assistant"""
    aim_active: bool = False
    flick_active: bool = False
    trigger_active: bool = False
    current_target: Optional[Target] = None
    
    # Performance metrics
    capture_fps: float = 0.0
    detection_fps: float = 0.0
    loop_fps: float = 0.0
    latency_ms: float = 0.0


class Assistant:
    """
    Main assistant that coordinates all subsystems
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize assistant
        
        Args:
            config: Configuration object (uses global if not provided)
        """
        self.config = config or get_config()
        
        # Subsystems
        self.capture: Optional[ScreenCapture] = None
        self.detector: Optional[TensorRTEngine] = None
        self.selector: Optional[TargetSelector] = None
        self.mouse: MouseController = get_mouse_controller()
        self.aim_humanizer: Humanizer = Humanizer()
        self.flick_humanizer: Humanizer = Humanizer()
        
        # State
        self.state = AssistantState()
        self._running = False
        self._paused = False
        
        # Threading
        self._main_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Input callbacks
        self._aim_key_pressed = False
        self._flick_key_pressed = False
        self._trigger_key_pressed = False
        
        # Timing
        self._last_aim_time = 0.0
        self._aim_start_time = 0.0
        self._loop_count = 0
        self._last_fps_time = 0.0
        self._last_trigger_time = 0.0  # Trigger cooldown
        
        # Callbacks for UI updates
        self.on_state_change: Optional[Callable[[AssistantState], None]] = None
        self.on_detection: Optional[Callable[[list], None]] = None
    
    def initialize(self) -> bool:
        """Initialize all subsystems"""
        try:
            # Initialize screen capture
            self.capture = ScreenCapture(
                capture_width=self.config.capture.capture_width,
                capture_height=self.config.capture.capture_height,
                target_fps=self.config.capture.max_fps,
                monitor=self.config.capture.monitor
            )
            
            # Initialize detector (only if model exists)
            try:
                self.detector = TensorRTEngine(
                    model_path=self.config.detection.model_path,
                    confidence_threshold=self.config.detection.confidence_threshold,
                    use_tensorrt=self.config.detection.use_tensorrt,
                    input_size=self.config.capture.capture_width
                )
            except Exception as e:
                print(f"Warning: Could not load AI model: {e}")
                print("Detection will be disabled until a valid model is loaded")
                self.detector = None
            
            # Initialize target selector
            self.selector = TargetSelector(
                frame_width=self.config.capture.capture_width,
                frame_height=self.config.capture.capture_height
            )
            
            # Configure mouse
            self.mouse.set_sensitivity(
                dpi=self.config.mouse.dpi_value,
                in_game=self.config.mouse.in_game_sens,
                reference=self.config.mouse.reference_sens
            )
            self.mouse.normalization_enabled = self.config.mouse.sens_normalization_enabled
            self.mouse.sens_multiplier = self.config.mouse.sens_multiplier
            
            # Configure humanizers
            self._configure_humanizer(self.aim_humanizer, self.config.humanizer)
            self._configure_humanizer(self.flick_humanizer, self.config.flick_humanizer)
            
            print("Assistant initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize assistant: {e}")
            return False
    
    def _configure_humanizer(self, humanizer: Humanizer, config):
        """Configure a humanizer from config - ALL settings"""
        # Use the load_from_config method which handles all settings
        humanizer.load_from_config(config)
    
    def start(self) -> bool:
        """Start the assistant"""
        if self._running:
            return True
        
        # Start capture
        if not self.capture.start():
            print("Failed to start screen capture")
            return False
        
        # Start main loop
        self._running = True
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        
        print("Assistant started")
        return True
    
    def stop(self):
        """Stop the assistant"""
        self._running = False
        
        if self._main_thread:
            self._main_thread.join(timeout=2.0)
            self._main_thread = None
        
        if self.capture:
            self.capture.stop()
        
        print("Assistant stopped")
    
    def pause(self):
        """Pause the assistant"""
        self._paused = True
    
    def resume(self):
        """Resume the assistant"""
        self._paused = False
    
    def _main_loop(self):
        """Main processing loop"""
        while self._running:
            start_time = time.perf_counter()
            
            if self._paused:
                time.sleep(0.01)
                continue
            
            try:
                # Check input state (mouse buttons)
                self._check_input_state()
                
                # Get frame
                frame = self.capture.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Run detection
                detections = []
                if self.detector and self.detector.is_loaded:
                    detections = self.detector.detect(frame)
                    
                    # Callback for visualization
                    if self.on_detection:
                        self.on_detection(detections)
                
                # Update selector and mouse config (live updates)
                self.selector.set_fov(self.config.aim.aim_fov)
                self.selector.set_aim_bone(
                    self.config.detection.aim_bone,
                    self.config.detection.bone_scale
                )
                self.selector.set_class_filter(
                    self.config.detection.enabled_classes,
                    self.config.detection.disabled_classes
                )
                
                # Update mouse sensitivity (live)
                self.mouse.set_sensitivity(
                    dpi=self.config.mouse.dpi_value,
                    in_game=self.config.mouse.in_game_sens,
                    reference=self.config.mouse.reference_sens
                )
                self.mouse.normalization_enabled = self.config.mouse.sens_normalization_enabled
                self.mouse.sens_multiplier = self.config.mouse.sens_multiplier
                
                # Update detector confidence
                if self.detector:
                    self.detector.set_confidence(self.config.detection.confidence_threshold)
                
                # Update humanizers (live)
                self._configure_humanizer(self.aim_humanizer, self.config.humanizer)
                self._configure_humanizer(self.flick_humanizer, self.config.flick_humanizer)
                
                # Process aim
                if self._aim_key_pressed and self.config.aim.enabled:
                    self._process_aim(detections)
                else:
                    self.state.aim_active = False
                    self._aim_start_time = 0
                
                # Process flick
                if self._flick_key_pressed and self.config.flick.enabled:
                    self._process_flick(detections)
                else:
                    self.state.flick_active = False
                
                # Process trigger
                if self._trigger_key_pressed and self.config.trigger.enabled:
                    self._process_trigger(detections)
                else:
                    self.state.trigger_active = False
                
                # Update metrics
                self._update_metrics(start_time)
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(0.01)
    
    def _process_aim(self, detections: list):
        """Process aim assist"""
        # Reaction time delay
        if self._aim_start_time == 0:
            self._aim_start_time = time.perf_counter()
        
        elapsed_ms = (time.perf_counter() - self._aim_start_time) * 1000
        if elapsed_ms < self.config.aim.reaction_time:
            return
        
        # Max aim time check
        if self.config.aim.max_aim_time_enabled:
            if elapsed_ms > self.config.aim.max_aim_time * 1000:
                self.state.aim_active = False
                return
        
        # Select target
        target = self.selector.select_target(
            detections,
            aim_fov=self.config.aim.aim_fov
        )
        
        if target is None:
            self.state.aim_active = False
            self.state.current_target = None
            return
        
        self.state.aim_active = True
        self.state.current_target = target
        
        # Calculate delta
        delta_x, delta_y = self.selector.calculate_delta(target)
        
        # Apply humanization
        if self.config.humanizer.enabled:
            # Check for random pause
            pause = self.aim_humanizer.should_pause()
            if pause:
                time.sleep(pause / 1000)
                return
            
            # Check proximity pause
            prox_pause = self.aim_humanizer.should_proximity_pause(target.distance_to_crosshair)
            if prox_pause:
                time.sleep(prox_pause / 1000)
                return
            
            # Humanize movement
            h_dx, h_dy = self.aim_humanizer.humanize_movement(
                delta_x, delta_y,
                self.config.aim.smooth_x,
                self.config.aim.smooth_y
            )
        else:
            # Basic smoothing only
            h_dx = delta_x / max(1.0, self.config.aim.smooth_x)
            h_dy = delta_y / max(1.0, self.config.aim.smooth_y)
        
        # Apply tracking deadzone
        if abs(delta_x) < self.config.tracking.tracking_deadzone and \
           abs(delta_y) < self.config.tracking.tracking_deadzone:
            return
        
        # Move mouse
        self.mouse.move(h_dx, h_dy)
        
        # Delay
        if self.config.mouse.aim_delay > 0:
            time.sleep(self.config.mouse.aim_delay / 1000)
    
    def _process_flick(self, detections: list):
        """Process flick assist"""
        target = self.selector.select_target(
            detections,
            aim_fov=self.config.flick.flick_fov
        )
        
        if target is None:
            self.state.flick_active = False
            return
        
        self.state.flick_active = True
        
        # Calculate delta
        delta_x, delta_y = self.selector.calculate_delta(target)
        
        # Apply humanization if enabled
        if self.config.flick_humanizer.enabled:
            # Check for random pause
            pause = self.flick_humanizer.should_pause()
            if pause:
                time.sleep(pause / 1000)
                return
            
            # Humanize movement
            h_dx, h_dy = self.flick_humanizer.humanize_movement(
                delta_x, delta_y,
                self.config.flick.smooth_x,
                self.config.flick.smooth_y
            )
        else:
            # Basic smoothing only
            h_dx = delta_x / max(1.0, self.config.flick.smooth_x)
            h_dy = delta_y / max(1.0, self.config.flick.smooth_y)
        
        # Apply tracking deadzone
        if abs(delta_x) < self.config.tracking.tracking_deadzone and \
           abs(delta_y) < self.config.tracking.tracking_deadzone:
            return
        
        # Move mouse
        self.mouse.move(h_dx, h_dy)
        
        # Delay
        if self.config.mouse.flick_delay > 0:
            time.sleep(self.config.mouse.flick_delay / 1000)
    
    def _process_trigger(self, detections: list):
        """Process triggerbot"""
        target = self.selector.select_target_for_trigger(
            detections,
            trigger_scale=self.config.trigger.trigger_scale
        )
        
        if target is None:
            self.state.trigger_active = False
            self._last_trigger_time = 0  # Reset for next engagement
            return
        
        self.state.trigger_active = True
        
        now = time.perf_counter() * 1000  # Current time in ms
        
        # Determine delay based on whether this is first shot or follow-up
        if self._last_trigger_time == 0:
            # First shot - use first shot delay
            delay = random.uniform(
                self.config.trigger.first_shot_delay_min,
                self.config.trigger.first_shot_delay_max
            )
        else:
            # Subsequent shot - check if enough time has passed
            elapsed = now - self._last_trigger_time
            min_delay = random.uniform(
                self.config.trigger.multi_shot_delay_min,
                self.config.trigger.multi_shot_delay_max
            )
            if elapsed < min_delay:
                return  # Not enough time passed
            delay = 0  # Already waited
        
        if delay > 0:
            time.sleep(delay / 1000)
        
        # Click
        self.mouse.click("left")
        self._last_trigger_time = time.perf_counter() * 1000
    
    def _update_metrics(self, start_time: float):
        """Update performance metrics"""
        self._loop_count += 1
        now = time.perf_counter()
        
        if now - self._last_fps_time >= 1.0:
            self.state.loop_fps = self._loop_count / (now - self._last_fps_time)
            self._loop_count = 0
            self._last_fps_time = now
            
            # Get subsystem FPS
            if self.capture:
                self.state.capture_fps = self.capture.fps
            if self.detector:
                self.state.detection_fps = self.detector.fps
            
            # Callback
            if self.on_state_change:
                self.on_state_change(self.state)
        
        self.state.latency_ms = (now - start_time) * 1000
    
    # Input handling
    def set_aim_key_state(self, pressed: bool):
        """Set aim key state"""
        with self._lock:
            if pressed and not self._aim_key_pressed:
                self._aim_start_time = 0
                self.aim_humanizer.reset()
            self._aim_key_pressed = pressed
    
    def set_flick_key_state(self, pressed: bool):
        """Set flick key state"""
        with self._lock:
            if pressed and not self._flick_key_pressed:
                self.flick_humanizer.reset()
            self._flick_key_pressed = pressed
    
    def set_trigger_key_state(self, pressed: bool):
        """Set trigger key state"""
        with self._lock:
            self._trigger_key_pressed = pressed
    
    def reload_config(self):
        """Reload configuration"""
        self.config = get_config()
        
        # Update subsystems
        if self.selector:
            self.selector.set_fov(self.config.aim.aim_fov)
            self.selector.set_aim_bone(
                self.config.detection.aim_bone,
                self.config.detection.bone_scale
            )
        
        if self.detector:
            self.detector.set_confidence(self.config.detection.confidence_threshold)
        
        self.mouse.set_sensitivity(
            dpi=self.config.mouse.dpi_value,
            in_game=self.config.mouse.in_game_sens,
            reference=self.config.mouse.reference_sens
        )
        
        self._configure_humanizer(self.aim_humanizer, self.config.humanizer)
        self._configure_humanizer(self.flick_humanizer, self.config.flick_humanizer)
    
    @property
    def is_running(self) -> bool:
        """Check if assistant is running"""
        return self._running
    
    @property  
    def is_paused(self) -> bool:
        """Check if assistant is paused"""
        return self._paused
    
    def _check_input_state(self):
        """Check mouse button states using Windows API"""
        # Map config key names to virtual key codes
        key_map = {
            "left_click": VK_LBUTTON,
            "right_click": VK_RBUTTON,
            "middle_click": VK_MBUTTON,
            "forward_button": VK_XBUTTON2,
            "back_button": VK_XBUTTON1,
        }
        
        # Check aim key
        aim_vk = key_map.get(self.config.aim.aim_key, VK_XBUTTON2)
        aim_pressed = bool(GetAsyncKeyState(aim_vk) & 0x8000)
        
        if aim_pressed != self._aim_key_pressed:
            if aim_pressed and not self._aim_key_pressed:
                self._aim_start_time = 0
                self.aim_humanizer.reset()
            self._aim_key_pressed = aim_pressed
        
        # Check flick key
        flick_vk = key_map.get(self.config.flick.flick_key, VK_XBUTTON1)
        flick_pressed = bool(GetAsyncKeyState(flick_vk) & 0x8000)
        
        if flick_pressed != self._flick_key_pressed:
            if flick_pressed and not self._flick_key_pressed:
                self.flick_humanizer.reset()
            self._flick_key_pressed = flick_pressed
        
        # Check trigger key
        trigger_vk = key_map.get(self.config.trigger.trigger_key, VK_XBUTTON1)
        trigger_pressed = bool(GetAsyncKeyState(trigger_vk) & 0x8000)
        self._trigger_key_pressed = trigger_pressed
