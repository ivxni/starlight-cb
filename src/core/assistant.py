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
from ..detection.detection import Detection
from ..detection.color_detector import ColorDetector
from ..detection.target_selector import TargetSelector, Target
from ..movement.mouse_controller import MouseController, get_mouse_controller, configure_mouse_controller
from ..movement.humanizer import Humanizer
from ..movement.pid_controller import PIDController
from ..input.input_blocker import InputBlocker, get_input_blocker
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
    flick_target: Optional[Target] = None
    aim_key_down: bool = False
    flick_key_down: bool = False
    trigger_key_down: bool = False
    lifecycle_state: str = "Idle"
    
    # Performance metrics
    capture_fps: float = 0.0
    detection_fps: float = 0.0
    loop_fps: float = 0.0
    latency_ms: float = 0.0
    frame_age_ms: float = 0.0
    detector_device: str = ""


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
        self.detector: Optional[ColorDetector] = None
        self.selector: Optional[TargetSelector] = None
        self.mouse: MouseController = get_mouse_controller()
        self.aim_humanizer: Humanizer = Humanizer()
        self.flick_humanizer: Humanizer = Humanizer()
        self.input_blocker: InputBlocker = get_input_blocker()
        self.pid_controller: PIDController = PIDController()
        
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
        self._warmup_frames_left = 0
        
        # Callbacks for UI updates
        self.on_state_change: Optional[Callable[[AssistantState], None]] = None
        self.on_detection: Optional[Callable[[list], None]] = None
    
    def initialize(self) -> bool:
        """Initialize all subsystems"""
        try:
            self.state.lifecycle_state = "Initializing"
            # Initialize screen capture
            self.capture = ScreenCapture(
                capture_width=self.config.capture.capture_width,
                capture_height=self.config.capture.capture_height,
                target_fps=self.config.capture.max_fps,
                monitor=self.config.capture.monitor,
                mode=self.config.capture.capture_mode,
                window_name=self.config.capture.window_name,
                obs_stream_url=getattr(self.config.capture, "obs_stream_url", ""),
                capture_buffer_size=getattr(self.config.capture, "capture_buffer_size", 1),
                max_frame_age_ms=getattr(self.config.capture, "max_frame_age_ms", 0)
            )
            
            # Initialize detector based on mode
            self._init_detector()
            
            # Initialize target selector
            self.selector = TargetSelector(
                frame_width=self.config.capture.capture_width,
                frame_height=self.config.capture.capture_height
            )
            
            # Configure mouse device (internal or arduino)
            self.mouse = configure_mouse_controller(
                device=self.config.mouse.device,
                port=self.config.mouse.arduino_port,
                jitter=self.config.mouse.arduino_jitter,
                tremor=self.config.mouse.arduino_tremor,
                humanization=self.config.mouse.arduino_humanization
            )
            
            # Configure mouse sensitivity
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
            
            # Configure input blocker (hide hotkeys from game)
            self._configure_input_blocker()
            
            print("Assistant initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize assistant: {e}")
            return False
    
    def _configure_humanizer(self, humanizer: Humanizer, config):
        """Configure a humanizer from config - ALL settings"""
        # Use the load_from_config method which handles all settings
        humanizer.load_from_config(config)
    
    def _configure_mouse_device(self):
        """Configure mouse device (internal or arduino)"""
        self.mouse = configure_mouse_controller(
            device=self.config.mouse.device,
            port=self.config.mouse.arduino_port,
            jitter=self.config.mouse.arduino_jitter,
            tremor=self.config.mouse.arduino_tremor,
            humanization=self.config.mouse.arduino_humanization
        )
        
        # Also update sensitivity
        self.mouse.set_sensitivity(
            dpi=self.config.mouse.dpi_value,
            in_game=self.config.mouse.in_game_sens,
            reference=self.config.mouse.reference_sens
        )
        self.mouse.normalization_enabled = self.config.mouse.sens_normalization_enabled
        self.mouse.sens_multiplier = self.config.mouse.sens_multiplier
        
        print(f"Mouse device: {self.mouse.device_mode}" + 
              (f" (Arduino connected)" if self.mouse.arduino_connected else ""))
    
    def _configure_input_blocker(self):
        """Configure input blocker based on config settings"""
        self.input_blocker.clear_blocked_buttons()
        
        if not self.config.mouse.input_blocking_enabled:
            print("Input blocking: DISABLED")
            return
        
        # Add blocked buttons based on config
        m = self.config.mouse
        
        if m.block_left_click:
            self.input_blocker.add_blocked_button("left_button")
        if m.block_right_click:
            self.input_blocker.add_blocked_button("right_button")
        if m.block_middle_click:
            self.input_blocker.add_blocked_button("middle_button")
        if m.block_forward_button:
            self.input_blocker.add_blocked_button("forward_button")
        if m.block_back_button:
            self.input_blocker.add_blocked_button("back_button")
        
        if self.input_blocker._blocked_buttons:
            print(f"Input blocking: ENABLED for {self.input_blocker._blocked_buttons}")
        else:
            print("Input blocking: ENABLED but no buttons selected")
    
    def _init_detector(self):
        """Initialize Color Detector"""
        self.detector = ColorDetector(config=self.config.detection)
        print(f"Color Detection - Preset: {self.config.detection.color_preset}")
        self.state.detector_device = getattr(self.detector, "device_status", "CPU")
    
    def reload_detector(self):
        """Reload detector (call when detection mode changes)"""
        self._init_detector()
        print("Detector reloaded")
    
    def start(self) -> bool:
        """Start the assistant"""
        if self._running:
            return True
        
        self.state.lifecycle_state = "Initializing"
        
        # Reconfigure mouse device (in case settings changed)
        self._configure_mouse_device()
        
        # Reconfigure input blocker
        self._configure_input_blocker()
        
        # Start capture
        if not self.capture.start():
            print("Failed to start screen capture")
            self.state.lifecycle_state = "Idle"
            return False
        
        # Start input blocker (if enabled)
        if self.config.mouse.input_blocking_enabled:
            self.input_blocker.start()
        
        # Start main loop
        self._running = True
        self._warmup_frames_left = max(0, int(getattr(self.config.capture, "startup_warmup_frames", 0)))
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        
        print("Assistant started")
        return True
    
    def stop(self):
        """Stop the assistant"""
        self.state.lifecycle_state = "Stopping"
        self._running = False
        
        if self._main_thread:
            self._main_thread.join(timeout=2.0)
            self._main_thread = None
        
        if self.capture:
            self.capture.stop()
        
        # Stop input blocker
        self.input_blocker.stop()
        
        self.reset_state()
        
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
                frame, frame_ts = self.capture.get_frame_with_timestamp()
                if frame is None or frame_ts == 0:
                    time.sleep(0.001)
                    continue
                
                # Drop stale frames (reduce perceived lag)
                frame_age_ms = (time.perf_counter() - frame_ts) * 1000
                self.state.frame_age_ms = frame_age_ms
                max_age = getattr(self.config.capture, "max_frame_age_ms", 0)
                if max_age and frame_age_ms > max_age:
                    continue
                
                # Warmup (ignore first frames to stabilize)
                if self._warmup_frames_left > 0:
                    self._warmup_frames_left -= 1
                    self.state.lifecycle_state = "Warmup"
                    self._update_metrics(start_time)
                    continue
                else:
                    self.state.lifecycle_state = "Running"
                
                # Run detection
                detections = []
                if self.detector and self.detector.is_loaded:
                    detections = self.detector.detect(frame)
                    # Refresh device status in case runtime selected a different provider
                    self.state.detector_device = getattr(self.detector, "device_status", self.state.detector_device)
                    
                    # Callback for visualization
                    if self.on_detection:
                        self.on_detection(detections)
                
                # Update selector and mouse config (live updates)
                self.selector.set_fov(self.config.aim.aim_fov)
                self.selector.set_aim_bone(
                    self.config.detection.aim_bone,
                    self.config.detection.bone_scale
                )
                # Class filtering removed - not needed for color detection
                
                # Update mouse sensitivity (live)
                self.mouse.set_sensitivity(
                    dpi=self.config.mouse.dpi_value,
                    in_game=self.config.mouse.in_game_sens,
                    reference=self.config.mouse.reference_sens
                )
                self.mouse.normalization_enabled = self.config.mouse.sens_normalization_enabled
                self.mouse.sens_multiplier = self.config.mouse.sens_multiplier
                
                # Update detector settings (live)
                if self.detector:
                    self.detector.update_hsv(
                        self.config.detection.color_h_min,
                        self.config.detection.color_h_max,
                        self.config.detection.color_s_min,
                        self.config.detection.color_s_max,
                        self.config.detection.color_v_min,
                        self.config.detection.color_v_max
                    )
                    self.detector.update_morphology(
                        self.config.detection.color_dilate,
                        self.config.detection.color_erode
                    )
                    self.detector.update_area_filter(
                        self.config.detection.color_min_area,
                        self.config.detection.color_max_area
                    )
                
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
        """Process aim assist with Sticky Aim and PID"""
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
            self.pid_controller.reset()
            return
        
        self.state.aim_active = True
        self.state.current_target = target
        
        # Calculate delta
        delta_x, delta_y = self.selector.calculate_delta(target)
        distance = target.distance_to_crosshair
        
        # Determine smoothing based on Sticky Aim (Close/Far)
        if self.config.aim.sticky_aim_enabled:
            sticky_zone = self.config.aim.sticky_aim_zone
            
            if distance <= sticky_zone:
                # Close to target - use close smoothing (more responsive)
                smooth_x = self.config.aim.smooth_x_close
                smooth_y = self.config.aim.smooth_y_close
            else:
                # Far from target - interpolate between close and far
                t = min(1.0, (distance - sticky_zone) / sticky_zone)
                smooth_x = self.config.aim.smooth_x_close + t * (self.config.aim.smooth_x_far - self.config.aim.smooth_x_close)
                smooth_y = self.config.aim.smooth_y_close + t * (self.config.aim.smooth_y_far - self.config.aim.smooth_y_close)
        else:
            # Use standard smoothing
            smooth_x = self.config.aim.smooth_x
            smooth_y = self.config.aim.smooth_y
        
        # Apply humanization
        if self.config.humanizer.enabled:
            # Check for random pause
            pause = self.aim_humanizer.should_pause()
            if pause:
                time.sleep(pause / 1000)
                return
            
            # Check proximity pause
            prox_pause = self.aim_humanizer.should_proximity_pause(distance)
            if prox_pause:
                time.sleep(prox_pause / 1000)
                return
            
            # Humanize movement
            h_dx, h_dy = self.aim_humanizer.humanize_movement(
                delta_x, delta_y,
                smooth_x,
                smooth_y
            )
        else:
            # Basic smoothing only
            h_dx = delta_x / max(1.0, smooth_x)
            h_dy = delta_y / max(1.0, smooth_y)
        
        # Apply PID stabilization
        if self.config.aim.pid_enabled:
            # Update PID gains from config
            self.pid_controller.set_gains(
                self.config.aim.pid_kp,
                self.config.aim.pid_ki,
                self.config.aim.pid_kd
            )
            self.pid_controller.activation_distance = self.config.aim.pid_activation_dist
            
            # Get PID correction
            pid_x, pid_y = self.pid_controller.update(delta_x, delta_y)
            h_dx += pid_x
            h_dy += pid_y
        
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
            self.state.flick_target = None
            return
        
        self.state.flick_active = True
        self.state.flick_target = target
        
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
        
        # Update FPS calculation every 100ms for smoother display
        if now - self._last_fps_time >= 0.1:
            elapsed = now - self._last_fps_time
            self.state.loop_fps = self._loop_count / elapsed
            self._loop_count = 0
            self._last_fps_time = now
            
            # Get subsystem FPS
            if self.capture:
                self.state.capture_fps = self.capture.fps
            if self.detector:
                self.state.detection_fps = self.detector.fps
            
            # Callback - fires every 100ms now
            if self.on_state_change:
                self.on_state_change(self.state)
        
        self.state.latency_ms = (now - start_time) * 1000

    def reset_state(self):
        """Reset internal state and counters"""
        self.state = AssistantState()
        self._paused = False
        self._aim_key_pressed = False
        self._flick_key_pressed = False
        self._trigger_key_pressed = False
        self._aim_start_time = 0.0
        self._last_trigger_time = 0.0
        self._loop_count = 0
        self._last_fps_time = 0.0
        self._warmup_frames_left = 0
        
        self.aim_humanizer.reset()
        self.flick_humanizer.reset()
        self.pid_controller.reset()
        
        if self.detector and hasattr(self.detector, "reset"):
            try:
                self.detector.reset()
            except Exception:
                pass
    
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
        """Check mouse button states - uses InputBlocker for blocked buttons"""
        # Map config key names to virtual key codes
        key_map = {
            "left_click": VK_LBUTTON,
            "right_click": VK_RBUTTON,
            "middle_click": VK_MBUTTON,
            "forward_button": VK_XBUTTON2,
            "back_button": VK_XBUTTON1,
        }
        
        # Helper to check button state (uses blocker if enabled)
        def is_button_pressed(button_name: str) -> bool:
            # Prefer hook state if available (works reliably for X buttons)
            if self.input_blocker.is_running and self.input_blocker.has_state(button_name):
                return self.input_blocker.is_button_pressed(button_name)
            # Otherwise use Windows API
            vk = key_map.get(button_name, VK_XBUTTON2)
            return bool(GetAsyncKeyState(vk) & 0x8000)
        
        # Check aim key
        aim_pressed = is_button_pressed(self.config.aim.aim_key)
        
        if aim_pressed != self._aim_key_pressed:
            if aim_pressed and not self._aim_key_pressed:
                self._aim_start_time = 0
                self.aim_humanizer.reset()
            self._aim_key_pressed = aim_pressed
        self.state.aim_key_down = self._aim_key_pressed
        
        # Check flick key
        flick_pressed = is_button_pressed(self.config.flick.flick_key)
        
        if flick_pressed != self._flick_key_pressed:
            if flick_pressed and not self._flick_key_pressed:
                self.flick_humanizer.reset()
            self._flick_key_pressed = flick_pressed
        self.state.flick_key_down = self._flick_key_pressed
        
        # Check trigger key
        trigger_pressed = is_button_pressed(self.config.trigger.trigger_key)
        self._trigger_key_pressed = trigger_pressed
        self.state.trigger_key_down = self._trigger_key_pressed
