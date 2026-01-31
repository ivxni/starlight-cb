"""
Configuration management for Starlight
Handles loading, saving, and accessing all settings
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path


@dataclass
class CaptureConfig:
    """Screen capture settings"""
    enabled: bool = True
    monitor: int = 0  # Primary monitor
    capture_width: int = 640
    capture_height: int = 640
    max_fps: int = 240
    
    # Debug display
    debug_enabled: bool = True
    debug_scale: float = 1.0
    debug_max_fps: int = 60


@dataclass
class DetectionConfig:
    """AI detection settings"""
    model_path: str = "models/model.onnx"
    confidence_threshold: float = 0.60
    use_tensorrt: bool = True
    trt_optimization_level: int = 3
    async_inference: bool = False
    
    # Aim bones/points selection
    aim_bone: str = "upper_head"  # top_head, upper_head, head, neck, chest, etc.
    bone_scale: float = 1.0
    
    # Class filtering
    enabled_classes: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    disabled_classes: List[int] = field(default_factory=list)


@dataclass
class AimConfig:
    """Aim assist settings"""
    enabled: bool = False
    aim_key: str = "forward_button"  # Mouse button
    aim_mode: str = "hold"  # hold, toggle
    
    # Targeting
    aim_fov: int = 25
    dynamic_fov_enabled: bool = False
    dynamic_fov_min: int = 15
    dynamic_fov_max: int = 40
    x_offset: int = 0
    y_offset: int = 0
    
    # Controls
    aim_type: str = "aim_v2"  # legacy, aim_v2
    reaction_time: int = 60  # ms
    max_aim_time_enabled: bool = False
    max_aim_time: float = 2.0  # seconds
    
    # Smoothing
    smooth_x: float = 40.0
    smooth_y: float = 80.0


@dataclass
class FlickConfig:
    """Flick settings"""
    enabled: bool = False
    flick_key: str = "back_button"
    
    # Targeting
    flick_fov: int = 30
    x_offset: int = 0
    y_offset: int = 0
    
    # Controls
    reaction_time: int = 60
    smooth_x: float = 200.0
    smooth_y: float = 200.0


@dataclass
class TriggerConfig:
    """Triggerbot settings"""
    enabled: bool = False
    trigger_key: str = "back_button"
    
    # Detection
    trigger_scale: int = 24
    
    # Timing
    first_shot_delay_min: int = 82
    first_shot_delay_max: int = 124
    multi_shot_delay_min: int = 40
    multi_shot_delay_max: int = 45


@dataclass
class WindMouseConfig:
    """WindMouse humanization algorithm"""
    enabled: bool = False
    min_gravity: float = 5.0
    max_gravity: float = 12.0
    min_wind: float = 5.0
    max_wind: float = 16.0
    min_speed: float = 15.0
    max_speed: float = 25.0
    min_damp: float = 1.0
    max_damp: float = 2.0


@dataclass
class HumanizerConfig:
    """Humanization settings for aim"""
    enabled: bool = False
    mode: str = "traditional"  # traditional, maku
    
    # WindMouse
    wind_mouse: WindMouseConfig = field(default_factory=WindMouseConfig)
    
    # Advanced features
    momentum_tracking: bool = False
    
    # Stop/Pause system
    stop_pause_enabled: bool = True
    stop_pause_chance: float = 0.002
    stop_pause_min: float = 1.0
    stop_pause_max: float = 120.0
    
    # Pattern masking
    pattern_masking_enabled: bool = False
    pattern_masking_intensity: float = 1.0
    pattern_masking_scale: float = 1.0
    
    # Sub-movement decomposition
    sub_movements_enabled: bool = False
    sub_movements_min_pause: float = 50.0
    sub_movements_max_pause: float = 150.0
    sub_movements_min_dist: float = 30.0
    sub_movements_chance: float = 0.35
    
    # Proximity pause
    proximity_pause_enabled: bool = False
    proximity_threshold: int = 15
    proximity_chance: float = 0.27
    proximity_min_pause: float = 50.0
    proximity_max_pause: float = 150.0
    proximity_cooldown: float = 300.0
    
    # Easing system
    ease_out: float = 80.0
    ease_in: float = 80.0
    ease_curve: float = 3.0
    
    # Momentum tracking system
    momentum_decay: float = 0.88
    momentum_lead_bias: float = 1.18
    momentum_deadzone: float = 4.0
    momentum_corr_prob: float = 0.52
    momentum_corr_str: float = 0.42
    
    # Entropy-aware humanization
    speed_variance_enabled: bool = True
    speed_variance_min: float = 0.46
    speed_variance_max: float = 0.90
    speed_variance_freq_min: float = 0.09
    speed_variance_freq_max: float = 0.30
    
    path_curvature_enabled: bool = True
    path_curvature_min: float = 1.1
    path_curvature_max: float = 3.0
    path_curvature_freq_min: float = 0.04
    path_curvature_freq_max: float = 0.28
    
    endpoint_settling_enabled: bool = False
    endpoint_settling_dist: float = 9.0
    endpoint_settling_intensity: float = 2.5


@dataclass  
class FlickHumanizerConfig:
    """Separate humanization settings for flick"""
    enabled: bool = False
    mode: str = "traditional"  # traditional, maku
    
    # WindMouse
    wind_mouse: WindMouseConfig = field(default_factory=WindMouseConfig)
    
    # Advanced features
    momentum_tracking: bool = False
    
    # Stop/Pause system
    stop_pause_enabled: bool = False
    stop_pause_chance: float = 0.002
    stop_pause_min: float = 1.0
    stop_pause_max: float = 120.0
    
    # Pattern masking
    pattern_masking_enabled: bool = False
    pattern_masking_intensity: float = 1.0
    pattern_masking_scale: float = 1.0
    
    # Sub-movement decomposition
    sub_movements_enabled: bool = False
    sub_movements_min_pause: float = 50.0
    sub_movements_max_pause: float = 150.0
    sub_movements_min_dist: float = 30.0
    sub_movements_chance: float = 0.35
    
    # Proximity pause
    proximity_pause_enabled: bool = False
    proximity_threshold: int = 15
    proximity_chance: float = 0.27
    proximity_min_pause: float = 50.0
    proximity_max_pause: float = 150.0
    proximity_cooldown: float = 300.0
    
    # Easing system
    ease_out: float = 80.0
    ease_in: float = 80.0
    ease_curve: float = 3.0
    
    # Momentum tracking system
    momentum_decay: float = 0.88
    momentum_lead_bias: float = 1.18
    momentum_deadzone: float = 4.0
    momentum_corr_prob: float = 0.52
    momentum_corr_str: float = 0.42
    
    # Entropy-aware humanization
    speed_variance_enabled: bool = False
    speed_variance_min: float = 0.46
    speed_variance_max: float = 0.90
    speed_variance_freq_min: float = 0.09
    speed_variance_freq_max: float = 0.30
    
    path_curvature_enabled: bool = False
    path_curvature_min: float = 1.1
    path_curvature_max: float = 3.0
    path_curvature_freq_min: float = 0.04
    path_curvature_freq_max: float = 0.28
    
    endpoint_settling_enabled: bool = False
    endpoint_settling_dist: float = 9.0
    endpoint_settling_intensity: float = 2.5


@dataclass
class MouseConfig:
    """Mouse/HID settings"""
    device: str = "internal"  # internal (SendInput)
    
    # Sensitivity normalization
    sens_normalization_enabled: bool = True
    dpi_value: int = 800
    in_game_sens: float = 0.41
    reference_sens: float = 0.70
    legacy_distance_smoothing: bool = False
    
    # Delays
    device_delay: int = 1
    aim_delay: int = 3
    flick_delay: int = 3
    sens_multiplier: float = 1.0


@dataclass
class TrackingConfig:
    """Basic tracking settings"""
    min_smoothing: float = 2.0
    tracking_deadzone: int = 1
    extra_smoothing: float = 1.0
    
    # Movement clamping
    movement_clamping_enabled: bool = False
    clamp_fov: int = 50
    clamp_max: int = 14


@dataclass
class FilteringConfig:
    """Detection filtering settings"""
    # Enemy color filter
    enemy_color_enabled: bool = True
    enemy_color: str = "purple"  # purple, pink, yellow, red
    
    # Color+Shape ignore filter
    color_shape_filter_enabled: bool = False
    color_shape_decay: int = 150


@dataclass
class Config:
    """Main configuration container"""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    aim: AimConfig = field(default_factory=AimConfig)
    flick: FlickConfig = field(default_factory=FlickConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    humanizer: HumanizerConfig = field(default_factory=HumanizerConfig)
    flick_humanizer: FlickHumanizerConfig = field(default_factory=FlickHumanizerConfig)
    mouse: MouseConfig = field(default_factory=MouseConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    
    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        """Load config from JSON file"""
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return cls._from_dict(data)
            except Exception as e:
                print(f"Error loading config: {e}")
                return cls()
        return cls()
    
    @classmethod
    def _from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary"""
        config = cls()
        
        # Load each section
        if 'capture' in data:
            config.capture = CaptureConfig(**data['capture'])
        if 'detection' in data:
            config.detection = DetectionConfig(**data['detection'])
        if 'aim' in data:
            config.aim = AimConfig(**data['aim'])
        if 'flick' in data:
            config.flick = FlickConfig(**data['flick'])
        if 'trigger' in data:
            config.trigger = TriggerConfig(**data['trigger'])
        if 'humanizer' in data:
            hum_data = data['humanizer']
            if 'wind_mouse' in hum_data:
                hum_data['wind_mouse'] = WindMouseConfig(**hum_data['wind_mouse'])
            config.humanizer = HumanizerConfig(**hum_data)
        if 'flick_humanizer' in data:
            fh_data = data['flick_humanizer']
            if 'wind_mouse' in fh_data:
                fh_data['wind_mouse'] = WindMouseConfig(**fh_data['wind_mouse'])
            config.flick_humanizer = FlickHumanizerConfig(**fh_data)
        if 'mouse' in data:
            config.mouse = MouseConfig(**data['mouse'])
        if 'tracking' in data:
            config.tracking = TrackingConfig(**data['tracking'])
        if 'filtering' in data:
            config.filtering = FilteringConfig(**data['filtering'])
            
        return config
    
    def save(self, path: str = "config.json"):
        """Save config to JSON file"""
        data = self._to_dict()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'capture': asdict(self.capture),
            'detection': asdict(self.detection),
            'aim': asdict(self.aim),
            'flick': asdict(self.flick),
            'trigger': asdict(self.trigger),
            'humanizer': asdict(self.humanizer),
            'flick_humanizer': asdict(self.flick_humanizer),
            'mouse': asdict(self.mouse),
            'tracking': asdict(self.tracking),
            'filtering': asdict(self.filtering),
        }


# Singleton instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config.load()
    return _config_instance


def save_config():
    """Save the global config instance"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.save()
