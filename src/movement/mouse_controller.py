"""
Mouse Controller Module
Uses Windows SendInput API for hardware-level mouse simulation
"""

import ctypes
from ctypes import wintypes
import time
from typing import Tuple, Optional
from dataclasses import dataclass
import threading


# Windows API Constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
INPUT_HARDWARE = 2

# X button constants
XBUTTON1 = 0x0001  # Back button
XBUTTON2 = 0x0002  # Forward button


# Windows API Structures
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


@dataclass
class MouseState:
    """Current mouse button state"""
    left: bool = False
    right: bool = False
    middle: bool = False
    x1: bool = False  # Back button
    x2: bool = False  # Forward button


class MouseController:
    """
    Low-level mouse controller using Windows SendInput API
    Simulates hardware-level mouse input that appears to come from the physical device
    """
    
    def __init__(self):
        """Initialize mouse controller"""
        # Windows API
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32
        
        # Extra info pointer (required for some games)
        self._extra = ctypes.c_ulong(0)
        
        # Movement accumulator for sub-pixel precision
        self._accumulated_x: float = 0.0
        self._accumulated_y: float = 0.0
        
        # Rate limiting
        self._last_move_time: float = 0.0
        self._min_move_interval: float = 0.001  # 1ms minimum between moves
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # State tracking
        self._state = MouseState()
        
        # Sensitivity normalization
        self.sens_multiplier: float = 1.0
        self.dpi_value: int = 800
        self.in_game_sens: float = 0.41
        self.reference_sens: float = 0.70
        self.normalization_enabled: bool = True
    
    def _send_input(self, input_struct: INPUT) -> int:
        """Send a single input event"""
        return self._user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))
    
    def _create_mouse_input(self, dx: int = 0, dy: int = 0, flags: int = 0, 
                           mouse_data: int = 0) -> INPUT:
        """Create a mouse input structure"""
        mi = MOUSEINPUT(
            dx=dx,
            dy=dy,
            mouseData=mouse_data,
            dwFlags=flags,
            time=0,
            dwExtraInfo=ctypes.pointer(self._extra)
        )
        
        input_struct = INPUT()
        input_struct.type = INPUT_MOUSE
        input_struct.union.mi = mi
        
        return input_struct
    
    def move(self, dx: float, dy: float) -> bool:
        """
        Move mouse by relative amount (appears as hardware input)
        
        Args:
            dx: Horizontal movement in pixels (can be fractional)
            dy: Vertical movement in pixels (can be fractional)
            
        Returns:
            True if movement was sent
        """
        with self._lock:
            # Apply sensitivity normalization
            if self.normalization_enabled:
                multiplier = self._calculate_normalization()
                dx *= multiplier
                dy *= multiplier
            
            # Apply sens multiplier
            dx *= self.sens_multiplier
            dy *= self.sens_multiplier
            
            # Accumulate fractional movement for sub-pixel precision
            self._accumulated_x += dx
            self._accumulated_y += dy
            
            # Get integer movement
            move_x = int(self._accumulated_x)
            move_y = int(self._accumulated_y)
            
            # Keep fractional remainder
            self._accumulated_x -= move_x
            self._accumulated_y -= move_y
            
            # Skip if no integer movement
            if move_x == 0 and move_y == 0:
                return False
            
            # Rate limiting
            now = time.perf_counter()
            if now - self._last_move_time < self._min_move_interval:
                return False
            self._last_move_time = now
            
            # Send move
            input_struct = self._create_mouse_input(
                dx=move_x,
                dy=move_y,
                flags=MOUSEEVENTF_MOVE
            )
            
            result = self._send_input(input_struct)
            return result > 0
    
    def move_smooth(self, dx: float, dy: float, smooth_x: float, smooth_y: float) -> Tuple[float, float]:
        """
        Move mouse with smoothing applied
        
        Args:
            dx: Target horizontal movement
            dy: Target vertical movement
            smooth_x: Horizontal smoothing factor (higher = slower)
            smooth_y: Vertical smoothing factor (higher = slower)
            
        Returns:
            Tuple of (actual_dx, actual_dy) that was moved
        """
        # Apply smoothing (divide by smooth factor)
        smooth_x = max(1.0, smooth_x)
        smooth_y = max(1.0, smooth_y)
        
        actual_dx = dx / smooth_x
        actual_dy = dy / smooth_y
        
        self.move(actual_dx, actual_dy)
        
        return (actual_dx, actual_dy)
    
    def click(self, button: str = "left"):
        """
        Perform a mouse click
        
        Args:
            button: "left", "right", "middle", "x1" (back), "x2" (forward)
        """
        down_flag, up_flag, mouse_data = self._get_button_flags(button)
        
        # Press
        input_down = self._create_mouse_input(
            flags=down_flag,
            mouse_data=mouse_data
        )
        self._send_input(input_down)
        
        # Small delay
        time.sleep(0.01)
        
        # Release
        input_up = self._create_mouse_input(
            flags=up_flag,
            mouse_data=mouse_data
        )
        self._send_input(input_up)
    
    def press(self, button: str = "left"):
        """Press and hold a mouse button"""
        down_flag, _, mouse_data = self._get_button_flags(button)
        
        input_struct = self._create_mouse_input(
            flags=down_flag,
            mouse_data=mouse_data
        )
        self._send_input(input_struct)
        
        self._update_state(button, True)
    
    def release(self, button: str = "left"):
        """Release a mouse button"""
        _, up_flag, mouse_data = self._get_button_flags(button)
        
        input_struct = self._create_mouse_input(
            flags=up_flag,
            mouse_data=mouse_data
        )
        self._send_input(input_struct)
        
        self._update_state(button, False)
    
    def _get_button_flags(self, button: str) -> Tuple[int, int, int]:
        """Get Windows flags for a button"""
        buttons = {
            "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, 0),
            "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, 0),
            "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, 0),
            "x1": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON1),
            "x2": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON2),
            "back": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON1),
            "forward": (MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, XBUTTON2),
        }
        return buttons.get(button.lower(), buttons["left"])
    
    def _update_state(self, button: str, pressed: bool):
        """Update internal button state"""
        button = button.lower()
        if button == "left":
            self._state.left = pressed
        elif button == "right":
            self._state.right = pressed
        elif button == "middle":
            self._state.middle = pressed
        elif button in ("x1", "back"):
            self._state.x1 = pressed
        elif button in ("x2", "forward"):
            self._state.x2 = pressed
    
    def _calculate_normalization(self) -> float:
        """
        Calculate sensitivity normalization multiplier
        
        This ensures consistent aim feel regardless of DPI/sensitivity
        """
        if not self.normalization_enabled:
            return 1.0
        
        # Calculate eDPI (effective DPI)
        edpi = self.dpi_value * self.in_game_sens
        reference_edpi = self.dpi_value * self.reference_sens
        
        # Normalization ratio
        if edpi > 0:
            return reference_edpi / edpi
        return 1.0
    
    def set_sensitivity(self, dpi: int, in_game: float, reference: float):
        """Set sensitivity normalization values"""
        self.dpi_value = dpi
        self.in_game_sens = in_game
        self.reference_sens = reference
    
    def reset_accumulator(self):
        """Reset fractional movement accumulator"""
        with self._lock:
            self._accumulated_x = 0.0
            self._accumulated_y = 0.0
    
    @property
    def state(self) -> MouseState:
        """Get current mouse button state"""
        return self._state
    
    def is_button_pressed(self, button: str) -> bool:
        """Check if a button is currently pressed"""
        button = button.lower()
        if button == "left":
            return self._state.left
        elif button == "right":
            return self._state.right
        elif button == "middle":
            return self._state.middle
        elif button in ("x1", "back"):
            return self._state.x1
        elif button in ("x2", "forward"):
            return self._state.x2
        return False


# Global instance
_mouse_controller: Optional[MouseController] = None


def get_mouse_controller() -> MouseController:
    """Get the global mouse controller instance"""
    global _mouse_controller
    if _mouse_controller is None:
        _mouse_controller = MouseController()
    return _mouse_controller
