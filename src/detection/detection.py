"""
Detection dataclass - shared between AI and Color detection
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Detection:
    """Single detection result"""
    x1: float  # Left
    y1: float  # Top
    x2: float  # Right
    y2: float  # Bottom
    confidence: float
    class_id: int
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.center_x, self.center_y)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def get_aim_point(self, bone: str = "upper_head", bone_scale: float = 1.0) -> Tuple[float, float]:
        """
        Get aim point based on bone selection
        
        Args:
            bone: Target bone (top_head, upper_head, head, neck, upper_chest, chest, etc.)
            bone_scale: Scale factor for bone position
            
        Returns:
            (x, y) coordinates for aim point
        """
        cx = self.center_x
        h = self.height
        
        # Bone offsets from top of bounding box (as percentage of height)
        bone_offsets = {
            "top_head": 0.05,
            "upper_head": 0.10,
            "head": 0.15,
            "neck": 0.20,
            "upper_chest": 0.30,
            "chest": 0.40,
            "lower_chest": 0.50,
            "upper_stomach": 0.55,
            "stomach": 0.60,
            "lower_stomach": 0.65,
            "pelvis": 0.75,
        }
        
        offset = bone_offsets.get(bone, 0.15)  # Default to head
        y = self.y1 + (h * offset * bone_scale)
        
        return (cx, y)
