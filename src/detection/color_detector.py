"""
Color Detection Module - Advanced
High-performance color-based detection with:
- Color presets (Valorant enemy colors)
- HSV and RGB modes
- Color differential filters
- Morphological operations
- Detection smoothing (Anti-Wobble)
- Gaussian blur pre-processing
- GPU acceleration via CuPy (CUDA)
"""

import numpy as np
import cv2
import time
from typing import List, Tuple, Optional
from collections import deque
from dataclasses import dataclass

from .detection import Detection

# Try to import CuPy for GPU acceleration
CUPY_AVAILABLE = False
CUPY_ERROR = ""
cp = None
try:
    import cupy as cp
    # Test if CUDA is actually available AND working
    device_count = cp.cuda.runtime.getDeviceCount()
    if device_count > 0:
        # Test actual GPU operation to verify CUDA Toolkit is present
        test_arr = cp.array([1, 2, 3])
        _ = cp.asnumpy(test_arr)
        CUPY_AVAILABLE = True
except Exception as e:
    cp = None
    CUPY_AVAILABLE = False
    CUPY_ERROR = str(e)
    # Check for common CUDA Toolkit missing error
    if "nvrtc" in str(e).lower() or "dll" in str(e).lower():
        CUPY_ERROR = "CUDA Toolkit not installed (only driver present)"


# Valorant Enemy Color Presets (HSV values)
COLOR_PRESETS = {
    # Purple outlines (default enemy)
    "purple": {
        "space": "hsv",
        "h_min": 125, "h_max": 155,
        "s_min": 80, "s_max": 255,
        "v_min": 80, "v_max": 255,
    },
    "purple2": {
        "space": "hsv",
        "h_min": 130, "h_max": 160,
        "s_min": 100, "s_max": 255,
        "v_min": 100, "v_max": 255,
    },
    "purple3": {
        "space": "hsv",
        "h_min": 120, "h_max": 150,
        "s_min": 60, "s_max": 255,
        "v_min": 60, "v_max": 255,
    },
    # Yellow outlines (tagged/revealed enemies)
    "yellow": {
        "space": "hsv",
        "h_min": 20, "h_max": 35,
        "s_min": 150, "s_max": 255,
        "v_min": 180, "v_max": 255,
    },
    "yellow2": {
        "space": "hsv",
        "h_min": 15, "h_max": 40,
        "s_min": 120, "s_max": 255,
        "v_min": 150, "v_max": 255,
    },
    # Red outlines (Cypher cam, etc)
    "red": {
        "space": "hsv",
        "h_min": 0, "h_max": 10,
        "s_min": 150, "s_max": 255,
        "v_min": 150, "v_max": 255,
    },
    "red2": {
        "space": "hsv",
        "h_min": 170, "h_max": 180,  # Red wraps around
        "s_min": 150, "s_max": 255,
        "v_min": 150, "v_max": 255,
    },
}


@dataclass
class SmoothedDetection:
    """Detection with smoothing history"""
    x1: float
    y1: float
    x2: float
    y2: float
    frames_seen: int = 1


class ColorDetector:
    """
    Advanced color-based detection engine
    Provides same interface as TensorRTEngine for compatibility
    """
    
    def __init__(self, config=None):
        """
        Initialize color detector from config or defaults
        """
        # Default values
        self.preset = "purple2"
        self.color_space = "hsv"
        
        # HSV range
        self.h_min = 130
        self.h_max = 160
        self.s_min = 100
        self.s_max = 255
        self.v_min = 100
        self.v_max = 255
        
        # RGB range
        self.r_min = 162
        self.r_max = 214
        self.g_min = 0
        self.g_max = 104
        self.b_min = 205
        self.b_max = 255
        
        # Color differentials (RGB mode)
        self.rg_diff = 82
        self.bg_diff = -255
        self.rb_max_diff = 255
        
        # Morphology
        self.dilate_iter = 2
        self.erode_iter = 1
        self.closing_enabled = False
        self.closing_size = 2
        
        # Blur
        self.blur_enabled = True
        self.blur_size = 3
        
        # Area filtering
        self.min_area = 50
        self.max_area = 50000
        
        # Smoothing
        self.smoothing_enabled = True
        self.smoothing_window = 5
        self.outlier_threshold = 50
        self.bbox_smoothing = 0.95
        
        # Head offset
        self.head_offset = 0.15
        
        # Confidence threshold
        self._confidence_threshold = 0.5

        # Device selection
        self.use_gpu = False
        self._gpu_available = False
        self._gpu_backend = "none"  # "cupy", "opencv_cuda", or "none"
        self._gpu_reason = "Not checked"
        self._cupy_failed_logged = False  # Prevent error spam
        
        # Check GPU availability - prefer CuPy, fallback to OpenCV CUDA
        if CUPY_AVAILABLE:
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count > 0:
                    self._gpu_available = True
                    self._gpu_backend = "cupy"
                    device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                    self._gpu_reason = f"CuPy CUDA ({device_name})"
            except Exception as e:
                self._gpu_reason = f"CuPy error: {e}"
        elif CUPY_ERROR:
            self._gpu_reason = CUPY_ERROR
        
        # Fallback: Check OpenCV CUDA
        if not self._gpu_available:
            try:
                if hasattr(cv2, "cuda"):
                    device_count = cv2.cuda.getCudaEnabledDeviceCount()
                    if device_count > 0:
                        self._gpu_available = True
                        self._gpu_backend = "opencv_cuda"
                        self._gpu_reason = f"OpenCV CUDA ({device_count} device(s))"
                    elif not CUPY_ERROR:
                        self._gpu_reason = "No CUDA devices found"
                elif not CUPY_ERROR:
                    self._gpu_reason = "No GPU backend available"
            except Exception as e:
                if not CUPY_ERROR:
                    self._gpu_reason = f"CUDA check error: {e}"
        
        # Performance tracking
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._inference_time = 0.0
        
        # Morphological kernels
        self._kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Detection history for smoothing
        self._detection_history: deque = deque(maxlen=10)
        self._smoothed_detections: List[SmoothedDetection] = []
        
        # Load from config if provided
        if config:
            self.load_from_config(config)
        
        self._is_loaded = True
        print(f"ColorDetector initialized - Preset: {self.preset}")
    
    def load_from_config(self, config):
        """Load settings from DetectionConfig"""
        requested_gpu = getattr(config, "color_device", "cpu") == "gpu"
        
        if requested_gpu:
            if self._gpu_available:
                self.use_gpu = True
                self._cupy_failed_logged = False  # Reset error flag
                print(f"ColorDetector: GPU active - {self._gpu_reason}")
            else:
                self.use_gpu = False
                print(f"ColorDetector: GPU not available - using CPU")
                print(f"  Reason: {self._gpu_reason}")
                if "toolkit" in self._gpu_reason.lower():
                    print("  Fix: Install CUDA Toolkit 12.x from nvidia.com/cuda-downloads")
        else:
            self.use_gpu = False
            print("ColorDetector: CPU mode (fast enough for color detection)")
        self.preset = config.color_preset
        self.color_space = config.color_space
        
        # Apply preset if not custom
        if self.preset != "custom" and self.preset in COLOR_PRESETS:
            preset = COLOR_PRESETS[self.preset]
            self.color_space = preset["space"]
            self.h_min = preset["h_min"]
            self.h_max = preset["h_max"]
            self.s_min = preset["s_min"]
            self.s_max = preset["s_max"]
            self.v_min = preset["v_min"]
            self.v_max = preset["v_max"]
        else:
            # Custom values
            self.h_min = config.color_h_min
            self.h_max = config.color_h_max
            self.s_min = config.color_s_min
            self.s_max = config.color_s_max
            self.v_min = config.color_v_min
            self.v_max = config.color_v_max
        
        # RGB settings
        self.r_min = config.color_r_min
        self.r_max = config.color_r_max
        self.g_min = config.color_g_min
        self.g_max = config.color_g_max
        self.b_min = config.color_b_min
        self.b_max = config.color_b_max
        
        self.rg_diff = config.color_rg_diff
        self.bg_diff = config.color_bg_diff
        self.rb_max_diff = config.color_rb_max_diff
        
        # Morphology
        self.dilate_iter = config.color_dilate
        self.erode_iter = config.color_erode
        self.closing_enabled = config.color_closing
        self.closing_size = config.color_closing_size
        
        # Blur
        self.blur_enabled = config.color_blur_enabled
        self.blur_size = config.color_blur_size
        
        # Area
        self.min_area = config.color_min_area
        self.max_area = config.color_max_area
        
        # Smoothing
        self.smoothing_enabled = config.smoothing_enabled
        self.smoothing_window = config.smoothing_window
        self.outlier_threshold = config.smoothing_outlier
        self.bbox_smoothing = config.bbox_smoothing
        
        # Head offset
        self.head_offset = config.color_head_offset
        
        # Update kernels
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.closing_size, self.closing_size)
        )
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect colored regions in frame
        
        Args:
            frame: BGR numpy array
            
        Returns:
            List of Detection objects
        """
        start_time = time.perf_counter()
        
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Pre-processing and mask creation
            if self.use_gpu and self._gpu_available:
                if self._gpu_backend == "cupy":
                    mask = self._create_mask_cupy(frame)
                elif self._gpu_backend == "opencv_cuda":
                    mask = self._create_hsv_mask_gpu(frame)
                else:
                    mask = self._create_mask_cpu(frame)
            else:
                mask = self._create_mask_cpu(frame)
            
            # Morphological operations (always CPU - fast enough)
            mask = self._apply_morphology(mask)
            
            # Find contours and create detections
            raw_detections = self._find_detections(mask)
            
            # Apply smoothing if enabled
            if self.smoothing_enabled and raw_detections:
                detections = self._smooth_detections(raw_detections)
            else:
                detections = raw_detections
            
        except Exception as e:
            print(f"ColorDetector error: {e}")
            detections = []
        
        # Update performance metrics
        self._inference_time = (time.perf_counter() - start_time) * 1000
        self._update_fps()
        
        return detections
    
    def _create_mask_cpu(self, frame: np.ndarray) -> np.ndarray:
        """Create mask using CPU (NumPy/OpenCV)"""
        if self.blur_enabled and self.blur_size > 0:
            blur_k = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
            frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
        
        if self.color_space == "rgb":
            return self._create_rgb_mask(frame)
        else:
            return self._create_hsv_mask(frame)
    
    def _create_mask_cupy(self, frame: np.ndarray) -> np.ndarray:
        """Create mask using CuPy (CUDA GPU)"""
        if not CUPY_AVAILABLE or cp is None:
            # CuPy not available, silently fall back
            self.use_gpu = False
            self._gpu_backend = "none"
            return self._create_mask_cpu(frame)
        
        try:
            # Upload frame to GPU
            frame_gpu = cp.asarray(frame)
            
            # Gaussian blur on GPU (if enabled)
            if self.blur_enabled and self.blur_size > 0:
                # CuPy doesn't have GaussianBlur, do on CPU before upload
                blur_k = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
                frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
                frame_gpu = cp.asarray(frame)
            
            if self.color_space == "rgb":
                # RGB mask on GPU
                b, g, r = frame_gpu[:, :, 0], frame_gpu[:, :, 1], frame_gpu[:, :, 2]
                
                r_mask = (r >= self.r_min) & (r <= self.r_max)
                g_mask = (g >= self.g_min) & (g <= self.g_max)
                b_mask = (b >= self.b_min) & (b <= self.b_max)
                
                r_int = r.astype(cp.int16)
                g_int = g.astype(cp.int16)
                b_int = b.astype(cp.int16)
                
                rg_diff_mask = (r_int - g_int) >= self.rg_diff
                bg_diff_mask = (b_int - g_int) >= self.bg_diff
                rb_diff_mask = cp.abs(r_int - b_int) <= self.rb_max_diff
                
                combined = r_mask & g_mask & b_mask & rg_diff_mask & bg_diff_mask & rb_diff_mask
                mask_gpu = combined.astype(cp.uint8) * 255
            else:
                # HSV mask on GPU
                frame_float = frame_gpu.astype(cp.float32) / 255.0
                b, g, r = frame_float[:, :, 0], frame_float[:, :, 1], frame_float[:, :, 2]
                
                v = cp.maximum(cp.maximum(r, g), b)
                c = v - cp.minimum(cp.minimum(r, g), b)
                s = cp.where(v > 0, c / v, 0)
                
                h = cp.zeros_like(v)
                mask_r = (v == r) & (c > 0)
                mask_g = (v == g) & (c > 0)
                mask_b = (v == b) & (c > 0)
                
                h = cp.where(mask_r, 60 * ((g - b) / (c + 1e-10)) % 360, h)
                h = cp.where(mask_g, 60 * ((b - r) / (c + 1e-10)) + 120, h)
                h = cp.where(mask_b, 60 * ((r - g) / (c + 1e-10)) + 240, h)
                h = h / 2
                
                h = h.astype(cp.uint8)
                s = (s * 255).astype(cp.uint8)
                v = (v * 255).astype(cp.uint8)
                
                h_mask = (h >= self.h_min) & (h <= self.h_max)
                s_mask = (s >= self.s_min) & (s <= self.s_max)
                v_mask = (v >= self.v_min) & (v <= self.v_max)
                
                mask_gpu = (h_mask & s_mask & v_mask).astype(cp.uint8) * 255
            
            mask = cp.asnumpy(mask_gpu)
            return mask
            
        except Exception as e:
            # Log error only once, then fall back silently
            if not self._cupy_failed_logged:
                print(f"CuPy GPU error (falling back to CPU): {e}")
                self._cupy_failed_logged = True
            self.use_gpu = False
            self._gpu_backend = "none"
            return self._create_mask_cpu(frame)

    def _create_hsv_mask_gpu(self, frame: np.ndarray) -> np.ndarray:
        """Create HSV mask using GPU (if available)."""
        try:
            blur_k = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
            gpu = cv2.cuda_GpuMat()
            gpu.upload(frame)
            if self.blur_enabled and blur_k > 1:
                gpu = cv2.cuda.GaussianBlur(gpu, (blur_k, blur_k), 0)
            hsv_gpu = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2HSV)
            lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
            upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
            mask_gpu = cv2.cuda.inRange(hsv_gpu, lower, upper)
            mask = mask_gpu.download()
            return mask
        except Exception:
            # Fallback to CPU if any GPU op fails
            if self.blur_enabled and self.blur_size > 0:
                blur_k = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
                frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 0)
            return self._create_hsv_mask(frame)
    
    def _create_hsv_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create mask using HSV color space"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower = np.array([self.h_min, self.s_min, self.v_min], dtype=np.uint8)
        upper = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Handle red color wrapping (hue 0 and 180)
        if self.preset in ["red", "red2"] or (self.h_min > 150 or self.h_max < 20):
            # Red wraps around, need second mask
            if self.h_max < self.h_min:
                lower2 = np.array([0, self.s_min, self.v_min], dtype=np.uint8)
                upper2 = np.array([self.h_max, self.s_max, self.v_max], dtype=np.uint8)
                mask2 = cv2.inRange(hsv, lower2, upper2)
                mask = cv2.bitwise_or(mask, mask2)
        
        return mask
    
    def _create_rgb_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create mask using RGB color space with differential filters"""
        # Split channels (BGR format)
        b, g, r = cv2.split(frame)
        
        # Basic RGB range
        r_mask = (r >= self.r_min) & (r <= self.r_max)
        g_mask = (g >= self.g_min) & (g <= self.g_max)
        b_mask = (b >= self.b_min) & (b <= self.b_max)
        
        # Color differential filters
        r_int = r.astype(np.int16)
        g_int = g.astype(np.int16)
        b_int = b.astype(np.int16)
        
        rg_diff_mask = (r_int - g_int) >= self.rg_diff
        bg_diff_mask = (b_int - g_int) >= self.bg_diff
        rb_diff_mask = np.abs(r_int - b_int) <= self.rb_max_diff
        
        # Combine all masks
        combined = r_mask & g_mask & b_mask & rg_diff_mask & bg_diff_mask & rb_diff_mask
        
        return combined.astype(np.uint8) * 255
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations - connect body parts of same enemy"""
        # Dilation - expand bright areas slightly
        if self.dilate_iter > 0:
            mask = cv2.dilate(mask, self._kernel_dilate, iterations=self.dilate_iter)
        
        # ALWAYS apply closing to connect head/arms/body of SAME enemy
        # Use vertical kernel - enemies are separated horizontally, body parts vertically
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Small horizontal closing to connect arm fragments
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small)
        
        # Additional closing if enabled in config
        if self.closing_enabled:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        
        # Erosion - remove noise
        if self.erode_iter > 0:
            mask = cv2.erode(mask, self._kernel_erode, iterations=self.erode_iter)
        
        return mask
    
    def _find_detections(self, mask: np.ndarray) -> List[Detection]:
        """Find contours and create Detection objects with NMS"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by aspect ratio
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio < 0.3:  # Too wide
                continue
            
            # Calculate confidence based on fill ratio
            fill_ratio = area / (w * h) if w * h > 0 else 0
            confidence = min(1.0, fill_ratio * 1.5)
            
            if confidence < self._confidence_threshold:
                continue
            
            detection = Detection(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                confidence=confidence,
                class_id=0
            )
            raw_detections.append(detection)
        
        # Apply Non-Maximum Suppression - only merge OVERLAPPING boxes, not nearby ones
        # Higher threshold = less merging = separate enemies stay separate
        # Light NMS - only merge very overlapping boxes (IoU > 0.7)
        # Keep separate detections (head, body parts) for better targeting
        detections = self._apply_nms(raw_detections, iou_threshold=0.7)
        
        # Sort by area (largest first)
        detections.sort(key=lambda d: d.area, reverse=True)
        
        return detections
    
    def _apply_nms(self, detections: List[Detection], iou_threshold: float = 0.3) -> List[Detection]:
        """
        Smart NMS - merge body parts of same enemy, keep different enemies separate
        """
        if not detections:
            return []
        
        # Sort by Y position (top to bottom) then by area
        detections = sorted(detections, key=lambda d: (d.y1, -d.area))
        
        # Group detections that belong to same enemy (overlapping X range)
        groups = []
        used = [False] * len(detections)
        
        for i, det in enumerate(detections):
            if used[i]:
                continue
            
            # Start a new group
            group = [det]
            used[i] = True
            
            # Find all detections that overlap horizontally (same enemy)
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                
                other = detections[j]
                
                # Check horizontal overlap - if X ranges overlap significantly, same enemy
                x_overlap = min(det.x2, other.x2) - max(det.x1, other.x1)
                min_width = min(det.width, other.width)
                
                # If more than 30% horizontal overlap, probably same enemy
                if x_overlap > min_width * 0.3:
                    # Also check if they're not too far apart vertically
                    y_gap = max(0, other.y1 - det.y2)  # Gap between boxes
                    if y_gap < max(det.height, other.height) * 1.5:
                        group.append(other)
                        used[j] = True
            
            groups.append(group)
        
        # Merge each group into one detection
        result = []
        for group in groups:
            if len(group) == 1:
                result.append(group[0])
            else:
                # Merge all boxes in group
                x1 = min(d.x1 for d in group)
                y1 = min(d.y1 for d in group)
                x2 = max(d.x2 for d in group)
                y2 = max(d.y2 for d in group)
                conf = max(d.confidence for d in group)
                
                merged = Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    class_id=0
                )
                result.append(merged)
        
        return result
    
    def _calculate_iou(self, det1: Detection, det2: Detection) -> float:
        """Calculate Intersection over Union of two detections"""
        x1 = max(det1.x1, det2.x1)
        y1 = max(det1.y1, det2.y1)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = det1.area + det2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_detections(self, detections: List[Detection]) -> Detection:
        """Merge multiple detections into one bounding box"""
        # Take the bounding box that contains all detections
        x1 = min(d.x1 for d in detections)
        y1 = min(d.y1 for d in detections)
        x2 = max(d.x2 for d in detections)
        y2 = max(d.y2 for d in detections)
        
        # Average confidence
        avg_conf = sum(d.confidence for d in detections) / len(detections)
        
        return Detection(
            x1=x1, y1=y1, x2=x2, y2=y2,
            confidence=avg_conf,
            class_id=0
        )
    
    def _smooth_detections(self, detections: List[Detection]) -> List[Detection]:
        """Apply detection smoothing (Anti-Wobble)"""
        # Add to history
        self._detection_history.append(detections)
        
        if len(self._detection_history) < 2:
            return detections
        
        smoothed = []
        for det in detections:
            # Find matching detection in history
            matched_boxes = []
            
            for hist_dets in self._detection_history:
                for hist_det in hist_dets:
                    # Check if same target (IoU or distance based)
                    dist = np.sqrt(
                        (det.center_x - hist_det.center_x) ** 2 +
                        (det.center_y - hist_det.center_y) ** 2
                    )
                    if dist < self.outlier_threshold:
                        matched_boxes.append(hist_det)
            
            if not matched_boxes:
                smoothed.append(det)
                continue
            
            # EMA smoothing of bounding box
            alpha = self.bbox_smoothing
            avg_x1 = sum(d.x1 for d in matched_boxes) / len(matched_boxes)
            avg_y1 = sum(d.y1 for d in matched_boxes) / len(matched_boxes)
            avg_x2 = sum(d.x2 for d in matched_boxes) / len(matched_boxes)
            avg_y2 = sum(d.y2 for d in matched_boxes) / len(matched_boxes)
            
            smooth_x1 = alpha * det.x1 + (1 - alpha) * avg_x1
            smooth_y1 = alpha * det.y1 + (1 - alpha) * avg_y1
            smooth_x2 = alpha * det.x2 + (1 - alpha) * avg_x2
            smooth_y2 = alpha * det.y2 + (1 - alpha) * avg_y2
            
            smoothed_det = Detection(
                x1=smooth_x1,
                y1=smooth_y1,
                x2=smooth_x2,
                y2=smooth_y2,
                confidence=det.confidence,
                class_id=det.class_id
            )
            smoothed.append(smoothed_det)
        
        return smoothed
    
    def _update_fps(self):
        """Update FPS counter"""
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = now

    def reset(self):
        """Reset internal history and performance counters"""
        self._detection_history.clear()
        self._smoothed_detections = []
        self._fps = 0.0
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._inference_time = 0.0
    
    def set_confidence(self, threshold: float):
        """Set confidence threshold"""
        self._confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_preset(self, preset: str):
        """Set color preset"""
        if preset in COLOR_PRESETS:
            self.preset = preset
            p = COLOR_PRESETS[preset]
            self.color_space = p["space"]
            self.h_min = p["h_min"]
            self.h_max = p["h_max"]
            self.s_min = p["s_min"]
            self.s_max = p["s_max"]
            self.v_min = p["v_min"]
            self.v_max = p["v_max"]
            print(f"ColorDetector: Preset changed to {preset}")
    
    def update_hsv(self, h_min: int, h_max: int, s_min: int, s_max: int, v_min: int, v_max: int):
        """Update HSV range"""
        self.h_min = h_min
        self.h_max = h_max
        self.s_min = s_min
        self.s_max = s_max
        self.v_min = v_min
        self.v_max = v_max
        self.preset = "custom"
    
    def update_rgb(self, r_min: int, r_max: int, g_min: int, g_max: int, b_min: int, b_max: int):
        """Update RGB range"""
        self.r_min = r_min
        self.r_max = r_max
        self.g_min = g_min
        self.g_max = g_max
        self.b_min = b_min
        self.b_max = b_max
    
    def update_morphology(self, dilate: int, erode: int):
        """Update morphological operation iterations"""
        self.dilate_iter = dilate
        self.erode_iter = erode
    
    def update_area_filter(self, min_area: int, max_area: int):
        """Update area filtering"""
        self.min_area = min_area
        self.max_area = max_area
    
    @property
    def fps(self) -> float:
        """Get current detection FPS"""
        return self._fps
    
    @property
    def inference_time(self) -> float:
        """Get last inference time in ms"""
        return self._inference_time

    @property
    def inference_time_ms(self) -> float:
        """Compatibility alias (MainWindow expects this name)."""
        return self._inference_time
    
    @property
    def is_loaded(self) -> bool:
        """Check if detector is ready"""
        return self._is_loaded
    
    @property
    def input_size(self) -> int:
        """Input size (for compatibility)"""
        return 0

    @property
    def device_status(self) -> str:
        """Human-readable active device."""
        if self.use_gpu and self._gpu_available:
            if self._gpu_backend == "cupy":
                return "GPU (CuPy)"
            elif self._gpu_backend == "opencv_cuda":
                return "GPU (OpenCV)"
            return "GPU"
        else:
            return "CPU"
