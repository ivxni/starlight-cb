"""
TensorRT / ONNX Detection Engine
Handles AI model inference for object detection
"""

import numpy as np
import cv2
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Try importing ONNX Runtime (more compatible)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print(f"ONNX Runtime loaded: {ort.__version__}, Providers: {ort.get_available_providers()}")
except Exception as e:
    ONNX_AVAILABLE = False
    print(f"Warning: onnxruntime not available - {type(e).__name__}: {e}")

# Try importing TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
    print(f"TensorRT loaded: {trt.__version__}")
except Exception as e:
    TRT_AVAILABLE = False
    print(f"Note: TensorRT not available ({type(e).__name__}), using ONNX Runtime")


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


class TensorRTEngine:
    """
    AI Detection Engine supporting both TensorRT and ONNX Runtime
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 use_tensorrt: bool = True, input_size: int = 640):
        """
        Initialize detection engine
        
        Args:
            model_path: Path to .onnx or .engine model file
            confidence_threshold: Minimum confidence for detections
            use_tensorrt: Prefer TensorRT over ONNX Runtime
            input_size: Model input size (assumes square)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        
        # Runtime
        self._session = None
        self._backend = None
        
        # TensorRT specific
        self._trt_engine = None
        self._trt_context = None
        self._trt_inputs = None
        self._trt_outputs = None
        self._trt_bindings = None
        self._trt_stream = None
        
        # Performance tracking
        self._inference_count: int = 0
        self._inference_fps: float = 0
        self._last_fps_time: float = 0
        self._last_inference_time: float = 0
        
        # Initialize
        self._load_model(use_tensorrt)
    
    def _load_model(self, use_tensorrt: bool):
        """Load the AI model"""
        suffix = self.model_path.suffix.lower()
        
        # Try TensorRT first if requested and available
        if use_tensorrt and TRT_AVAILABLE and suffix in ['.engine', '.trt']:
            if self._load_tensorrt():
                return
        
        # Try ONNX Runtime
        if ONNX_AVAILABLE and suffix in ['.onnx', '.enc']:
            if self._load_onnx():
                return
        
        # Try converting ONNX to TensorRT
        if use_tensorrt and TRT_AVAILABLE and suffix == '.onnx':
            print("TensorRT engine not found, would need to convert from ONNX")
            # For now, fall back to ONNX
            if self._load_onnx():
                return
        
        raise RuntimeError(f"Could not load model: {self.model_path}")
    
    def _load_onnx(self) -> bool:
        """Load model using ONNX Runtime"""
        try:
            # Configure providers
            providers = []
            
            # Try CUDA first
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }))
            
            # Try DirectML (Windows)
            if 'DmlExecutionProvider' in ort.get_available_providers():
                providers.append('DmlExecutionProvider')
            
            # CPU fallback
            providers.append('CPUExecutionProvider')
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )
            
            self._backend = "onnx"
            provider = self._session.get_providers()[0]
            print(f"Model loaded (ONNX Runtime - {provider}): {self.model_path}")
            return True
            
        except Exception as e:
            print(f"ONNX Runtime failed: {e}")
            return False
    
    def _load_tensorrt(self) -> bool:
        """Load model using TensorRT"""
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            
            with open(self.model_path, 'rb') as f:
                runtime = trt.Runtime(logger)
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())
            
            self._trt_context = self._trt_engine.create_execution_context()
            self._trt_stream = cuda.Stream()
            
            # Allocate buffers
            self._trt_inputs = []
            self._trt_outputs = []
            self._trt_bindings = []
            
            for i in range(self._trt_engine.num_bindings):
                dtype = trt.nptype(self._trt_engine.get_binding_dtype(i))
                shape = self._trt_engine.get_binding_shape(i)
                size = trt.volume(shape)
                
                # Allocate host and device memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self._trt_bindings.append(int(device_mem))
                
                if self._trt_engine.binding_is_input(i):
                    self._trt_inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                else:
                    self._trt_outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            
            self._backend = "tensorrt"
            print(f"Model loaded (TensorRT): {self.model_path}")
            return True
            
        except Exception as e:
            print(f"TensorRT failed: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a frame
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of Detection objects
        """
        if frame is None:
            return []
        
        start_time = time.perf_counter()
        
        # Preprocess
        input_tensor = self._preprocess(frame)
        
        # Run inference
        if self._backend == "onnx":
            outputs = self._infer_onnx(input_tensor)
        elif self._backend == "tensorrt":
            outputs = self._infer_tensorrt(input_tensor)
        else:
            return []
        
        # Postprocess
        detections = self._postprocess(outputs, frame.shape)
        
        # Update performance stats
        self._last_inference_time = time.perf_counter() - start_time
        self._update_fps()
        
        return detections
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input"""
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def _infer_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run ONNX Runtime inference"""
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})
        return outputs[0]
    
    def _infer_tensorrt(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TensorRT inference"""
        # Copy input to device
        np.copyto(self._trt_inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self._trt_inputs[0]['device'],
            self._trt_inputs[0]['host'],
            self._trt_stream
        )
        
        # Execute
        self._trt_context.execute_async_v2(
            bindings=self._trt_bindings,
            stream_handle=self._trt_stream.handle
        )
        
        # Copy output to host
        cuda.memcpy_dtoh_async(
            self._trt_outputs[0]['host'],
            self._trt_outputs[0]['device'],
            self._trt_stream
        )
        
        self._trt_stream.synchronize()
        
        return self._trt_outputs[0]['host'].reshape(self._trt_outputs[0]['shape'])
    
    def _postprocess(self, outputs: np.ndarray, frame_shape: Tuple[int, int, int]) -> List[Detection]:
        """
        Postprocess model outputs to Detection objects
        Handles YOLOv8 output format
        """
        detections = []
        
        frame_height, frame_width = frame_shape[:2]
        scale_x = frame_width / self.input_size
        scale_y = frame_height / self.input_size
        
        # YOLOv8 output: [batch, num_detections, 4 + num_classes] or transposed
        if outputs.ndim == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # Check if we need to transpose (YOLOv8 outputs [4+classes, num_boxes])
        if outputs.shape[0] < outputs.shape[1]:
            outputs = outputs.T
        
        for row in outputs:
            # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
            if len(row) < 5:
                continue
                
            x_center, y_center, w, h = row[:4]
            class_scores = row[4:]
            
            # Get best class
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            
            if confidence < self.confidence_threshold:
                continue
            
            # Convert to corner format and scale to frame size
            x1 = (x_center - w / 2) * scale_x
            y1 = (y_center - h / 2) * scale_y
            x2 = (x_center + w / 2) * scale_x
            y2 = (y_center + h / 2) * scale_y
            
            # Clamp to frame bounds
            x1 = max(0, min(x1, frame_width))
            y1 = max(0, min(y1, frame_height))
            x2 = max(0, min(x2, frame_width))
            y2 = max(0, min(y2, frame_height))
            
            detections.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(confidence),
                class_id=int(class_id)
            ))
        
        # Apply NMS
        detections = self._nms(detections, iou_threshold=0.45)
        
        return detections
    
    def _nms(self, detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        """Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            detections = [
                d for d in detections
                if self._iou(best, d) < iou_threshold
            ]
        
        return keep
    
    def _iou(self, a: Detection, b: Detection) -> float:
        """Calculate Intersection over Union"""
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = a.area + b.area - intersection
        
        return intersection / union if union > 0 else 0
    
    def _update_fps(self):
        """Update FPS counter"""
        self._inference_count += 1
        now = time.perf_counter()
        
        if now - self._last_fps_time >= 1.0:
            self._inference_fps = self._inference_count / (now - self._last_fps_time)
            self._inference_count = 0
            self._last_fps_time = now
    
    @property
    def fps(self) -> float:
        """Get current inference FPS"""
        return self._inference_fps
    
    @property
    def inference_time_ms(self) -> float:
        """Get last inference time in milliseconds"""
        return self._last_inference_time * 1000
    
    @property
    def backend(self) -> str:
        """Get current backend name"""
        return self._backend or "none"
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._session is not None or self._trt_engine is not None
    
    def set_confidence(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.01, min(0.99, threshold))
