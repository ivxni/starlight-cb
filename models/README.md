# AI Models Folder

Place your ONNX or TensorRT engine model files in this folder.

## Supported Formats

- `.onnx` - ONNX models (works with DirectML, CUDA, CPU)
- `.engine` - TensorRT engine files (NVIDIA GPUs only, fastest)
- `.xml` / `.bin` - OpenVINO models (Intel hardware)
- `.enc` - Encrypted ONNX models

## Model Requirements

Models should be YOLO-style object detection models that output:
- Bounding boxes: [x_center, y_center, width, height]
- Confidence scores
- Class predictions

Typical input size: 640x640 pixels

## Where to Get Models

1. **Train your own** - Use YOLOv8 or YOLOv5 with custom dataset
2. **Convert existing models** - Use ONNX export from PyTorch/TensorFlow
3. **Community models** - Search for pre-trained game detection models

## Model Conversion

### PyTorch to ONNX (YOLOv8)
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', opset=12)
```

### ONNX to TensorRT
```bash
# Requires TensorRT and CUDA installed
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

## Performance Tips

- **TensorRT** (.engine): Fastest on NVIDIA GPUs, requires model rebuild per GPU
- **ONNX + DirectML**: Universal, works on AMD/Intel/NVIDIA GPUs
- **ONNX + CUDA**: Fast on NVIDIA, no rebuild required
- **OpenVINO**: Best for Intel CPUs and integrated graphics
