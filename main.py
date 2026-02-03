"""
Starlight - AI Assistive Input Tool
Main entry point
"""

import sys
import os
import warnings

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set CUDA path if not already set (for CuPy)
if not os.environ.get("CUDA_PATH"):
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
    if os.path.exists(cuda_path):
        os.environ["CUDA_PATH"] = cuda_path
        # Also add bin to PATH for DLL loading
        bin_path = os.path.join(cuda_path, "bin")
        if bin_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

# Suppress CuPy CUDA path warning (non-critical)
warnings.filterwarnings("ignore", message=".*CUDA path could not be detected.*")

# FFmpeg options for OpenCV VideoCapture - reduce buffer and make overruns non-fatal
# This helps with UDP streams without needing URL parameters
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.core.config import Config, get_config, save_config
from src.core.assistant import Assistant
from src.core.cleanup import full_cleanup


def main():
    """Main entry point"""
    # Reduce Qt console spam for a known harmless warning seen on some setups.
    # (Qt internally may set point size to -1 when a pixel-size font is used.)
    try:
        from PyQt6.QtCore import qInstallMessageHandler

        def _qt_message_handler(mode, context, message):
            if message and "QFont::setPointSize" in message:
                return
            # fall back to default behavior
            print(message)

        qInstallMessageHandler(_qt_message_handler)
    except Exception:
        pass

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Starlight")
    app.setApplicationVersion("1.0.0")
    # Ensure a valid default point size (prevents QFont::setPointSize(-1) spam on some systems)
    app.setFont(QFont("Segoe UI", 10))
    
    # Cleanup zombie processes from previous runs
    full_cleanup()
    
    # Load config
    config = get_config()
    
    # Import and create main window
    from src.ui.main_window import MainWindow
    window = MainWindow(config)
    window.show()
    
    # Run application
    exit_code = app.exec()
    
    # Save config on exit
    save_config()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
