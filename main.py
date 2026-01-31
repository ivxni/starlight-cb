"""
Starlight - AI Assistive Input Tool
Main entry point
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.core.config import Config, get_config, save_config
from src.core.assistant import Assistant


def main():
    """Main entry point"""
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Starlight")
    app.setApplicationVersion("1.0.0")
    
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
