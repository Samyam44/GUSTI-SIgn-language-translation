"""
Webcam Capture Module
Provides webcam video capture functionality for ASL recognition.
"""

import cv2
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WebcamCapture:
    """
    Handles webcam video capture with configurable resolution.
    """
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        """
        Initialize webcam capture.
        
        Args:
            camera_index: Camera device index (default: 0 for primary camera)
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.capture = None
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the camera capture object."""
        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            # Set resolution
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set FPS (optional, camera may not support this)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify settings
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.capture.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Camera {self.camera_index} initialized: "
                       f"{actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            raise
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the webcam.
        
        Returns:
            Frame as numpy array (BGR format), or None if read fails
        """
        if not self.capture or not self.capture.isOpened():
            logger.error("Camera not opened")
            return None
        
        try:
            ret, frame = self.capture.read()
            
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
            
            return frame
            
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return None
    
    def is_opened(self) -> bool:
        """
        Check if camera is opened.
        
        Returns:
            True if camera is opened, False otherwise
        """
        return self.capture is not None and self.capture.isOpened()
    
    def get_fps(self) -> float:
        """
        Get camera FPS.
        
        Returns:
            FPS value or 0 if camera not opened
        """
        if self.capture and self.capture.isOpened():
            return self.capture.get(cv2.CAP_PROP_FPS)
        return 0.0
    
    def get_resolution(self) -> tuple[int, int]:
        """
        Get current camera resolution.
        
        Returns:
            Tuple of (width, height)
        """
        if self.capture and self.capture.isOpened():
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Change camera resolution.
        
        Args:
            width: New width in pixels
            height: New height in pixels
            
        Returns:
            True if successful, False otherwise
        """
        if not self.capture or not self.capture.isOpened():
            logger.error("Cannot set resolution: camera not opened")
            return False
        
        try:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width == width and actual_height == height:
                self.width = width
                self.height = height
                logger.info(f"Resolution changed to {width}x{height}")
                return True
            else:
                logger.warning(f"Requested {width}x{height}, got {actual_width}x{actual_height}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting resolution: {e}")
            return False
    
    def release(self):
        """Release the camera resource."""
        if self.capture:
            try:
                self.capture.release()
                logger.info(f"Camera {self.camera_index} released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.capture = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()


# Utility function to list available cameras
def list_available_cameras(max_cameras: int = 10) -> list[int]:
    """
    List available camera indices.
    
    Args:
        max_cameras: Maximum number of cameras to check
        
    Returns:
        List of available camera indices
    """
    available = []
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available


# Example usage
if __name__ == "__main__":
    # List available cameras
    cameras = list_available_cameras()
    print(f"Available cameras: {cameras}")
    
    # Test camera capture
    if cameras:
        with WebcamCapture(camera_index=cameras[0]) as cam:
            print(f"Camera resolution: {cam.get_resolution()}")
            print(f"Camera FPS: {cam.get_fps()}")
            
            # Read a test frame
            frame = cam.read_frame()
            if frame is not None:
                print(f"Frame shape: {frame.shape}")
            else:
                print("Failed to read frame")
