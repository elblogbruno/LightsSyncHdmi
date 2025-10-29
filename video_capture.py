"""Video capture initialization and management."""
import cv2


def setup_video_capture(device_index=0, frame_width=320, frame_height=240):
    """
    Set up video capture with specified parameters.
    
    Args:
        device_index: Video capture device index (default: 0)
        frame_width: Frame width in pixels
        frame_height: Frame height in pixels
    
    Returns:
        cv2.VideoCapture: Configured video capture object
    
    Raises:
        RuntimeError: If video source could not be opened
    """
    cap = cv2.VideoCapture(device_index)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video source.")
    
    return cap
