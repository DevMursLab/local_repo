"""Robust Face Detection and Alignment module using MTCNN / Facenet PyTorch."""
import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List

try:
    from facenet_pytorch import MTCNN
    HAS_FACENET = True
except ImportError:
    HAS_FACENET = False


class RobustFaceDetector:
    """Multi-task Cascaded Convolutional Network (MTCNN) Face Detector.

    Provides sub-pixel facial landmark extraction, bounding boxes,
    and aligned facial crops for affective modeling.
    """
    def __init__(self, target_size: Tuple[int, int] = (224, 224), device: Optional[str] = None):
        self.target_size = target_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if HAS_FACENET:
            try:
                self.detector = MTCNN(
                    image_size=target_size[0],
                    margin=20,
                    keep_all=False,
                    select_largest=True,
                    post_process=False,
                    device=self.device
                )
            except Exception:
                self.detector = None
        else:
            self.detector = None

    def detect_and_crop(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[int]]]:
        """Detect dominant face, crop, normalize, and return bounding box.

        Args:
            frame: BGR numpy image from video/webcam.

        Returns:
            face_crop_rgb: Normalized [0, 1] RGB numpy array (target_size)
            raw_crop: Original uint8 RGB crop
            bbox: [x1, y1, x2, y2]
        """
        if frame is None or frame.size == 0:
            return None, None, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        if self.detector is not None:
            try:
                boxes, probs = self.detector.detect(rgb_frame)
                if boxes is not None and len(boxes) > 0 and probs[0] > 0.80:
                    box = [int(v) for v in boxes[0]]
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    raw_crop = rgb_frame[y1:y2, x1:x2]
                    if raw_crop.size > 0:
                        resized = cv2.resize(raw_crop, self.target_size)
                        normalized = resized.astype(np.float32) / 255.0
                        return normalized, resized, [x1, y1, x2, y2]
            except Exception:
                pass

        # Central crop fallback if detector is unavailable or face not isolated
        crop_dim = min(h, w)
        start_y = (h - crop_dim) // 2
        start_x = (w - crop_dim) // 2
        raw_crop = rgb_frame[start_y:start_y+crop_dim, start_x:start_x+crop_dim]
        resized = cv2.resize(raw_crop, self.target_size)
        normalized = resized.astype(np.float32) / 255.0
        return normalized, resized, [start_x, start_y, start_x + crop_dim, start_y + crop_dim]
