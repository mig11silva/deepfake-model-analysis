"""
Face Extraction Module for MesoNet

Uses OpenCV's DNN face detector (or Haar Cascade as fallback) to extract
faces from video frames. This is critical for MesoNet to work correctly
as it was trained on cropped face images, not full frames.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional

# Path to store downloaded model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'face_detector')


class FaceExtractor:
    """
    Face detection and extraction using OpenCV.
    Supports multiple backends: DNN (preferred) and Haar Cascade (fallback).
    """
    
    # DNN model URLs (Caffe-based face detector)
    DNN_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    DNN_CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    def __init__(self, 
                 backend: str = 'auto',
                 confidence_threshold: float = 0.5,
                 target_size: Tuple[int, int] = (256, 256),
                 margin: float = 0.3):
        """
        Initialize face extractor.
        
        Args:
            backend: 'dnn', 'haar', or 'auto' (tries DNN first, falls back to Haar)
            confidence_threshold: Minimum confidence for DNN face detection
            target_size: Output face size (width, height)
            margin: Extra margin around detected face (0.3 = 30%)
        """
        self.confidence_threshold = confidence_threshold
        self.target_size = target_size
        self.margin = margin
        self.backend = backend
        self.detector = None
        self.detector_type = None
        
        self._init_detector(backend)
    
    def _init_detector(self, backend: str):
        """Initialize the face detector based on backend preference."""
        if backend == 'auto':
            # Try DNN first, fall back to Haar
            if self._try_init_dnn():
                self.detector_type = 'dnn'
            else:
                self._init_haar()
                self.detector_type = 'haar'
        elif backend == 'dnn':
            if not self._try_init_dnn():
                raise RuntimeError("Failed to initialize DNN face detector. Check model files.")
            self.detector_type = 'dnn'
        elif backend == 'haar':
            self._init_haar()
            self.detector_type = 'haar'
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        print(f"  Face detector initialized: {self.detector_type.upper()}")
    
    def _try_init_dnn(self) -> bool:
        """Try to initialize DNN-based face detector."""
        try:
            # Ensure model directory exists
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            prototxt_path = os.path.join(MODEL_DIR, 'deploy.prototxt')
            caffemodel_path = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')
            
            # Download models if they don't exist
            if not os.path.exists(prototxt_path):
                print("  Downloading DNN face detector prototxt...")
                self._download_file(self.DNN_PROTOTXT_URL, prototxt_path)
            
            if not os.path.exists(caffemodel_path):
                print("  Downloading DNN face detector caffemodel (~10MB)...")
                self._download_file(self.DNN_CAFFEMODEL_URL, caffemodel_path)
            
            # Load the model
            self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            return True
            
        except Exception as e:
            print(f"  DNN detector init failed: {e}")
            return False
    
    def _download_file(self, url: str, path: str):
        """Download a file from URL."""
        import urllib.request
        urllib.request.urlretrieve(url, path)
    
    def _init_haar(self):
        """Initialize Haar Cascade face detector (fallback)."""
        # Use OpenCV's built-in Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar cascade classifier")
    
    def detect_faces_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using DNN backend.
        
        Args:
            frame: RGB image
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        h, w = frame.shape[:2]
        
        # DNN expects BGR, but we have RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Prepare blob for DNN
        blob = cv2.dnn.blobFromImage(
            frame_bgr, 1.0, (300, 300),
            (104.0, 177.0, 123.0), False, False
        )
        
        self.detector.setInput(blob)
        detections = self.detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Convert to (x, y, w, h) format
                x = max(0, x1)
                y = max(0, y1)
                face_w = min(w, x2) - x
                face_h = min(h, y2) - y
                
                if face_w > 0 and face_h > 0:
                    faces.append((x, y, face_w, face_h))
        
        return faces
    
    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using Haar Cascade backend.
        
        Args:
            frame: RGB image
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        # Convert to grayscale for Haar
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        # Convert from numpy array to list of tuples
        return [tuple(f) for f in faces]
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in a frame using the configured backend.
        
        Args:
            frame: RGB image
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if self.detector_type == 'dnn':
            return self.detect_faces_dnn(frame)
        else:
            return self.detect_faces_haar(frame)
    
    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and resize a face from a frame given a bounding box.
        
        Args:
            frame: RGB image
            bbox: (x, y, w, h) bounding box
            
        Returns:
            Cropped and resized face image
        """
        h, w = frame.shape[:2]
        x, y, face_w, face_h = bbox
        
        # Add margin around the face
        margin_x = int(face_w * self.margin)
        margin_y = int(face_h * self.margin)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + face_w + margin_x)
        y2 = min(h, y + face_h + margin_y)
        
        # Crop face
        face = frame[y1:y2, x1:x2]
        
        # Resize to target size
        face_resized = cv2.resize(face, self.target_size)
        
        return face_resized
    
    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest face from a list of detections."""
        if not faces:
            return None
        
        # Find largest by area
        return max(faces, key=lambda f: f[2] * f[3])
    
    def extract_faces_from_frame(self, frame: np.ndarray, max_faces: int = 1) -> List[np.ndarray]:
        """
        Detect and extract faces from a frame.
        
        Args:
            frame: RGB image
            max_faces: Maximum number of faces to return (sorted by size)
            
        Returns:
            List of cropped face images
        """
        faces_bbox = self.detect_faces(frame)
        
        if not faces_bbox:
            return []
        
        # Sort by area (largest first)
        faces_bbox = sorted(faces_bbox, key=lambda f: f[2] * f[3], reverse=True)
        
        # Extract top N faces
        extracted_faces = []
        for bbox in faces_bbox[:max_faces]:
            face = self.extract_face(frame, bbox)
            extracted_faces.append(face)
        
        return extracted_faces


# Global instance for reuse
_face_extractor = None


def get_face_extractor(backend: str = 'auto') -> FaceExtractor:
    """Get or create a global FaceExtractor instance."""
    global _face_extractor
    if _face_extractor is None:
        _face_extractor = FaceExtractor(backend=backend)
    return _face_extractor


def extract_face_from_frame(frame: np.ndarray, 
                            extractor: Optional[FaceExtractor] = None) -> Optional[np.ndarray]:
    """
    Convenience function to extract the largest face from a frame.
    
    Args:
        frame: RGB image
        extractor: Optional FaceExtractor instance (uses global if None)
        
    Returns:
        Cropped face image or None if no face detected
    """
    if extractor is None:
        extractor = get_face_extractor()
    
    faces = extractor.extract_faces_from_frame(frame, max_faces=1)
    return faces[0] if faces else None

