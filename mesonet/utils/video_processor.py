"""
Video Processing Module for MesoNet

Handles video frame extraction and optional face detection/cropping.
"""

import cv2
import numpy as np
import os
from typing import List, Optional, Tuple

from utils.face_extractor import FaceExtractor, get_face_extractor


def extract_frames(video_path: str, 
                   num_frames: int = 10,
                   extract_faces: bool = False,
                   face_extractor: Optional[FaceExtractor] = None) -> List[np.ndarray]:
    """
    Extract evenly distributed frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 10)
        extract_faces: If True, detect and crop faces from each frame
        face_extractor: Optional FaceExtractor instance
    
    Returns:
        List of frame arrays in RGB format. If extract_faces=True, returns
        cropped face images (256x256). Otherwise returns raw frames.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle short videos or empty videos
    if total_frames <= 0:
        cap.release()
        return []
        
    if total_frames < num_frames:
        # If video is too short, just take all frames
        frame_indices = list(range(total_frames))
    else:
        # Evenly distribute indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Initialize face extractor if needed
    if extract_faces and face_extractor is None:
        face_extractor = get_face_extractor()
    
    frames = []
    last_face = None  # Cache last detected face for frames with no detection
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # Convert BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if extract_faces:
            # Extract face from frame
            extracted = face_extractor.extract_faces_from_frame(frame_rgb, max_faces=1)
            
            if extracted:
                face = extracted[0]
                last_face = face  # Cache for fallback
                frames.append(face)
            elif last_face is not None:
                # Use last detected face if detection failed on this frame
                frames.append(last_face)
            # else: skip this frame entirely
        else:
            frames.append(frame_rgb)
            
    cap.release()
    return frames


def extract_frames_with_faces(video_path: str, 
                               num_frames: int = 10,
                               face_extractor: Optional[FaceExtractor] = None) -> Tuple[List[np.ndarray], int]:
    """
    Extract frames from video and detect faces.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        face_extractor: Optional FaceExtractor instance
        
    Returns:
        Tuple of (list of face images, number of successful detections)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return [], 0
        
    # Determine which frames to sample
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Initialize face extractor
    if face_extractor is None:
        face_extractor = get_face_extractor()
    
    faces = []
    successful_detections = 0
    last_face = None
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        extracted = face_extractor.extract_faces_from_frame(frame_rgb, max_faces=1)
        
        if extracted:
            face = extracted[0]
            last_face = face
            faces.append(face)
            successful_detections += 1
        elif last_face is not None:
            # Fallback to last known face
            faces.append(last_face)
    
    cap.release()
    return faces, successful_detections
