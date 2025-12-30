"""
MesoNet Predictor Module

Handles model loading, preprocessing, and inference for deepfake detection.
Supports both Deepfakes and Face2Face datasets with respective weights.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MesoNetPredictor:
    """
    Wrapper for MesoNet deepfake detection model.
    Handles model loading, preprocessing, and inference.
    
    IMPORTANT: MesoNet was trained on cropped FACE images (256x256).
    This predictor now supports face extraction for accurate predictions.
    """
    
    def __init__(self, dataset_type: str, model_type: str = 'meso4', use_face_extraction: bool = True):
        """
        Initialize MesoNet model with appropriate weights.
        
        Args:
            dataset_type: 'deepfakes' or 'face2face'
            model_type: 'meso4' or 'inception' (default: 'meso4')
            use_face_extraction: Whether to extract faces from frames (recommended: True)
        """
        self.dataset_type = dataset_type
        self.use_face_extraction = use_face_extraction
        self.face_extractor = None
        
        # 1. Load Model Architecture
        print(f"  Loading {model_type} model...")
        if model_type == 'meso4':
            from models.MesoNet.classifiers import Meso4
            self.model = Meso4()
        elif model_type == 'inception':
            from models.MesoNet.classifiers import MesoInception4
            self.model = MesoInception4()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 2. Determine Weights Path
        base_weights_dir = os.path.join("models", "MesoNet", "weights")
        
        if dataset_type == 'deepfakes':
            if model_type == 'meso4':
                weights_filename = 'Meso4_DF.h5'
            else:
                weights_filename = 'MesoInception_DF.h5'
        elif dataset_type == 'face2face':
            if model_type == 'meso4':
                weights_filename = 'Meso4_F2F.h5'
            else:
                weights_filename = 'MesoInception_F2F.h5'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
            
        weights_path = os.path.join(base_weights_dir, weights_filename)
        
        # 3. Load Weights
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
            
        print(f"  Loading weights: {weights_path}")
        self.model.load(weights_path)
        
        # 4. Initialize Face Extractor if needed
        if use_face_extraction:
            from utils.face_extractor import FaceExtractor
            self.face_extractor = FaceExtractor(backend='auto')
        
    def predict_face(self, face: np.ndarray) -> Dict[str, Any]:
        """
        Predict if a face image is real or fake.
        
        Args:
            face: RGB face image (already cropped, will be resized to 256x256)
        
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float (0.5 to 1.0),
                'raw_score': float (probability of fake, 0-1)
            }
        """
        # Preprocess: resize and normalize
        face_resized = cv2.resize(face, (256, 256))
        face_normalized = face_resized / 255.0
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        # Get model prediction
        model_output = float(self.model.predict(face_batch)[0][0])
        
        # IMPORTANT: MesoNet's output appears to be probability of REAL, not FAKE
        # Analysis showed: real videos get high scores (~0.66), fake videos get low (~0.28)
        # So we invert: fake_probability = 1 - model_output
        fake_probability = 1.0 - model_output
        
        # Interpret result (score > 0.5 = FAKE)
        prediction = 'fake' if fake_probability > 0.5 else 'real'
        
        # Confidence is the probability of the predicted class
        confidence = fake_probability if prediction == 'fake' else 1 - fake_probability
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'raw_score': fake_probability  # Probability of being fake
        }
    
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict if a frame is real or fake.
        If face extraction is enabled, extracts face first.
        
        Args:
            frame: RGB image (full frame or face crop)
        
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float,
                'raw_score': float,
                'face_detected': bool (if face extraction enabled)
            }
        """
        if self.use_face_extraction and self.face_extractor is not None:
            # Extract face from frame
            faces = self.face_extractor.extract_faces_from_frame(frame, max_faces=1)
            
            if faces:
                result = self.predict_face(faces[0])
                result['face_detected'] = True
                return result
            else:
                # No face detected - return unknown or use full frame as fallback
                return {
                    'prediction': 'unknown',
                    'confidence': 0.0,
                    'raw_score': 0.5,
                    'face_detected': False
                }
        else:
            # No face extraction - predict on full frame (less accurate)
            result = self.predict_face(frame)
            result['face_detected'] = None  # N/A
            return result
    
    def predict_video(self, video_path: str, num_frames: int = 10) -> Dict[str, Any]:
        """
        Predict if a video is real or fake.
        Extracts frames, detects faces, predicts each, then aggregates.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float,
                'frame_predictions': List of per-frame predictions,
                'aggregation': {
                    'fake_frames': int,
                    'real_frames': int,
                    'total_frames': int,
                    'faces_detected': int
                },
                'avg_raw_score': float
            }
        """
        from utils.video_processor import extract_frames
        
        # Extract frames with face detection
        frames = extract_frames(
            video_path, 
            num_frames, 
            extract_faces=self.use_face_extraction,
            face_extractor=self.face_extractor
        )
        
        if not frames:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'frame_predictions': [],
                'aggregation': {
                    'fake_frames': 0,
                    'real_frames': 0,
                    'total_frames': 0,
                    'faces_detected': 0
                },
                'avg_raw_score': 0.5
            }
        
        # Predict each face/frame
        frame_predictions = []
        faces_detected = 0
        
        for frame in frames:
            if self.use_face_extraction:
                # Frame is already a cropped face
                pred = self.predict_face(frame)
                pred['face_detected'] = True
                faces_detected += 1
            else:
                pred = self.predict_frame(frame)
                if pred.get('face_detected', True):
                    faces_detected += 1
            
            frame_predictions.append(pred)
        
        # Filter out unknown predictions for aggregation
        valid_predictions = [p for p in frame_predictions if p['prediction'] != 'unknown']
        
        if not valid_predictions:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'frame_predictions': frame_predictions,
                'aggregation': {
                    'fake_frames': 0,
                    'real_frames': 0,
                    'total_frames': len(frame_predictions),
                    'faces_detected': faces_detected
                },
                'avg_raw_score': 0.5
            }
        
        # Aggregate using majority voting
        fake_count = sum(1 for p in valid_predictions if p['prediction'] == 'fake')
        real_count = len(valid_predictions) - fake_count
        
        # Calculate average raw score
        avg_raw_score = sum(p['raw_score'] for p in valid_predictions) / len(valid_predictions)
        
        # Determine final prediction
        if fake_count > real_count:
            video_prediction = 'fake'
        elif real_count > fake_count:
            video_prediction = 'real'
        else:
            # Tie: use average raw score
            video_prediction = 'fake' if avg_raw_score > 0.5 else 'real'
        
        # Average confidence
        avg_confidence = sum(p['confidence'] for p in valid_predictions) / len(valid_predictions)
        
        return {
            'prediction': video_prediction,
            'confidence': avg_confidence,
            'frame_predictions': frame_predictions,
            'aggregation': {
                'fake_frames': fake_count,
                'real_frames': real_count,
                'total_frames': len(frame_predictions),
                'faces_detected': faces_detected
            },
            'avg_raw_score': avg_raw_score
        }
