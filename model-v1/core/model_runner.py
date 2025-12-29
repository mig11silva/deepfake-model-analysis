"""
Model Runner - Orchestrates the deepfake detection analysis pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, SiglipForImageClassification

from core.image_loader import ImageLoader, ImageData
from core.results_generator import ResultsGenerator


# Configuration
BASE_DIR = Path(__file__).parent.parent.absolute()
THREAD_POOL_SIZE = 8


class ModelRunner:
    """Runs deepfake detection on a dataset and generates results."""
    
    def __init__(self):
        """Initialize the runner."""
        self.model = None
        self.processor = None
        self.image_loader = ImageLoader()
        self.results_generator = ResultsGenerator()
        
        # Label mapping
        self.id2label = {0: "fake", 1: "real"}
    
    def _load_model(self) -> None:
        """Load the HuggingFace deepfake detection model."""
        print("Loading model...")
        
        model_name = "prithivMLmods/deepfake-detector-model-v1"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipForImageClassification.from_pretrained(model_name)
        self.model.eval()
        
        print("  ✓ Model loaded")
    
    def _predict(self, image_path: str) -> Dict[str, Any]:
        """Run inference on a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
            
            predicted_idx = torch.argmax(probs).item()
            
            return {
                'prediction': self.id2label[predicted_idx],
                'confidence': round(probs[predicted_idx].item(), 4),
                'image_path': image_path
            }
        except Exception as e:
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'image_path': image_path,
                'error': str(e)
            }
    
    def _process_images(self, images: List[ImageData]) -> List[Dict[str, Any]]:
        """Process all images using thread pool."""
        results = []
        
        print(f"\nProcessing {len(images)} images...")
        
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            future_to_image = {
                executor.submit(self._predict, img.path): img 
                for img in images
            }
            
            with tqdm(total=len(images), desc="Inference") as pbar:
                for future in as_completed(future_to_image):
                    image = future_to_image[future]
                    try:
                        result = future.result()
                        result['ground_truth'] = image.ground_truth
                        results.append(result)
                    except Exception as e:
                        results.append({
                            'prediction': 'error',
                            'confidence': 0.0,
                            'image_path': image.path,
                            'ground_truth': image.ground_truth,
                            'error': str(e)
                        })
                    pbar.update(1)
        
        return results
    
    def run(self, size_limit: int = None) -> Dict[str, float]:
        """Execute the analysis pipeline."""
        print("=" * 50)
        print("DEEPFAKE DETECTION ANALYSIS")
        print("=" * 50)
        
        # Load model
        self._load_model()
        
        # Load images
        print("\nLoading dataset...")
        images = self.image_loader.load_all_images(max_per_label=size_limit)
        
        if not images:
            raise RuntimeError("No images found in dataset")
        
        # Run inference
        predictions = self._process_images(images)
        
        # Count errors
        errors = sum(1 for p in predictions if p['prediction'] == 'error')
        if errors > 0:
            print(f"\n  ⚠ {errors} images failed")
        
        # Generate results
        metrics = self.results_generator.generate_all(predictions)
        
        print("\n" + "=" * 50)
        print("ANALYSIS COMPLETE")
        print("=" * 50)
        
        return metrics
