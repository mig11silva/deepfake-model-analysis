"""
Image Loader - Handles loading images from the dataset directory.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass


# Configuration
BASE_DIR = Path(__file__).parent.parent.absolute()
DATASET_DIR = BASE_DIR / "dataset"
FAKE_DIR = DATASET_DIR / "fake"
REAL_DIR = DATASET_DIR / "real"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


@dataclass
class ImageData:
    """Container for image data."""
    path: str
    ground_truth: str
    filename: str


class ImageLoader:
    """Loads and organizes images from the dataset."""
    
    def __init__(self):
        """Initialize with default paths."""
        self.fake_dir = FAKE_DIR
        self.real_dir = REAL_DIR
    
    def _is_image(self, filename: str) -> bool:
        """Check if file is a supported image."""
        return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
    
    def _scan_directory(self, directory: Path, label: str, max_images: int = None) -> List[ImageData]:
        """Scan directory for images."""
        images = []
        
        if not directory.exists():
            print(f"WARNING: Directory not found: {directory}")
            return images
        
        for filename in os.listdir(directory):
            if self._is_image(filename):
                images.append(ImageData(
                    path=str(directory / filename),
                    ground_truth=label,
                    filename=filename
                ))
                if max_images and len(images) >= max_images:
                    break
        
        return images
    
    def load_all_images(self, max_per_label: int = None) -> List[ImageData]:
        """Load all images from fake and real directories."""
        print("Scanning directories...")
        
        fake_images = self._scan_directory(self.fake_dir, "fake", max_per_label)
        real_images = self._scan_directory(self.real_dir, "real", max_per_label)
        
        all_images = fake_images + real_images
        
        print(f"  Found {len(fake_images)} fake, {len(real_images)} real images")
        if max_per_label:
            print(f"  (limited to {max_per_label} per category)")
        
        return all_images
