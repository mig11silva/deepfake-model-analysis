#!/usr/bin/env python3
"""
Deepfake Detection Analysis

Analyzes images using a HuggingFace deepfake detector model and generates
metrics and visualizations.

Usage:
    python main.py
    python main.py --size 100  # Process 100 fake + 100 real images
"""

import argparse
import sys
from pathlib import Path

from core.model_runner import ModelRunner


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze images with deepfake detection model"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=None,
        help="Limit images per category (e.g., --size 100 = 100 fake + 100 real)"
    )
    args = parser.parse_args()
    
    try:
        runner = ModelRunner()
        metrics = runner.run(size_limit=args.size)
        
        print(f"\nFinal Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall:    {metrics['recall']:.2%}")
        print(f"  F1-Score:  {metrics['f1_score']:.2%}")
        print(f"\nResults saved to: results/")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
