#!/usr/bin/env python3
"""
MesoNet Deepfake Detection Test

Tests the MesoNet model on Deepfakes or Face2Face datasets.
Now includes face extraction for accurate predictions.

Usage:
    python main.py --dataset deepfakes
    python main.py --dataset face2face --size 100 --frames 20
"""

import argparse
import os
import sys
import time
import json
import random
from tqdm import tqdm
from utils.mesonet_predictor import MesoNetPredictor
from utils.results_generator import generate_visualizations


def parse_args():
    parser = argparse.ArgumentParser(
        description='MesoNet Deepfake Detection Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset deepfakes                    # Test all videos
  python main.py --dataset face2face --size 100         # Test 100 videos per class
  python main.py --dataset deepfakes --frames 20        # Extract 20 frames per video
  python main.py --dataset deepfakes --no-face-extract  # Disable face extraction
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['deepfakes', 'face2face'],
        help='Dataset to use: deepfakes or face2face'
    )
    
    parser.add_argument(
        '--size',
        type=int,
        default=None,
        help='Number of videos to test from each class (default: all videos)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=10,
        help='Number of frames to extract per video (default: 10)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='meso4',
        choices=['meso4', 'inception'],
        help='Model variant to use (default: meso4)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory (default: results/)'
    )
    
    parser.add_argument(
        '--no-face-extract',
        action='store_true',
        help='Disable face extraction (not recommended - may reduce accuracy)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for video selection when --size is used (default: 42)'
    )
    
    return parser.parse_args()


def get_video_files(directory: str, size: int = None, seed: int = 42) -> list:
    """
    Get video files from a directory.
    
    Args:
        directory: Path to directory containing videos
        size: Maximum number of videos to return (None = all)
        seed: Random seed for reproducible sampling
        
    Returns:
        List of video filenames
    """
    if not os.path.exists(directory):
        return []
    
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(directory) 
              if f.lower().endswith(valid_exts) and not f.startswith('.')]
    
    # Sort for reproducibility
    videos.sort()
    
    # Sample if size is specified
    if size is not None and len(videos) > size:
        random.seed(seed)
        videos = random.sample(videos, size)
        videos.sort()  # Sort again after sampling
    
    return videos


def main():
    args = parse_args()
    start_time = time.time()
    
    # Setup paths
    real_dir = os.path.join('dataset', 'original')
    fake_dir = os.path.join('dataset', args.dataset)
    
    output_dir = os.path.join(args.output, args.dataset)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    use_face_extraction = not args.no_face_extract
    
    print(f"\n{'='*60}")
    print(f"MesoNet Deepfake Detection Test")
    print(f"{'='*60}")
    print(f"\nConfiguration:")
    print(f"  Dataset:          {args.dataset}")
    print(f"  Model:            {args.model}")
    print(f"  Frames/video:     {args.frames}")
    print(f"  Face extraction:  {'Enabled' if use_face_extraction else 'Disabled'}")
    print(f"  Videos per class: {args.size if args.size else 'All'}")
    print(f"  Output:           {output_dir}")
    
    # 1. Load Model
    print(f"\nStep 1: Loading model...")
    try:
        predictor = MesoNetPredictor(
            args.dataset, 
            args.model,
            use_face_extraction=use_face_extraction
        )
        print(f"  ✓ Model loaded successfully")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        sys.exit(1)
        
    # 2. Load Dataset
    print(f"\nStep 2: Loading dataset...")
    real_videos = get_video_files(real_dir, args.size, args.seed)
    fake_videos = get_video_files(fake_dir, args.size, args.seed)
    
    print(f"  ✓ Found {len(real_videos)} real videos in {real_dir}")
    print(f"  ✓ Found {len(fake_videos)} fake videos in {fake_dir}")
    
    if args.size:
        print(f"  → Using {len(real_videos)} real + {len(fake_videos)} fake = {len(real_videos) + len(fake_videos)} total")
    
    if len(real_videos) == 0 and len(fake_videos) == 0:
        print("\n  ✗ No videos found! Please add videos to the dataset folders.")
        print("  See README.md or setup.py output for instructions.")
        sys.exit(0)
        
    # 3. Run Inference
    print(f"\nStep 3: Running inference...")
    
    results = []
    failed_videos = []
    
    # Process Real Videos
    if real_videos:
        print(f"\n  Processing REAL videos ({len(real_videos)} total)...")
        for video in tqdm(real_videos, desc="  Real Videos", unit="video"):
            video_path = os.path.join(real_dir, video)
            try:
                prediction = predictor.predict_video(video_path, args.frames)
                results.append({
                    'video_name': video,
                    'ground_truth': 'real',
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'avg_raw_score': prediction.get('avg_raw_score', prediction['confidence']),
                    'faces_detected': prediction['aggregation'].get('faces_detected', 0),
                    'details': prediction
                })
            except Exception as e:
                failed_videos.append((video, str(e)))

    # Process Fake Videos
    if fake_videos:
        print(f"\n  Processing FAKE videos ({len(fake_videos)} total)...")
        for video in tqdm(fake_videos, desc="  Fake Videos", unit="video"):
            video_path = os.path.join(fake_dir, video)
            try:
                prediction = predictor.predict_video(video_path, args.frames)
                results.append({
                    'video_name': video,
                    'ground_truth': 'fake',
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'avg_raw_score': prediction.get('avg_raw_score', prediction['confidence']),
                    'faces_detected': prediction['aggregation'].get('faces_detected', 0),
                    'details': prediction
                })
            except Exception as e:
                failed_videos.append((video, str(e)))
    
    # Report failures
    if failed_videos:
        print(f"\n  ⚠ Failed to process {len(failed_videos)} videos:")
        for video, error in failed_videos[:5]:  # Show first 5
            print(f"    - {video}: {error}")
        if len(failed_videos) > 5:
            print(f"    ... and {len(failed_videos) - 5} more")
            
    # 4. Calculate Metrics & Generate Visualizations
    if not results:
        print("\n  ✗ No results generated.")
        sys.exit(0)
        
    print(f"\nStep 4: Calculating metrics and generating visualizations...")
    
    # Filter out unknown predictions
    valid_results = [r for r in results if r['prediction'] != 'unknown']
    
    if not valid_results:
        print("  ✗ No valid predictions (all faces failed detection).")
        sys.exit(0)
    
    print(f"  → {len(valid_results)}/{len(results)} videos with valid predictions")
    
    true_labels = [r['ground_truth'] for r in valid_results]
    predicted_labels = [r['prediction'] for r in valid_results]
    confidences = [r['confidence'] for r in valid_results]
    
    # Calculate raw scores for ROC
    raw_scores = [r['avg_raw_score'] for r in valid_results]
    
    model_display_name = f"MesoNet {args.model.capitalize()} ({args.dataset.capitalize()})"
    
    metrics = generate_visualizations(
        true_labels, 
        predicted_labels, 
        confidences, 
        raw_scores, 
        output_dir, 
        model_name=model_display_name
    )
    
    # Calculate face detection rate
    total_faces = sum(r['faces_detected'] for r in results)
    max_possible = len(results) * args.frames
    face_detection_rate = total_faces / max_possible if max_possible > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f} ({v*100:.2f}%)")
    
    print(f"\nFace Detection:")
    print(f"  Faces detected: {total_faces:,} / {max_possible:,} ({face_detection_rate*100:.1f}%)")
    
    print(f"\nPrediction Breakdown:")
    correct = sum(1 for r in valid_results if r['ground_truth'] == r['prediction'])
    incorrect = len(valid_results) - correct
    print(f"  Correct:   {correct:,} ({correct/len(valid_results)*100:.1f}%)")
    print(f"  Incorrect: {incorrect:,} ({incorrect/len(valid_results)*100:.1f}%)")
        
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}")
    print(f"Total processing time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
    
    other_dataset = 'face2face' if args.dataset == 'deepfakes' else 'deepfakes'
    print(f"\nTo test {other_dataset.capitalize()} dataset, run:")
    print(f"  python main.py --dataset {other_dataset}")
    print()


if __name__ == "__main__":
    main()
