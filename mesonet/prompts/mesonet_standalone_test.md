# Create Standalone MesoNet Testing Project

## Overview
Create a **complete, standalone project from scratch** to test the MesoNet deepfake detection models from DariusAf's repository. This project will:

1. Clone the MesoNet repository with pre-trained weights
2. Use two FaceForensics++ datasets: **Deepfakes** and **Face2Face** (user provides after setup)
3. Run inference using the appropriate pre-trained weights for each dataset
4. Generate academic-quality visualizations and metrics (matching the style from the XceptionNet project)

**Repository**: https://github.com/DariusAf/MesoNet

**This is a separate, independent project** - not integrated with your existing deepfake analysis project.

---

## Project Goals

### What This Project Should Do:
1. ✅ Download and setup MesoNet models with both pre-trained weight sets
2. ✅ Support two dataset types via command-line flag:
   - **Deepfakes** dataset (with `Meso4_DF.h5` weights)
   - **Face2Face** dataset (with `Meso4_F2F.h5` weights)
3. ✅ Extract frames from videos (10 frames per video)
4. ✅ Run appropriate MesoNet predictions based on dataset type
5. ✅ Aggregate frame predictions to video-level predictions
6. ✅ Generate the exact same 5 visualizations as the XceptionNet project
7. ✅ Save all results to JSON file
8. ✅ Allow easy switching between datasets via command-line

---

## Expected Project Structure

```
mesonet-deepfake-test/
├── README.md                           # Setup and usage instructions
├── requirements.txt                    # Python dependencies
├── setup.py                            # Setup script (downloads model)
├── main.py                             # Main inference script
│
├── models/
│   └── MesoNet/                        # Cloned from GitHub
│       ├── classifiers.py              # Meso4 and MesoInception4 models
│       ├── example.py
│       ├── pipeline.py
│       └── weights/
│           ├── Meso4_DF.h5            # Pre-trained for Deepfakes
│           ├── Meso4_F2F.h5           # Pre-trained for Face2Face
│           ├── MesoInception_DF.h5    # (optional, not used)
│           └── MesoInception_F2F.h5   # (optional, not used)
│
├── dataset/
│   ├── deepfakes/                      # From FaceForensics++ Deepfakes
│   │   ├── real/                       # Original videos
│   │   │   ├── 000.mp4
│   │   │   └── ... (user adds 10 videos)
│   │   └── fake/                       # Deepfake manipulated videos
│   │       ├── 000_003.mp4
│   │       └── ... (user adds 10 videos)
│   └── face2face/                      # From FaceForensics++ Face2Face
│       ├── real/                       # Original videos
│       │   ├── 000.mp4
│       │   └── ... (user adds 10 videos)
│       └── fake/                       # Face2Face manipulated videos
│           ├── 000_003.mp4
│           └── ... (user adds 10 videos)
│
├── utils/
│   ├── video_processor.py              # Frame extraction
│   ├── mesonet_predictor.py            # MesoNet inference wrapper
│   └── results_generator.py            # Visualization generation
│
└── results/
    ├── deepfakes/                      # Results for Deepfakes dataset
    │   ├── confusion_matrix.png
    │   ├── metrics_bar_chart.png
    │   ├── prediction_distribution.png
    │   ├── confidence_distribution.png
    │   ├── roc_curve.png
    └── face2face/                      # Results for Face2Face dataset
        ├── confusion_matrix.png
        ├── metrics_bar_chart.png
        ├── prediction_distribution.png
        ├── confidence_distribution.png
        ├── roc_curve.png
```

---

## Command-Line Usage

### Basic Usage:
```bash
# Test with Deepfakes dataset (uses Meso4_DF.h5 weights)
python main.py --dataset deepfakes

# Test with Face2Face dataset (uses Meso4_F2F.h5 weights)
python main.py --dataset face2face
```

### Advanced Options:
```bash
# Extract more frames per video
python main.py --dataset deepfakes --frames 15

# Specify custom output directory
python main.py --dataset face2face --output results/experiment1/

# Use MesoInception model instead of Meso4
python main.py --dataset deepfakes --model inception
```

---

## Requirements

### 1. Setup Script (`setup.py`)

**Purpose**: Download MesoNet repository and create folder structure

**What it should do:**
1. Clone DariusAf/MesoNet repository
2. Verify pre-trained weights exist in `weights/` folder
3. Create dataset folder structure for both datasets
4. Print clear instructions for adding videos

**Key features:**
- Check if Keras/TensorFlow is installed
- Verify both weight files are present:
  - `Meso4_DF.h5` (for Deepfakes)
  - `Meso4_F2F.h5` (for Face2Face)
- Create empty dataset folders
- Print dataset preparation instructions

**Dataset preparation instructions to include:**
```
Dataset Preparation Instructions:
==================================

1. Download FaceForensics++ dataset (requires form approval):
   https://github.com/ondyari/FaceForensics

2. For Deepfakes dataset:
   - Copy 10 videos from original_sequences/youtube/c23/videos/ to dataset/deepfakes/real/
   - Copy 10 videos from manipulated_sequences/Deepfakes/c23/videos/ to dataset/deepfakes/fake/

3. For Face2Face dataset:
   - Copy 10 videos from original_sequences/youtube/c23/videos/ to dataset/face2face/real/
   - Copy 10 videos from manipulated_sequences/Face2Face/c23/videos/ to dataset/face2face/fake/

4. Optionally rename videos for simplicity:
   - Real: 000.mp4, 001.mp4, ..., 009.mp4
   - Fake: 000_fake.mp4, 001_fake.mp4, ..., 009_fake.mp4

5. Run inference:
   python main.py --dataset deepfakes
   python main.py --dataset face2face
```

### 2. Video Processor (`utils/video_processor.py`)

**Purpose**: Extract frames from videos

**Same as XceptionNet project** - reuse the logic:
```python
def extract_frames(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
    """
    Extract evenly distributed frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 10)
    
    Returns:
        List of frame arrays in RGB format (256x256 for MesoNet)
    """
```

**Key differences from XceptionNet:**
- Resize frames to **256x256** (not 299x299)
- MesoNet expects this size

### 3. MesoNet Predictor (`utils/mesonet_predictor.py`)

**Purpose**: Wrapper for MesoNet model inference

**Key class:**
```python
class MesoNetPredictor:
    """
    Wrapper for MesoNet deepfake detection model.
    Handles model loading, preprocessing, and inference.
    Supports both Deepfakes and Face2Face datasets with respective weights.
    """
    
    def __init__(self, dataset_type: str, model_type: str = 'meso4'):
        """
        Initialize MesoNet model with appropriate weights.
        
        Args:
            dataset_type: 'deepfakes' or 'face2face'
            model_type: 'meso4' or 'inception' (default: 'meso4')
        """
        # Load appropriate model
        if model_type == 'meso4':
            from models.MesoNet.classifiers import Meso4
            self.model = Meso4()
        else:
            from models.MesoNet.classifiers import MesoInception4
            self.model = MesoInception4()
        
        # Load appropriate weights based on dataset
        if dataset_type == 'deepfakes':
            weights_path = 'models/MesoNet/weights/Meso4_DF.h5'
        elif dataset_type == 'face2face':
            weights_path = 'models/MesoNet/weights/Meso4_F2F.h5'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        self.model.load(weights_path)
        self.dataset_type = dataset_type
        
    def predict_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Predict if a single frame is real or fake.
        
        Args:
            frame: RGB image array (any size, will be resized to 256x256)
        
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float (0.0 to 1.0),
                'raw_score': float (model's raw output)
            }
        """
        # Preprocess frame
        # MesoNet expects: (1, 256, 256, 3) shape, values 0-1
        frame_resized = cv2.resize(frame, (256, 256))
        frame_normalized = frame_resized / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        # Predict
        # MesoNet returns single value (probability of being FAKE)
        prediction_score = float(self.model.predict(frame_batch)[0][0])
        
        # Interpret result
        prediction = 'fake' if prediction_score > 0.5 else 'real'
        confidence = prediction_score if prediction == 'fake' else 1 - prediction_score
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'raw_score': prediction_score
        }
    
    def predict_video(self, video_path: str, num_frames: int = 10) -> Dict[str, Any]:
        """
        Predict if a video is real or fake.
        Extracts frames, predicts each, then aggregates.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
        
        Returns:
            {
                'prediction': 'real' or 'fake',
                'confidence': float,
                'frame_predictions': List of per-frame predictions,
                'aggregation': {'fake_frames': int, 'real_frames': int}
            }
        """
        # Extract frames
        from utils.video_processor import extract_frames
        frames = extract_frames(video_path, num_frames)
        
        # Predict each frame
        frame_predictions = []
        for frame in frames:
            pred = self.predict_frame(frame)
            frame_predictions.append(pred)
        
        # Aggregate using majority voting
        fake_count = sum(1 for p in frame_predictions if p['prediction'] == 'fake')
        real_count = len(frame_predictions) - fake_count
        
        video_prediction = 'fake' if fake_count > real_count else 'real'
        avg_confidence = sum(p['confidence'] for p in frame_predictions) / len(frame_predictions)
        
        return {
            'prediction': video_prediction,
            'confidence': avg_confidence,
            'frame_predictions': frame_predictions,
            'aggregation': {
                'fake_frames': fake_count,
                'real_frames': real_count,
                'total_frames': len(frame_predictions)
            }
        }
```

**Key implementation notes:**
- **MesoNet uses Keras/TensorFlow** (not PyTorch like XceptionNet)
- **Input size**: 256x256 pixels (not 299x299)
- **Normalization**: Divide by 255.0 (simple 0-1 scaling)
- **Output**: Single probability value (0-1) where high = fake
- **Weights mapping**:
  - Deepfakes dataset → `Meso4_DF.h5`
  - Face2Face dataset → `Meso4_F2F.h5`

### 4. Results Generator (`utils/results_generator.py`)

**IMPORTANT**: Use the **EXACT SAME** implementation as the XceptionNet project.

This should generate identical visualizations:
1. Confusion Matrix
2. Metrics Bar Chart
3. Prediction Distribution
4. Confidence Distribution
5. ROC Curve

**Only differences:**
- Model name in titles: "MesoNet (Deepfakes)" or "MesoNet (Face2Face)"
- Output directories: `results/deepfakes/` or `results/face2face/`

**All styling must match exactly:**
- Same colors
- Same figure sizes
- Same fonts
- Same DPI (300)
- Same layout

### 5. Main Script (`main.py`)

**Purpose**: Orchestrate the entire pipeline with command-line options

**Command-line arguments:**
```python
import argparse

parser = argparse.ArgumentParser(
    description='MesoNet Deepfake Detection Test'
)

parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    choices=['deepfakes', 'face2face'],
    help='Dataset to use: deepfakes or face2face'
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
```

**Execution flow:**
```python
1. Parse command-line arguments
2. Load MesoNet model with appropriate weights based on dataset type:
   - If --dataset deepfakes → load Meso4_DF.h5
   - If --dataset face2face → load Meso4_F2F.h5
3. Find all videos in dataset/{dataset_type}/real/ and dataset/{dataset_type}/fake/
4. For each video:
   a. Extract N frames (default 10)
   b. Run MesoNet prediction on each frame
   c. Aggregate to video-level prediction
   d. Store result with ground truth
5. Calculate overall metrics (accuracy, precision, recall, F1)
6. Generate all 5 visualizations
7. Save results to results/{dataset_type}/
8. Print summary with comparison note
```

**Example output:**
```
MesoNet Deepfake Detection Test
================================

Configuration:
  Dataset:     deepfakes
  Model:       Meso4
  Weights:     Meso4_DF.h5
  Frames/vid:  10
  Output:      results/deepfakes/

Step 1: Loading model...
  ✓ Loaded MesoNet Meso4
  ✓ Loaded weights: models/MesoNet/weights/Meso4_DF.h5

Step 2: Loading dataset...
  ✓ Found 10 real videos in dataset/deepfakes/real/
  ✓ Found 10 fake videos in dataset/deepfakes/fake/

Step 3: Running inference...
  Processing video 1/20: 000.mp4 (real)
    - Extracting 10 frames...
    - Predicting frames... [████████] 10/10
    - Prediction: REAL (confidence: 0.89)
  
  Processing video 2/20: 001.mp4 (real)
    - Extracting 10 frames...
    - Predicting frames... [████████] 10/10
    - Prediction: REAL (confidence: 0.91)
  
  ... (continue for all 20 videos)

Step 4: Calculating metrics...
  Accuracy:  0.95 (95.00%)
  Precision: 0.94 (94.00%)
  Recall:    0.95 (95.00%)
  F1-Score:  0.94 (94.00%)

Step 5: Generating visualizations...
  ✓ Saved: confusion_matrix.png
  ✓ Saved: metrics_bar_chart.png
  ✓ Saved: prediction_distribution.png
  ✓ Saved: confidence_distribution.png
  ✓ Saved: roc_curve.png
  ✓ Saved: results.json
  ✓ Saved: video_predictions.json

================================
All results saved to: results/deepfakes/
Total processing time: 1m 47s

To test Face2Face dataset, run:
  python main.py --dataset face2face
```

---

## Code Style Requirements

**CRITICAL: Keep it beginner-friendly!**

✅ **DO:**
- Add extensive comments explaining every step
- Use descriptive variable names
- Keep functions short and focused (<50 lines)
- Add type hints for all function parameters
- Include docstrings for all classes and functions
- Print progress messages for each video
- Handle errors gracefully with try-except
- Show clear error messages
- **Explain the difference between Deepfakes and Face2Face** in comments

❌ **DON'T:**
- Use complex design patterns
- Create deep inheritance hierarchies
- Use advanced Python features unnecessarily
- Write cryptic one-liners
- Mix up the weight files for different datasets

---

## Key Implementation Details

### Weight Selection Logic
```python
# CRITICAL: Match weights to dataset type
WEIGHTS_MAP = {
    'deepfakes': {
        'meso4': 'models/MesoNet/weights/Meso4_DF.h5',
        'inception': 'models/MesoNet/weights/MesoInception_DF.h5'
    },
    'face2face': {
        'meso4': 'models/MesoNet/weights/Meso4_F2F.h5',
        'inception': 'models/MesoNet/weights/MesoInception_F2F.h5'
    }
}

# Example usage
dataset_type = args.dataset  # 'deepfakes' or 'face2face'
model_type = args.model       # 'meso4' or 'inception'
weights_path = WEIGHTS_MAP[dataset_type][model_type]
```

### MesoNet Output Interpretation
```python
# MesoNet returns probability of being FAKE (0-1)
fake_probability = model.predict(image)[0][0]

# Convert to prediction
if fake_probability > 0.5:
    prediction = 'fake'
    confidence = fake_probability
else:
    prediction = 'real'
    confidence = 1 - fake_probability
```

### Frame Preprocessing for MesoNet
```python
# MesoNet expects:
# - Shape: (batch_size, 256, 256, 3)
# - Values: 0.0 to 1.0 (normalized)
# - Format: RGB

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (256, 256))
frame_normalized = frame_resized / 255.0
frame_batch = np.expand_dims(frame_normalized, axis=0)
```

---

## Testing Checklist

Before considering complete:

✅ Setup script successfully clones MesoNet repository
✅ Both weight files are present and accessible
✅ Can load Meso4 model with Deepfakes weights
✅ Can load Meso4 model with Face2Face weights
✅ Can switch between datasets via --dataset flag
✅ Correct weights are loaded for each dataset type
✅ Can extract frames from videos (256x256)
✅ Can predict on a single frame
✅ Can predict on a full video (with frame aggregation)
✅ Generates all 5 required visualizations (matching XceptionNet style)
✅ Results saved to correct subdirectory (deepfakes/ or face2face/)
✅ Clear progress indicators throughout
✅ Handles errors gracefully
✅ Code is well-commented and beginner-friendly
✅ Can run both datasets sequentially without issues

---

## Expected Results Structure

```
results/
├── deepfakes/
│   ├── confusion_matrix.png            # 2x2 heatmap
│   ├── metrics_bar_chart.png           # 4 bars with percentages
│   ├── prediction_distribution.png     # Fake vs Real counts
│   ├── confidence_distribution.png     # Overlapping histograms
│   ├── roc_curve.png                   # ROC with AUC
│   ├── results.json                    # All metrics
│   └── video_predictions.json          # Per-video details
└── face2face/
    ├── confusion_matrix.png            # Same visualizations
    ├── metrics_bar_chart.png
    ├── prediction_distribution.png
    ├── confidence_distribution.png
    ├── roc_curve.png
    ├── results.json
    └── video_predictions.json
```

---

## README.md Content

Include in the generated README:

```markdown
# MesoNet Deepfake Detection Test

Test implementation of MesoNet models on FaceForensics++ dataset samples.

Supports two manipulation methods:
- **Deepfakes** - GAN-based face swapping
- **Face2Face** - Real-time facial reenactment

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run setup (clones MesoNet repo):
   ```bash
   python setup.py
   ```

3. Add dataset videos:
   - **Deepfakes**: Add 10 real + 10 fake videos to `dataset/deepfakes/`
   - **Face2Face**: Add 10 real + 10 fake videos to `dataset/face2face/`

## Usage

Test with Deepfakes dataset:
```bash
python main.py --dataset deepfakes
```

Test with Face2Face dataset:
```bash
python main.py --dataset face2face
```

Advanced options:
```bash
python main.py --dataset deepfakes --frames 15 --model inception
```

Results will be saved to `results/{dataset_type}/`

## Models

- **Meso4**: Lightweight CNN (4 layers)
- **MesoInception4**: Inception-based variant (more sophisticated)

## Pre-trained Weights

- `Meso4_DF.h5` - Trained on Deepfakes
- `Meso4_F2F.h5` - Trained on Face2Face
- `MesoInception_DF.h5` - Inception variant for Deepfakes
- `MesoInception_F2F.h5` - Inception variant for Face2Face

## Requirements
- Python 3.5+
- Keras 2.1.5+
- TensorFlow
- OpenCV
- NumPy, Matplotlib, Seaborn, Scikit-learn

## Dataset
Uses FaceForensics++ dataset. See setup instructions for download.

## Original Model
MesoNet from: https://github.com/DariusAf/MesoNet

## Citation
```
@inproceedings{afchar2018mesonet,
  title={Mesonet: a compact facial video forgery detection network},
  author={Afchar, Darius and Nozick, Vincent and Yamagishi, Junichi and Echizen, Isao},
  booktitle={IEEE WIFS},
  year={2018}
}
```
```

---

## Dependencies (`requirements.txt`)

```txt
tensorflow>=2.0.0
keras>=2.1.5
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```

---

## Comparison Workflow

After running both datasets, you can compare results:

```bash
# Run both datasets
python main.py --dataset deepfakes
python main.py --dataset face2face

# Results will be in:
# - results/deepfakes/
# - results/face2face/

# Compare the metrics from results.json files
# Compare the visualizations side-by-side
```

**Expected observations:**
- MesoNet should perform slightly differently on each dataset
- Deepfakes weights may perform better on Deepfakes dataset
- Face2Face weights may perform better on Face2Face dataset
- This demonstrates the importance of training data specificity

---

## Success Criteria

✅ Complete standalone project that works out-of-the-box
✅ Clones and sets up MesoNet automatically
✅ Supports both Deepfakes and Face2Face datasets via flag
✅ Automatically selects correct pre-trained weights
✅ Processes videos and generates predictions
✅ Creates publication-quality visualizations (matching XceptionNet style)
✅ Separates results by dataset type
✅ Well-documented with clear README
✅ Beginner-friendly code with extensive comments
✅ Handles common errors gracefully
✅ Easy comparison between two datasets

---

## Key Differences from XceptionNet Project

| Aspect | XceptionNet | MesoNet |
|--------|------------|---------|
| Framework | PyTorch | Keras/TensorFlow |
| Input Size | 299x299 | 256x256 |
| Output | 2-class softmax | Single probability |
| Normalization | ImageNet (mean/std) | Simple (0-1) |
| Weights | Single file | Two files (DF/F2F) |
| Dataset Support | One at a time | **Two datasets selectable** |

---

## Final Notes

This project is designed to:
- ✅ **Test MesoNet on two different manipulation types**
- ✅ **Compare Deepfakes vs Face2Face detection**
- ✅ **Use dataset-specific pre-trained weights**
- ✅ **Generate identical visualizations** to XceptionNet for comparison
- ✅ **Remain simple and educational**

Perfect for understanding how MesoNet performs on different deepfake generation methods!

Good luck! 🚀