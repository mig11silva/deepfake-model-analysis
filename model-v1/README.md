# Deepfake Detection Analysis

Analyzes images using a HuggingFace deepfake detection model and generates metrics and visualizations.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
python main.py

# Run with limited dataset
python main.py --size 100
```

## Usage

```bash
python main.py              # Process all images
python main.py --size 50    # Process 50 fake + 50 real images
python main.py --help       # Show help
```

## Results

Results are saved to `results/`:
- `confusion_matrix.png` - True vs predicted labels
- `metrics_bar_chart.png` - Accuracy, Precision, Recall, F1
- `prediction_distribution.png` - Count of fake/real predictions
- `confidence_distribution.png` - Histogram of confidence scores
- `roc_curve.png` - ROC curve with AUC score

## Project Structure

```
├── main.py              # Entry point
├── core/
│   ├── model_runner.py  # Model loading and inference
│   ├── image_loader.py  # Dataset loading
│   └── results_generator.py  # Metrics & visualizations
├── dataset/
│   ├── fake/            # Fake images
│   └── real/            # Real images
└── results/             # Generated output
```

## Requirements

- Python 3.8+
- PyTorch
- HuggingFace Transformers
- See `requirements.txt`
