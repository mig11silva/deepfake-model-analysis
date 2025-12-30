# MesoNet Deepfake Detection Test

Test implementation of MesoNet models on FaceForensics++ dataset samples.

This project uses the MesoNet architecture (Meso4 and MesoInception4) to detect deepfakes in videos. It supports two specific manipulation datasets from FaceForensics++: Deepfakes and Face2Face.

## Features
- **Standalone Project**: Fully self-contained environment.
- **Dual Dataset Support**: dedicated support for Deepfakes and Face2Face.
- **Academic Visualizations**: Generates Confusion Matrices, ROC Curves, and more.
- **Easy Setup**: Automated setup script for dependencies and model weights.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Add dataset videos**:
   > **Note**: You need to download the FaceForensics++ dataset manually.
   
   - **Deepfakes**: Add 10 real + 10 fake videos to `dataset/deepfakes/`
     - Real: `dataset/deepfakes/real/`
     - Fake: `dataset/deepfakes/fake/`
   - **Face2Face**: Add 10 real + 10 fake videos to `dataset/face2face/`
     - Real: `dataset/face2face/real/`
     - Fake: `dataset/face2face/fake/`

## Usage

### Test with Deepfakes dataset
Uses `Meso4_DF.h5` weights.
```bash
python main.py --dataset deepfakes
```

### Test with Face2Face dataset
Uses `Meso4_F2F.h5` weights.
```bash
python main.py --dataset face2face
```

### Advanced options
```bash
# Extract 15 frames instead of 10
python main.py --dataset deepfakes --frames 15

# Use Inception model (requires MesoInception_DF.h5)
python main.py --dataset deepfakes --model inception

# Custom output directory
python main.py --dataset face2face --output my_results/
```

Results will be saved to `results/{dataset_type}/`.

## Models

- **Meso4**: Lightweight CNN (4 layers). Good balance of speed and accuracy.
- **MesoInception4**: Inception-based variant (more sophisticated features).

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

## Citation
```
@inproceedings{afchar2018mesonet,
  title={Mesonet: a compact facial video forgery detection network},
  author={Afchar, Darius and Nozick, Vincent and Yamagishi, Junichi and Echizen, Isao},
  booktitle={IEEE WIFS},
  year={2018}
}
```
