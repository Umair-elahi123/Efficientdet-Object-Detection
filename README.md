# EfficientDet Object Detection Project

This project demonstrates how to use a pretrained EfficientDet model for object detection on images and videos.

## Project Structure

```
efficientdet_project/
│
├── models/saved_model/          # Saved model weights
├── data/
│   ├── test_images/             # Input images for testing
│   └── test_videos/             # Input videos for testing
├── output/
│   ├── images/                  # Detected images output
│   └── videos/                  # Detected videos output
├── src/
│   ├── model_loader.py          # Model loading and saving
│   ├── image_detector.py        # Image detection
│   ├── video_detector.py        # Video detection
│   └── utils.py                 # Utility functions
├── requirements.txt             # Dependencies
├── main.py                      # Main entry point
└── README.md                    # This file
```

## Installation

1. Clone or create the project directory:
```bash
mkdir efficientdet_project
cd efficientdet_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (All Steps)

Run the complete pipeline - download model, save it, and test on images and videos:

```bash
python main.py
```

### Options

```bash
# Use a different model (d0-d7)
python main.py --model tf_efficientdet_d1

# Set confidence threshold
python main.py --confidence 0.6

# Process every 3rd frame in videos (faster)
python main.py --skip-frames 3

# Skip model download (use existing model)
python main.py --skip-download

# Process only images
python main.py --images-only

# Process only videos
python main.py --videos-only
```

### Step-by-Step Usage

#### 1. Load and Save Model

```python
from src.model_loader import EfficientDetLoader

loader = EfficientDetLoader(model_name='tf_efficientdet_d0')
model = loader.load_pretrained_model()
loader.save_model('models/saved_model/efficientdet_d0.pth')
```

#### 2. Detect Objects in Images

```python
from src.image_detector import ImageDetector

detector = ImageDetector(model, device, confidence_threshold=0.5)

# Single image
detector.detect_single_image('data/test_images/image1.jpg', 
                             'output/images/image1_detected.jpg')

# Batch processing
detector.detect_batch_images('data/test_images', 'output/images')
```

#### 3. Detect Objects in Videos

```python
from src.video_detector import VideoDetector

detector = VideoDetector(model, device, confidence_threshold=0.5)

# Single video
detector.detect_video('data/test_videos/video1.mp4',
                     'output/videos/video1_detected.mp4',
                     skip_frames=2)

# Batch processing
detector.detect_batch_videos('data/test_videos', 'output/videos', skip_frames=2)
```

## Preparing Test Data

1. Place your test images in `data/test_images/`
   - Supported formats: .jpg, .jpeg, .png, .bmp, .tiff

2. Place your test videos in `data/test_videos/`
   - Supported formats: .mp4, .avi, .mov, .mkv, .flv

## Model Variants

EfficientDet comes in different sizes:
- `tf_efficientdet_d0` - Fastest, smallest
- `tf_efficientdet_d1` - Balanced
- `tf_efficientdet_d2` - Better accuracy
- `tf_efficientdet_d3` - Even better accuracy
- `tf_efficientdet_d4` - High accuracy
- `tf_efficientdet_d5` - Very high accuracy
- `tf_efficientdet_d6` - Highest accuracy
- `tf_efficientdet_d7` - Maximum accuracy, slowest

## Configuration

### Confidence Threshold
- Range: 0.0 to 1.0
- Default: 0.5
- Higher values = fewer but more confident detections

### Skip Frames (Videos)
- Default: 2 (process every 2nd frame)
- Higher values = faster processing but less smooth
- Use 1 for best quality (process every frame)

## Output

- **Images**: Saved with `_detected` suffix in `output/images/`
- **Videos**: Saved with `_detected` suffix in `output/videos/`
- Bounding boxes show detected objects with class labels and confidence scores

## Troubleshooting

### CUDA Out of Memory
- Use a smaller model (d0 instead of d7)
- Increase `skip_frames` for videos
- Process fewer images at once

### Slow Processing
- Increase `skip_frames` for videos
- Use GPU if available (automatically detected)
- Use a smaller model variant

### No Detections
- Lower the confidence threshold (--confidence 0.3)
- Check if objects are in COCO dataset classes
- Ensure good image/video quality

## COCO Classes

The model detects 90 object classes from the COCO dataset, including:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, 
bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, 
handbag, bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, 
pizza, donut, cake, chair, couch, bed, dining table, toilet, tv, laptop, mouse, 
keyboard, cell phone, book, clock, vase, scissors, teddy bear, and more.

## License

This project uses pretrained models from the EfficientDet repository. Please refer to their respective licenses.

## Credits

- EfficientDet implementation: [rwightman/efficientdet-pytorch](https://github.com/rwightman/efficientdet-pytorch)
- Original paper: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)