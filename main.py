import utils.session
import os
import sys
import argparse

# Add src to path
sys.path.append('src')

from src.model_loader import EfficientDetLoader
from src.image_detector import ImageDetector
from src.video_detector import VideoDetector


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'models/saved_model',
        'data/test_images',
        'data/test_videos',
        'output/images',
        'output/videos'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure created successfully!")


def download_and_save_model(model_name='tf_efficientdet_d0'):
    """Download pretrained model and save it"""
    print("\n=== Step 1: Loading and Saving Pretrained Model ===")
    
    loader = EfficientDetLoader(model_name=model_name)
    model = loader.load_pretrained_model()
    
    save_path = f'models/saved_model/{model_name}.pth'
    loader.save_model(save_path)
    
    print(f"\nModel saved successfully to {save_path}")
    return loader


def test_on_images(loader, confidence_threshold=0.5):
    """Test model on images"""
    print("\n=== Step 2: Testing on Images ===")
    
    input_dir = 'data/test_images'
    output_dir = 'output/images'
    
    # Check if there are images to process
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        print("Please add some test images to the data/test_images/ folder")
        return
    
    # Create detector
    detector = ImageDetector(
        loader.get_model(), 
        loader.get_device(), 
        confidence_threshold=confidence_threshold
    )
    
    # Process images
    detector.detect_batch_images(input_dir, output_dir)
    print(f"\nImage detection completed! Check {output_dir} for results")


def test_on_videos(loader, confidence_threshold=0.5, skip_frames=2):
    """Test model on videos"""
    print("\n=== Step 3: Testing on Videos ===")
    
    input_dir = 'data/test_videos'
    output_dir = 'output/videos'
    
    # Check if there are videos to process
    video_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No videos found in {input_dir}")
        print("Please add some test videos to the data/test_videos/ folder")
        return
    
    # Create detector
    detector = VideoDetector(
        loader.get_model(), 
        loader.get_device(), 
        confidence_threshold=confidence_threshold
    )
    
    # Process videos
    detector.detect_batch_videos(input_dir, output_dir, skip_frames=skip_frames)
    print(f"\nVideo detection completed! Check {output_dir} for results")


def main():
    parser = argparse.ArgumentParser(description='EfficientDet Object Detection')
    parser.add_argument('--model', type=str, default='tf_efficientdet_d0',
                       help='Model name (d0-d7)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--skip-frames', type=int, default=2,
                       help='Process every nth frame in videos')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip model download and use existing model')
    parser.add_argument('--images-only', action='store_true',
                       help='Process only images')
    parser.add_argument('--videos-only', action='store_true',
                       help='Process only videos')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("EfficientDet Object Detection Pipeline")
    print("=" * 60)
    
    # Setup directories
    setup_directories()
    
    # Load or download model
    if args.skip_download:
        print("\n=== Loading Existing Model ===")
        loader = EfficientDetLoader(model_name=args.model)
        loader.load_saved_model(f'models/saved_model/{args.model}.pth')
    else:
        loader = download_and_save_model(args.model)
    
    # Process based on flags
    if args.videos_only:
        test_on_videos(loader, args.confidence, args.skip_frames)
    elif args.images_only:
        test_on_images(loader, args.confidence)
    else:
        # Process both
        test_on_images(loader, args.confidence)
        test_on_videos(loader, args.confidence, args.skip_frames)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()