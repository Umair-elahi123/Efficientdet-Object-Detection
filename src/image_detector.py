import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
from utils import preprocess_image, postprocess_detections, draw_boxes


class ImageDetector:
    def __init__(self, model, device, confidence_threshold=0.5):
        """
        Initialize Image Detector
        
        Args:
            model: Object detection model (Faster R-CNN)
            device: torch device
            confidence_threshold: Minimum confidence for detections
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
    
    def detect_single_image(self, image_path, output_path=None):
        """
        Detect objects in a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
        
        Returns:
            Image with detections and detection results
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        orig_h, orig_w = img.shape[:2]
        
        # Preprocess
        img_tensor = preprocess_image(img).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model([img_tensor])
        
        # Post-process
        detections = postprocess_detections(outputs, self.confidence_threshold)
        
        # Draw boxes (no scaling needed for Faster R-CNN)
        result_img = draw_boxes(img, detections, scale_x=1.0, scale_y=1.0)
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, result_img)
            print(f"Saved detection result to {output_path}")
        
        return result_img, detections
    
    def detect_batch_images(self, input_dir, output_dir):
        """
        Detect objects in all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        for img_file in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(input_dir, img_file)
            output_filename = f"{os.path.splitext(img_file)[0]}_detected{os.path.splitext(img_file)[1]}"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                _, detections = self.detect_single_image(input_path, output_path)
                print(f"  {img_file}: Found {len(detections)} objects")
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
        
        print(f"Batch processing completed! Results saved to {output_dir}")


if __name__ == "__main__":
    from model_loader import EfficientDetLoader
    
    # Load model
    loader = EfficientDetLoader()
    model = loader.load_saved_model()
    device = loader.get_device()
    
    # Create detector
    detector = ImageDetector(model, device, confidence_threshold=0.5)
    
    # Detect in batch
    detector.detect_batch_images(
        input_dir='data/test_images',
        output_dir='output/images'
    )