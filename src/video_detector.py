import cv2
import os
import torch
from tqdm import tqdm
from utils import preprocess_image, postprocess_detections, draw_boxes


class VideoDetector:
    def __init__(self, model, device, confidence_threshold=0.5):
        """
        Initialize Video Detector
        
        Args:
            model: Object detection model (Faster R-CNN)
            device: torch device
            confidence_threshold: Minimum confidence for detections
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
    
    def detect_video(self, video_path, output_path=None, skip_frames=1):
        """
        Detect objects in a video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            skip_frames: Process every nth frame (1 = all frames)
        
        Returns:
            Number of frames processed
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        last_detections = None
        
        # Process video
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame
                if frame_count % skip_frames == 0:
                    try:
                        # Debug: Check frame shape
                        if frame_count == 0:
                            print(f"First frame shape: {frame.shape}, dtype: {frame.dtype}")
                        
                        # Preprocess
                        img_tensor = preprocess_image(frame).to(self.device)
                        
                        # Debug: Check tensor shape
                        if frame_count == 0:
                            print(f"Tensor shape after preprocessing: {img_tensor.shape}")
                        
                        # Inference
                        with torch.no_grad():
                            outputs = self.model([img_tensor])
                        
                        # Post-process
                        detections = postprocess_detections(outputs, self.confidence_threshold)
                        last_detections = detections
                        
                        # Draw boxes
                        frame = draw_boxes(frame, detections, scale_x=1.0, scale_y=1.0)
                        processed_count += 1
                        
                    except Exception as e:
                        print(f"\nError processing frame {frame_count}: {str(e)}")
                        print(f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                        if frame_count == 0:
                            raise  # Re-raise on first frame to see the full error
                        # Use last detections if available
                        if last_detections is not None:
                            frame = draw_boxes(frame, last_detections, scale_x=1.0, scale_y=1.0)
                else:
                    # Use last detections for skipped frames
                    if last_detections is not None:
                        frame = draw_boxes(frame, last_detections, scale_x=1.0, scale_y=1.0)
                
                # Write frame
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
            print(f"Video saved to {output_path}")
        
        print(f"Processed {processed_count} out of {frame_count} frames")
        return processed_count
    
    def detect_batch_videos(self, input_dir, output_dir, skip_frames=1):
        """
        Detect objects in all videos in a directory
        
        Args:
            input_dir: Directory containing input videos
            output_dir: Directory to save output videos
            skip_frames: Process every nth frame
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = [f for f in os.listdir(input_dir) 
                      if os.path.splitext(f)[1].lower() in video_extensions]
        
        if not video_files:
            print(f"No videos found in {input_dir}")
            return
        
        print(f"Found {len(video_files)} videos to process")
        
        # Process each video
        for vid_file in video_files:
            input_path = os.path.join(input_dir, vid_file)
            output_filename = f"{os.path.splitext(vid_file)[0]}_detected.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                print(f"\nProcessing: {vid_file}")
                self.detect_video(input_path, output_path, skip_frames)
            except Exception as e:
                print(f"Error processing {vid_file}: {str(e)}")
        
        print(f"\nBatch processing completed! Results saved to {output_dir}")


if __name__ == "__main__":
    from model_loader import EfficientDetLoader
    
    # Load model
    loader = EfficientDetLoader()
    model = loader.load_saved_model()
    device = loader.get_device()
    
    # Create detector
    detector = VideoDetector(model, device, confidence_threshold=0.5)
    
    # Detect in batch
    detector.detect_batch_videos(
        input_dir='data/test_videos',
        output_dir='output/videos',
        skip_frames=2  # Process every 2nd frame for speed
    )