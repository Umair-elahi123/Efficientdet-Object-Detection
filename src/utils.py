import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


# COCO class names (91 classes including background at index 0)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def preprocess_image(image):
    """
    Preprocess image for Faster R-CNN
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Preprocessed tensor in [C, H, W] format
    """
    if isinstance(image, np.ndarray):
        # Check if image is grayscale (2D) and convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is contiguous and correct dtype
        image = np.ascontiguousarray(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        # Transpose from (H, W, C) to (C, H, W)
        image = image.transpose((2, 0, 1))
        
        # Convert to torch tensor
        tensor = torch.from_numpy(image).float()
    else:
        # If PIL Image
        transform = transforms.ToTensor()
        tensor = transform(image)
    
    return tensor


def postprocess_detections(outputs, confidence_threshold=0.5):
    """
    Post-process Faster R-CNN outputs
    
    Args:
        outputs: Model outputs (list of dicts)
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Filtered detections as numpy array
    """
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    
    # Filter by confidence
    mask = scores > confidence_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]
    
    # Combine into single array: [x1, y1, x2, y2, score, class]
    detections = np.column_stack([boxes, scores.reshape(-1, 1), labels.reshape(-1, 1)])
    
    return detections


def draw_boxes(image, detections, scale_x=1.0, scale_y=1.0):
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image (numpy array)
        detections: Detection results (Nx6 array)
        scale_x: Scale factor for x-axis
        scale_y: Scale factor for y-axis
    
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        
        # Coordinates are already in original image space for Faster R-CNN
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Get class name
        class_id = int(cls)
        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"Class {class_id}"
        
        # Skip N/A classes
        if class_name == 'N/A':
            continue
        
        # Generate color based on class
        color = get_color_for_class(class_id)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_name}: {conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Ensure label is within image bounds
        y1_label = max(y1, label_size[1] + 10)
        
        cv2.rectangle(img, 
                     (x1, y1_label - label_size[1] - 10), 
                     (x1 + label_size[0], y1_label), 
                     color, -1)
        cv2.putText(img, label, (x1, y1_label - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img


def get_color_for_class(class_id):
    """Generate a consistent color for each class"""
    np.random.seed(class_id)
    return tuple(np.random.randint(50, 255, 3).tolist())