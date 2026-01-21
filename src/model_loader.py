import torch
import os
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class EfficientDetLoader:
    def __init__(self, model_name='fasterrcnn_resnet50', num_classes=91):
        """
        Initialize object detection model loader
        Uses Faster R-CNN as it's readily available and works well
        
        Args:
            model_name: Model architecture
            num_classes: Number of classes (COCO has 91 including background)
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_pretrained_model(self):
        """Load pretrained Faster R-CNN model"""
        print(f"Loading {self.model_name} pretrained model...")
        
        # Load pretrained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        
        # The model already has 91 classes (COCO), so we can use it directly
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Model type: Faster R-CNN with ResNet50 backbone")
        
        return self.model
    
    def save_model(self, save_path='models/saved_model/efficientdet_d0.pth'):
        """
        Save the model to disk
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_pretrained_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_saved_model(self, load_path='models/saved_model/efficientdet_d0.pth'):
        """
        Load a previously saved model
        
        Args:
            load_path: Path to the saved model
        """
        print(f"Loading saved model from {load_path}...")
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Create model architecture
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        
        # Load state dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Saved model loaded successfully on {self.device}")
        return self.model
    
    def get_model(self):
        """Return the model instance"""
        return self.model
    
    def get_device(self):
        """Return the device"""
        return self.device


if __name__ == "__main__":
    # Example usage
    loader = EfficientDetLoader()
    model = loader.load_pretrained_model()
    loader.save_model()
    print("Model loading and saving completed!")