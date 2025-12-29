"""
YOLOv8 Integration for Eastern Services Chatbot
Replace LLaVA/Gemini with custom trained YOLOv8 model
"""

from ultralytics import YOLO
from PIL import Image
import io
import base64
from typing import Dict, Optional

class PestDetector:
    """Custom YOLOv8 pest detection model"""
    
    def __init__(self, model_path: str = 'models/best.pt'):
        """
        Initialize YOLOv8 model
        
        Args:
            model_path: Path to trained YOLOv8 model
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = ['mosquito', 'termite', 'cockroach', 'rodent']
            print(f"[OK] YOLOv8 model loaded from {model_path}")
        except Exception as e:
            print(f"[FAIL] Failed to load YOLOv8 model: {e}")
            self.model = None
    
    def detect_pest(self, image_data: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Detect pests in image
        
        Args:
            image_data: Base64 encoded image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with detection results
        """
        if not self.model:
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Decode image
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Run detection
            results = self.model(image, conf=confidence_threshold)
            
            # Extract detections
            detections = results[0].boxes
            
            if len(detections) == 0:
                return {
                    'success': True,
                    'pest_detected': False,
                    'message': 'No pests detected in the image'
                }
            
            # Get best detection (highest confidence)
            best_det = detections[0]
            class_id = int(best_det.cls[0])
            confidence = float(best_det.conf[0])
            pest_name = self.class_names[class_id]
            
            # Get bounding box
            box = best_det.xyxy[0].tolist()
            
            return {
                'success': True,
                'pest_detected': True,
                'pest_name': pest_name,
                'confidence': confidence,
                'bounding_box': {
                    'x1': box[0],
                    'y1': box[1],
                    'x2': box[2],
                    'y2': box[3]
                },
                'total_detections': len(detections),
                'message': f"Detected {pest_name} with {confidence:.1%} confidence"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def is_available(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None


# Global instance
pest_detector = None

def init_pest_detector(model_path: str = 'models/best.pt'):
    """Initialize global pest detector"""
    global pest_detector
    pest_detector = PestDetector(model_path)
    return pest_detector.is_available()
