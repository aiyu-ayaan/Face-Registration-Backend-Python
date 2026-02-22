import cv2
import numpy as np
from PIL import Image
import io
import os
import urllib.request
from typing import Optional, Tuple, List


class FaceRecognitionService:
    """Service for handling face recognition operations using OpenCV"""
    
    # Cosine similarity threshold for matching (higher = more strict)
    COSINE_THRESHOLD = 0.363
    
    # Model files
    MODELS_DIR = "models"
    FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
    FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
    
    # URLs for downloading models
    DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    
    _face_recognizer = None
    
    @classmethod
    def _ensure_models_dir(cls):
        """Ensure the models directory exists"""
        if not os.path.exists(cls.MODELS_DIR):
            os.makedirs(cls.MODELS_DIR)
    
    @classmethod
    def _download_file(cls, url: str, filepath: str) -> bool:
        """Download a file from URL"""
        try:
            print(f"Downloading {os.path.basename(filepath)}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded {os.path.basename(filepath)} ({os.path.getsize(filepath)} bytes)")
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    @classmethod
    def _get_face_detector(cls, image_width: int, image_height: int):
        """Get a new face detector sized for the given image dimensions"""
        cls._ensure_models_dir()
        
        model_path = os.path.join(cls.MODELS_DIR, cls.FACE_DETECTOR_MODEL)
        
        # Download if not exists
        if not os.path.exists(model_path):
            cls._download_file(cls.DETECTOR_URL, model_path)
        
        detector = cv2.FaceDetectorYN.create(
            model_path, 
            "", 
            (image_width, image_height),
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=5000
        )
        
        return detector
    
    @classmethod
    def _get_face_recognizer(cls):
        """Get or initialize the face recognizer"""
        if cls._face_recognizer is None:
            cls._ensure_models_dir()
            
            model_path = os.path.join(cls.MODELS_DIR, cls.FACE_RECOGNIZER_MODEL)
            
            # Download if not exists
            if not os.path.exists(model_path):
                cls._download_file(cls.RECOGNIZER_URL, model_path)
            
            cls._face_recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        
        return cls._face_recognizer
    
    @classmethod
    def _detect_face(cls, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face in image using YuNet detector.
        
        Returns:
            Face detection result (numpy row) or None if no face found.
            Each row contains [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
        """
        h, w = image.shape[:2]
        detector = cls._get_face_detector(w, h)
        
        _, faces = detector.detect(image)
        
        if faces is None or len(faces) == 0:
            return None
        
        # Pick the face with the highest confidence
        best_idx = np.argmax(faces[:, -1])
        return faces[best_idx]
    
    @classmethod
    def extract_face_encoding(cls, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract face encoding from an image.
        
        Args:
            image_bytes: Raw bytes of the image file
            
        Returns:
            Face encoding as numpy array, or None if no face found
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to OpenCV format (BGR)
            image_array = np.array(image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            print(f"[FaceService] Image shape: {image_bgr.shape}")
            
            # Detect face using YuNet
            face = cls._detect_face(image_bgr)
            
            if face is None:
                print("[FaceService] No face detected by YuNet")
                return None
            
            print(f"[FaceService] Face detected with confidence: {face[-1]:.4f}")
            
            # Get face recognizer
            recognizer = cls._get_face_recognizer()
            
            # Align and crop face using the detection result
            face_aligned = recognizer.alignCrop(image_bgr, face)
            
            # Get face feature/embedding
            face_feature = recognizer.feature(face_aligned)
            
            return face_feature.flatten()
            
        except Exception as e:
            print(f"[FaceService] Error extracting face encoding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def encoding_to_bytes(encoding: np.ndarray) -> bytes:
        """Convert numpy array encoding to bytes for database storage"""
        return encoding.astype(np.float32).tobytes()
    
    @staticmethod
    def bytes_to_encoding(encoding_bytes: bytes) -> np.ndarray:
        """Convert bytes back to numpy array encoding"""
        return np.frombuffer(encoding_bytes, dtype=np.float32)
    
    @classmethod
    def calculate_similarity(cls, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face encodings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Cosine similarity (1 = identical, 0 = completely different)
        """
        recognizer = cls._get_face_recognizer()
        
        score = recognizer.match(
            encoding1.reshape(1, -1).astype(np.float32), 
            encoding2.reshape(1, -1).astype(np.float32),
            cv2.FaceRecognizerSF_FR_COSINE
        )
        
        return float(score)
    
    @classmethod
    def compare_faces(cls, known_encoding: np.ndarray, unknown_encoding: np.ndarray) -> Tuple[bool, float]:
        """
        Compare two face encodings.
        
        Returns:
            Tuple of (is_match, similarity_score)
        """
        similarity = cls.calculate_similarity(known_encoding, unknown_encoding)
        is_match = similarity >= cls.COSINE_THRESHOLD
        
        return is_match, similarity
    
    @classmethod
    def find_best_match(cls, unknown_encoding: np.ndarray, 
                        known_encodings: List[Tuple[int, np.ndarray]]) -> Optional[Tuple[int, float]]:
        """
        Find the best matching face from a list of known encodings.
        
        Returns:
            Tuple of (user_id, confidence%) or None if no match found
        """
        if not known_encodings:
            return None
        
        best_match_id = None
        best_similarity = -1.0
        
        for user_id, encoding in known_encodings:
            is_match, similarity = cls.compare_faces(encoding, unknown_encoding)
            
            if is_match and similarity > best_similarity:
                best_match_id = user_id
                best_similarity = similarity
        
        if best_match_id is not None:
            # Convert similarity to confidence percentage
            confidence = best_similarity * 100
            return best_match_id, min(confidence, 100)
        
        return None
