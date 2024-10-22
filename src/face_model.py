import os
import cv2
import numpy as np
from PIL import Image
import torch

import face_recognition
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceModelBase:
    def __init__(self):
        self.target_face_encodings = []

    def load_target_face(self, folder_path):
        raise NotImplementedError

    def detect_faces(self, image):
        raise NotImplementedError

    def recognize_face(self, face_encoding):
        raise NotImplementedError

class DlibFaceModel(FaceModelBase):
    def __init__(self):
        super().__init__()
        print("Dlib model initiated.")
    
    def load_target_face(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                known_image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(known_image)
                
                if face_encoding:
                    self.target_face_encodings.append(face_encoding[0])
                    print(f"Loaded known face from: {image_path}")

    def detect_faces(self, image):
        return face_recognition.face_locations(image, model="hog")

    def recognize_face(self, face_image):
        if face_image is None or face_image.size == 0:
            return False, None
        
        try:
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            else:
                return False, None
            
            face_encodings = face_recognition.face_encodings(rgb_image, model="large")
            
            if not face_encodings:
                return False, None
            
            min_distance = float('inf')
            
            for face_encoding in face_encodings:
                distances = face_recognition.face_distance(self.target_face_encodings, face_encoding)
                if distances.size > 0:
                    current_min_distance = np.min(distances)
                    if current_min_distance < min_distance:
                        min_distance = current_min_distance
               
            is_match = min_distance <= 0.4
            return is_match, min_distance

                # matches = face_recognition.compare_faces(self.target_face_encodings, face_encoding, tolerance=0.4)
                # if True in matches:
                #     return True
            # return False
        except Exception as e:
            print(f"Error in recognize_face: {str(e)}")
            return False, None
    
class FacenetPytorchModel(FaceModelBase):
    def __init__(self):
        super().__init__()
        print("Facenet Pytorch model initiated.")
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def load_target_face(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                img = Image.open(image_path)
                face = self.mtcnn(img)
                if face is not None:
                    embedding = self.resnet(face.to(self.device)).detach().cpu().numpy()[0]
                    self.target_face_encodings.append(embedding)
                else:
                    print(f"No face found in the image: {image_path}")

    def detect_faces(self, image):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes, _ = self.mtcnn.detect(img)
        if boxes is None:
            return []
        return boxes.astype(int)

    def recognize_face(self, face_image):
        if face_image is None or face_image.size == 0:
            return False
        
        try:
            img = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            face = self.mtcnn(img)
            if face is None:
                return False
            embedding = self.resnet(face.to(self.device)).detach().cpu().numpy()[0]
            similarities = [np.dot(embedding, target_emb) / (np.linalg.norm(embedding) * np.linalg.norm(target_emb)) 
                            for target_emb in self.target_face_encodings]
            max_similarity = max(similarities) if similarities else 0
            return max_similarity > 0.6 
        except Exception as e:
            print(f"Error in recognize_face: {str(e)}")
            return False
        
class DeepFaceModel(FaceModelBase):
    def __init__(self):
        super().__init__()
        print("DeepFace model initiated.")
        
        self.target_face_paths = []
        self.model_name = "VGG-Face" 
        self.distance_metric = "cosine"
        self.detector_backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface"]
        self.detector_backend = "retinaface"
        self.current_backend_index = 0
            
    def load_target_face(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                self.target_face_paths.append(image_path)
                print(f"Loaded known face from: {image_path}")
                
    def detect_faces(self, image):
        print("inside detect_faces")
        faces = DeepFace.extract_faces(image, detector_backend=self.detector_backend)
        print("faces is ",faces)
        return [face['facial_area'] for face in faces]

    def recognize_face(self, face_image):
        if face_image is None or face_image.size == 0:
            return False

        try:
            for target_face_path in self.target_face_paths:
                result = DeepFace.verify(face_image, target_face_path, 
                                         model_name=self.model_name, 
                                         distance_metric=self.distance_metric,
                                         detector_backend=self.detector_backend)
                
                if result['verified']:
                    return True
            
            return False
        except Exception as e:
            print(f"Error in recognize_face: {str(e)}")
            return False

    def get_face_location(self, image):
        try:
            for target_face_path in self.target_face_paths:
                result = DeepFace.verify(image, target_face_path, 
                                         model_name=self.model_name, 
                                         distance_metric=self.distance_metric,
                                         detector_backend=self.detector_backend)
                
                if result['verified']:
                    facial_area = result['facial_areas']['img1']
                    return {
                        'x': facial_area['x'],
                        'y': facial_area['y'],
                        'w': facial_area['w'],
                        'h': facial_area['h']
                    }
            
            return None
        except Exception as e:
            print(f"Error in get_face_location: {str(e)}")
            return None
    
def get_face_model(model_name):
    if model_name.lower() == 'dlib':
        return DlibFaceModel()
    elif model_name.lower() == 'deepface':
        return DeepFaceModel()
    elif model_name.lower() == 'facenet-pytorch':
        return FacenetPytorchModel()
    else:
        raise ValueError("Invalid model name. Choose 'dlib' or 'deepface' or 'facenet-pytorch'.")