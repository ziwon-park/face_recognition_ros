import os
import cv2
import numpy as np
from PIL import Image

import face_recognition

import os
import cv2
import numpy as np
from PIL import Image
# import torch

import face_recognition
# from deepface import DeepFace
# from facenet_pytorch import MTCNN, InceptionResnetV1

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
    
    
def get_face_model(model_name):
    if model_name.lower() == 'dlib':
        return DlibFaceModel()
    else:
        raise ValueError("Invalid model name. Choose 'dlib' or 'deepface' or 'facenet-pytorch'.")