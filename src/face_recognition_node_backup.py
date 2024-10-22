#!/usr/bin/python3

import os
import numpy as np
import cv2
import face_recognition
import time

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Vector3

class FaceRecognition:
    def __init__(self):
        rospy.init_node('face_recognition_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        self.face_pub = rospy.Publisher('/face', Image, queue_size=10)

        self.target_face_encodings = []
        self.target_person_name = "Jiwon"
        self.load_target_face("/home/urllaptop/ziwon/face_recognition/jiwon")
        
        self.frame_times = []
        self.fps_update_interval = 100
        
        self.image_center_x = 0
        self.image_center_y = 0
        
        # for PID Control
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0
                
        rospy.loginfo(f"Face recognition node initiated")
        
    def load_target_face(self, folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, filename)
                known_image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(known_image)
                
                if face_encoding:
                    self.target_face_encodings.append(face_encoding[0])
                    rospy.loginfo(f"Loaded known face from: {image_path}")
                else:
                    rospy.logwarn(f"No face found in the image: {image_path}")
        
        if not self.target_face_encodings:
            rospy.logerr("No valid face images found in the specified folder")
        else:
            rospy.loginfo(f"Loaded {len(self.target_face_encodings)} face encodings")
            
    def calculate_robot_control(self, face_center_x, face_center_y):
        Kp = 0.001  
        Ki = 0.0001  
        Kd = 0.0005  

        error_x = self.image_center_x - face_center_x
        error_y = self.image_center_y - face_center_y

        self.integral_x += error_x
        self.integral_y += error_y
        derivative_x = error_x - self.prev_error_x
        derivative_y = error_y - self.prev_error_y

        roll = Kp * error_x + Ki * self.integral_x + Kd * derivative_x
        pitch = Kp * error_y + Ki * self.integral_y + Kd * derivative_y

        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return roll, pitch
            
    def calculate_fps(self):
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            fps = len(self.frame_times) / time_diff
            return fps
        return 0

    def image_callback(self, data):
        start_time = time.time()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')
        except CvBridgeError as e:
            print(e)
            
        original_size = cv_image.shape[:2]

        # 이미지 전처리
        
        scale_factor = 0.1
        new_height = int(original_size[0] * scale_factor)
        new_width = int(original_size[1] * scale_factor)
        small_frame = cv2.resize(cv_image, (new_width, new_height))
        resized_image = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)
        
        resized_height, resized_width = resized_image.shape[:2]
        self.image_center_y, self.image_center_x = resized_height // 2, resized_width // 2
        print("Resized image shape is : ", resized_width, resized_height)       
        
        # 얼굴 검출 및 식별

        face_locations = face_recognition.face_locations(resized_image, model="hog")
        face_encodings = face_recognition.face_encodings(resized_image, face_locations)
        
        target_face_found = False
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.target_face_encodings, face_encoding)
            print(top, right, bottom, left)
            
            if True in matches:
                print("inside true")

                target_face_found = True
                
                face_center_x = (left + right) // 2
                face_center_y = (top + bottom) // 2
                
                roll, pitch = self.calculate_robot_control(face_center_x / scale_factor, face_center_y / scale_factor)

                control_msg = Vector3()
                control_msg.x = roll
                control_msg.y = pitch
                control_msg.z = 0  
                self.robot_control_pub.publish(control_msg)
            
                cv2.rectangle(resized_image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(resized_image, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(resized_image, self.target_person_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        if target_face_found:
            rospy.loginfo("Target person found in the image")
        else:
            rospy.loginfo("Target person not found in the image")

        try:
            self.face_pub.publish(self.bridge.cv2_to_imgmsg(resized_image, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)

        # FPS 계산
        end_time = time.time()
        self.frame_times.append(end_time)
        if len(self.frame_times) > self.fps_update_interval:
            self.frame_times.pop(0)
        
        if len(self.frame_times) % self.fps_update_interval == 0:
            fps = self.calculate_fps()
            rospy.loginfo(f"Current FPS: {fps:.2f}")

        processing_time = end_time - start_time
        rospy.loginfo(f"Frame processing time: {processing_time:.4f} seconds")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        face_recognition_module = FaceRecognition()
        face_recognition_module.run()
    except rospy.ROSInterruptException:
        pass