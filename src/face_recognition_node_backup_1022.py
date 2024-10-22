#!/home/nuc/anaconda3/envs/ziwon/bin/python3

import os
import numpy as np
import cv2
import time
import psutil
import GPUtil
from threading import Thread

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Vector3
from face_model import get_face_model

class ResourceMonitor(Thread):
    def __init__(self, interval=3):
        Thread.__init__(self)
        self.interval = interval
        self.stopped = False
        self.process = psutil.Process(os.getpid())

        
    def run(self):
        while not self.stopped:
            # CPU 사용량
            cpu_usage = self.process.cpu_percent(interval=1)
            
            # 메모리 사용량
            memory_info = self.process.memory_info()
            memory_usage = memory_info.rss / psutil.virtual_memory().total * 100
            
            # GPU 사용량
            gpu_usage = self.get_gpu_usage()
            
            print(f"Process CPU Usage: {cpu_usage:.2f}%, "
                  f"Process Memory Usage: {memory_usage:.2f}%, "
                  f"Process GPU Usage: {gpu_usage:.2f}%")
            
            time.sleep(self.interval)
            
    def get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                import subprocess
                result = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,nounits,noheader'])
                for line in result.decode().strip().split('\n'):
                    pid, used_memory = map(int, line.split(','))
                    if pid == self.process.pid:
                        return (used_memory / gpus[0].memoryTotal) * 100
            return 0
        except Exception as e:
            print(f"Error getting GPU usage: {e}")
            return 0

    def stop(self):
        self.stopped = True

class FaceRecognition:
    def __init__(self, model_name='dlib'):
        rospy.init_node('face_recognition_node', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/go1_d435/color/image_raw', Image, self.image_callback, queue_size=1)
        self.face_pub = rospy.Publisher('/face', Image, queue_size=10)

        self.face_model = get_face_model(model_name)
        
        # self.target_face_folder = "/home/nuc/ros/face_ws/src/face_recognition/face/Myung"

        self.target_face_folder = rospy.get_param('~target_face_folder', 
            default="/home/nuc/ros/face_ws/src/face_recognition/face/Myung")

        self.target_person_name = os.path.basename(os.path.normpath(self.target_face_folder))
        self.face_model.load_target_face(self.target_face_folder)
        
        self.scale_factor = 0.5
        
        self.frame_times = []
        self.fps_update_interval = 100
        
        self.image_center_x = 0
        self.image_center_y = 0
        
        self.face_distance = None
        
        # for PID Control
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0
        
        # Face tracking
        self.face_tracker = cv2.TrackerKCF_create()
        self.tracking_face = False
        
        # resource usage monitoring
        self.resource_monitor = ResourceMonitor()
        self.resource_monitor.start()
        
        rospy.loginfo(f"Face recognition node initiated with {model_name} model")
        
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
        # os.system("clear")
        # print("inside image_callback")
        start_time = time.time()
        
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')
        except CvBridgeError as e:
            print(e)
            
        original_size = cv_image.shape[:2]
        
        new_height = int(original_size[0] * self.scale_factor)
        new_width = int(original_size[1] * self.scale_factor)
        small_frame = cv2.resize(cv_image, (new_width, new_height))
        resized_image = cv2.cvtColor(small_frame, cv2.COLOR_RGB2BGR)
        
        resized_height, resized_width = resized_image.shape[:2]
        self.image_center_y, self.image_center_x = resized_height // 2, resized_width // 2
        
        cv2.circle(resized_image, (self.image_center_x, self.image_center_y), 5, (0, 255, 255), -1)
        
        target_face_found = False
        is_match = False
        
        if self.tracking_face:
            success, box = self.face_tracker.update(resized_image)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_center_x = x + w // 2
                face_center_y = y + h // 2
                target_face_found = True
                
            else:
                self.tracking_face = False
        
        if not self.tracking_face:
            face_locations = self.face_model.detect_faces(resized_image)
            for face_location in face_locations:
                if isinstance(face_location, dict):  # DeepFace format
                    x, y, w, h = face_location['x'], face_location['y'], face_location['w'], face_location['h']
                elif len(face_location) == 4:  # dlib format
                    print("inside dlib format")
                    top, right, bottom, left = face_location
                    y, x, h, w = top, left, bottom - top, right - left
                else:  # FaceNet-PyTorch format
                    x, y, w, h = map(int, face_location)

                if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > resized_width or y+h > resized_height:
                    continue 

                face_img = resized_image[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue


                # is_target_face = self.face_model.recognize_face(face_img)
                is_match, self.face_distance = self.face_model.recognize_face(face_img)

                # face_encoding = self.face_model.recognize_face(resized_image[y:y+h, x:x+w])
                
                # if is_target_face:
                if is_match:
                    target_face_found = True
                    cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(resized_image, self.target_person_name, 
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    
                    self.face_tracker = cv2.TrackerCSRT_create()
                    self.face_tracker.init(resized_image, (x, y, w, h))
                    self.tracking_face = True
                    break
        
        if target_face_found:
            cv2.line(resized_image, (self.image_center_x, self.image_center_y), 
                     (face_center_x, face_center_y), (255, 0, 0), 2)
            # distance = np.sqrt((face_center_x - self.image_center_x)**2 + 
            #                    (face_center_y - self.image_center_y)**2)
            # cv2.putText(resized_image, f"Dist: {distance:.2f}", 
            #             (10, resized_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            #             0.5, (255, 255, 255), 1)
            
            distance_text = f"Embedding Dist: {self.face_distance:.4f}" if self.face_distance is not None else "Dist: N/A"
            cv2.putText(resized_image, f"{distance_text}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            roll, pitch = self.calculate_robot_control(face_center_x / self.scale_factor, face_center_y / self.scale_factor)
            rospy.loginfo(f"Target person found. Roll: {roll:.4f}, Pitch: {pitch:.4f}")
        else:
            rospy.loginfo("Target person not found in the image")

        try:
            self.face_pub.publish(self.bridge.cv2_to_imgmsg(resized_image, 'bgr8'))
        except CvBridgeError as e:
            rospy.logerr(e)

        end_time = time.time()
        self.frame_times.append(end_time)
        if len(self.frame_times) > self.fps_update_interval:
            self.frame_times.pop(0)
        
        if len(self.frame_times) % self.fps_update_interval == 0:
            fps = self.calculate_fps()
            rospy.loginfo(f"Current FPS: {fps:.2f}")

        processing_time = end_time - start_time
        # rospy.loginfo(f"Frame processing time: {processing_time:.4f} seconds")

    def run(self):
        try:
            rospy.spin()
        finally:
            self.resource_monitor.stop()
            self.resource_monitor.join()
            
if __name__ == '__main__':
    try:
        face_recognition_module = FaceRecognition(model_name='dlib') 
        face_recognition_module.run()
    except rospy.ROSInterruptException:
        pass