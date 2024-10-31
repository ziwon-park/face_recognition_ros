#!/home/urllaptop/anaconda3/envs/face-ros/bin/python

import os
import numpy as np
import cv2
import time

import rospy
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Float32
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Vector3

from face_model import get_face_model
from util import ResourceMonitor

class FaceRecognition:
    def __init__(self, model_name='dlib'):
        rospy.init_node('face_recognition_node', anonymous=True)
        self.bridge = CvBridge()
        
        # Parameters
        self.depth_roi = rospy.get_param('~depth_roi', default=False)
        self.scale_factor = rospy.get_param('~scale_factor', default=0.5)
        self.target_face_folder = rospy.get_param('~target_face_folder', default="/home/nuc/ros/face_ws/src/face_recognition/face/Myung")
        self.resource_monitor = rospy.get_param('~resource_monitor', default=False)
        
        # Face tracking
        self.face_tracker = cv2.TrackerCSRT_create()
        self.tracking_face = False
    
        self.face_model = get_face_model(model_name)
        
        self.target_person_name = os.path.basename(os.path.normpath(self.target_face_folder))
        self.face_model.load_target_face(self.target_face_folder)
        
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

        # Camera calibration data
        self.camera_matrix = None
        self.latest_color_image = None
        self.latest_depth_image = None
        self.got_camera_info = False

        # Resource usage monitoring
        self.is_resource_monitored = False
        
        if self.resource_monitor:    
            self.is_resource_monitored = True    
            self.resource_monitor = ResourceMonitor()
            self.resource_monitor.start()
        
        # Subscriber
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        if self.depth_roi:
            self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback, queue_size=1)
            self.camera_info_sub = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        
        # Publisher
        self.face_pub = rospy.Publisher('/face', Image, queue_size=10)
        self.bbox_pub = rospy.Publisher('/face_bbox', Float32MultiArray, queue_size=10)
        if self.depth_roi:
            self.plane_marker_pub = rospy.Publisher('/face_plane', Marker, queue_size=10)

            # self.roi_cloud_pub = rospy.Publisher('/face_depth_roi', PointCloud2, queue_size=10)
            self.face_depth_pub = rospy.Publisher('/face_depth', Float32, queue_size=10)
        
        # Logger
        rospy.loginfo(f"Face recognition node initiated with {model_name} model")
        if self.depth_roi:
            rospy.loginfo("Depth ROI visualization enabled")
        
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
        
    def publish_bbox(self, x, y, w, h):
        bbox_msg = Float32MultiArray()
        bbox_msg.layout.dim = [MultiArrayDimension()]
        bbox_msg.layout.dim[0].label = "bbox"
        bbox_msg.layout.dim[0].size = 4
        bbox_msg.layout.dim[0].stride = 4
        
        x_orig = x / self.scale_factor
        y_orig = y / self.scale_factor
        w_orig = w / self.scale_factor
        h_orig = h / self.scale_factor
        
        center_x = x_orig + (w_orig / 2)
        center_y = y_orig + (h_orig / 2)
        
        bbox_msg.data = [float(center_x), float(center_y), float(w_orig), float(h_orig)]
        self.bbox_pub.publish(bbox_msg)
        
    def create_plane_marker(self, center_point, normal, size=0.2):
        # 평면 visualize
        
        marker = Marker()
        marker.header.frame_id = "camera_color_optical_frame"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "face_plane"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        z_axis = normal / np.linalg.norm(normal)
        x_axis = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(x_axis, z_axis)) > 0.9:
            x_axis = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        corners = [
            center_point + size * (-x_axis - y_axis),
            center_point + size * (x_axis - y_axis),
            center_point + size * (x_axis + y_axis),
            center_point + size * (-x_axis + y_axis)
        ]

        p1 = Point()
        p1.x, p1.y, p1.z = corners[0]
        p2 = Point()
        p2.x, p2.y, p2.z = corners[1]
        p3 = Point()
        p3.x, p3.y, p3.z = corners[2]
        p4 = Point()
        p4.x, p4.y, p4.z = corners[3]

        # 두 개의 삼각형으로 평면 구성
        marker.points.append(p1)
        marker.points.append(p2)
        marker.points.append(p3)

        marker.points.append(p1)
        marker.points.append(p3)
        marker.points.append(p4)

        return marker

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.got_camera_info = True
            rospy.loginfo("Received camera calibration data")
            
    def depth_callback(self, msg):
        try: 
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting depth image: {e}")

    def create_point_cloud(self, depth_roi, color_roi, roi_coords):
        if not self.got_camera_info:
            rospy.logwarn("No camera info available")
            return None
            
        try:
            x, y, w, h = roi_coords
            
            if depth_roi.shape != color_roi.shape[:2]:
                rospy.logerr(f"Mismatched ROI shapes: depth {depth_roi.shape}, color {color_roi.shape[:2]}")
                return None
                
            if w <= 0 or h <= 0:
                rospy.logwarn("Invalid ROI dimensions in create_point_cloud")
                return None
            
            height, width = depth_roi.shape
            sampling_step = 3
            
            max_points = (height // sampling_step) * (width // sampling_step)
            points_3d = np.empty((max_points, 3), dtype=np.float32)
            points_color = np.empty((max_points, 3), dtype=np.uint8)
            
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            u_coords, v_coords = np.meshgrid(
                np.arange(0, width, sampling_step),
                np.arange(0, height, sampling_step)
            )
            
            depths = depth_roi[v_coords, u_coords]
            valid_mask = depths > 0
            
            valid_depths = depths[valid_mask]
            valid_u = u_coords[valid_mask]
            valid_v = v_coords[valid_mask]
            
            z = valid_depths / 1000.0  # mm to meters
            x_3d = ((valid_u + x) - cx) * z / fx
            y_3d = ((valid_v + y) - cy) * z / fy
            
            n_valid_points = len(z)
            points_3d[:n_valid_points] = np.column_stack((x_3d, y_3d, z))
                        
            points_3d = points_3d[:n_valid_points]

            if len(points_3d) < 3:
                rospy.logwarn("Not enough points for plane estimation")
                return None

            # RANSAC 파라미터
            best_plane = None
            max_inliers = 0
            threshold = 0.02  # 20mm threshold
            n_iterations = 100
            
            for _ in range(n_iterations):
                idx = np.random.choice(n_valid_points, 3, replace=False)
                p1, p2, p3 = points_3d[idx]
                
                v1 = p2 - p1
                v2 = p3 - p1
                normal = np.cross(v1, v2)
                
                if np.all(normal == 0):
                    continue
                    
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, p1)
                
                distances = np.abs(points_3d @ normal + d)
                inliers = distances < threshold
                n_inliers = np.sum(inliers)
                
                if n_inliers > max_inliers:
                    max_inliers = n_inliers
                    best_plane = (normal, d)
                    best_inliers = inliers

            if best_plane is None:
                rospy.logwarn("Could not estimate plane")
                return None
                
            inlier_points = points_3d[best_inliers]
            center_point = np.mean(inlier_points, axis=0)
            
            normal, d = best_plane
            plane_marker = self.create_plane_marker(center_point, normal)
            self.plane_marker_pub.publish(plane_marker)
            
            plane_distance = abs(d)
            
            # 거리 발행
            distance_msg = Float32()
            distance_msg.data = plane_distance
            self.face_depth_pub.publish(distance_msg)
            
            # rospy.loginfo(f"Estimated plane: {normal[0]:.3f}x + {normal[1]:.3f}y + {normal[2]:.3f}z + {d:.3f} = 0")
            # rospy.loginfo(f"Number of inliers: {len(inlier_points)} / {n_valid_points}")
            
            return None
                
        except Exception as e:
            rospy.logerr(f"Error in create_point_cloud: {e}")
            return None
    
    def process_face_depth(self, face_location):
        if self.latest_depth_image is None or self.latest_color_image is None:
            return None, None

        x, y, w, h = face_location
        height, width = self.latest_depth_image.shape[:2]
            
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            rospy.logwarn("Invalid ROI dimensions")
            return None, None

        try:
            depth_roi = self.latest_depth_image[y:y+h, x:x+w].copy()
            color_roi = self.latest_color_image[y:y+h, x:x+w].copy()
            
            # Verify ROI sizes
            if depth_roi.size == 0 or color_roi.size == 0:
                rospy.logwarn("Empty ROI detected")
                return None, None
                
            valid_depths = depth_roi[depth_roi > 0]
            if len(valid_depths) > 0:
                avg_depth = float(np.mean(valid_depths)) / 1000.0  
            else:
                avg_depth = None

            point_cloud = self.create_point_cloud(depth_roi, color_roi, (x, y, w, h))

            return avg_depth, point_cloud
            
        except Exception as e:
            rospy.logerr(f"Error in process_face_depth: {e}")
            return None, None
    
    def image_callback(self, data): 
        # os.system("clear")
        # print("inside image_callback")
        start_time = time.time()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'rgb8')
            if self.depth_roi:
                self.latest_color_image = cv_image.copy()
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
                face_location = (int(x/self.scale_factor), int(y/self.scale_factor), 
                            int(w/self.scale_factor), int(h/self.scale_factor))
                
            else:
                self.tracking_face = False
        
        if not self.tracking_face:
            face_locations = self.face_model.detect_faces(resized_image)
            for face_location in face_locations:
                if isinstance(face_location, dict):  # DeepFace format
                    x, y, w, h = face_location['x'], face_location['y'], face_location['w'], face_location['h']
                elif len(face_location) == 4:  # dlib format
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
                    face_location = (int(x/self.scale_factor), int(y/self.scale_factor), 
                                int(w/self.scale_factor), int(h/self.scale_factor))
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
            
            # roll, pitch = self.calculate_robot_control(face_center_x / self.scale_factor, face_center_y / self.scale_factor)
            # rospy.loginfo(f"Target person found. Roll: {roll:.4f}, Pitch: {pitch:.4f}")
            
            self.publish_bbox(x, y, w, h)
            
            # depth roi가 true 일 때
            if self.depth_roi and face_location is not None:
                avg_depth, point_cloud = self.process_face_depth(face_location)
                
                if avg_depth is not None:
                    depth_msg = Float32()
                    depth_msg.data = avg_depth
            
        # else:
        #     rospy.loginfo("Target person not found in the image")

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
            # rospy.loginfo(f"Current FPS: {fps:.2f}")

        processing_time = end_time - start_time
        # rospy.loginfo(f"Frame processing time: {processing_time:.4f} seconds")

    def run(self):
        try:
            rospy.spin()
        finally:
            if self.is_resource_monitored:
                self.resource_monitor.stop()
                self.resource_monitor.join()
            
if __name__ == '__main__':
    try:
        face_recognition_module = FaceRecognition(model_name='dlib') 
        face_recognition_module.run()
    except rospy.ROSInterruptException:
        pass