#!/home/urllaptop/anaconda3/envs/face-ros/bin/python3

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Float32, MultiArrayDimension

from cv_bridge import CvBridge
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler, rotation_matrix, concatenate_matrices, euler_from_matrix

class ObjectTracker:
    def __init__(self):
        rospy.init_node('tf_handler_node', anonymous=True)
        
        self.use_face_detection = rospy.get_param('~face_detection', False)
        self.bridge = CvBridge()
        
        try: 
            # Camera parameters
            self.fx = rospy.get_param('camera_face/fx')
            self.fy = rospy.get_param('camera_face/fy')
            self.cx = rospy.get_param('camera_face/cx')
            self.cy = rospy.get_param('camera_face/cy')
            self.image_width = rospy.get_param('camera_face/width')
            self.image_height = rospy.get_param('camera_face/height')    
                
            # Transform parameters
            self.trans_x = rospy.get_param('trans_x', 0.2785)  
            self.trans_y = rospy.get_param('trans_y', 0.0125)
            self.trans_z = rospy.get_param('trans_z', 0.0167)    
            
            # Rotation parameters            
            self.rot_params = {}
            for i in range(1, 4):  # r1, r2, r3
                param_name = f'rotation/r{i}'
                if rospy.has_param(param_name):
                    self.rot_params[f'r{i}'] = rospy.get_param(param_name)  

            self.got_camera_info = True
            
        except KeyError as e:
            rospy.logerr(f"Failed to get parameters: {e}")
            self.got_camera_info = False

        # self.camera_matrix = None
        # self.image_width = None
        # self.image_height = None
        # self.got_camera_info = False
        
        self.marker_pub = rospy.Publisher('/detected_object', Marker, queue_size=10)   
        self.pose_pub = rospy.Publisher('/detected_object_pose', PoseStamped, queue_size=10)
        self.bbox_pub = rospy.Publisher('/detected_object_bbox', Float32MultiArray, queue_size=10)
             
        # Subscribers based on detection mode
        if self.use_face_detection:
            rospy.loginfo("Using real camera")
            self.bbox_sub = rospy.Subscriber('/face_bbox', Float32MultiArray, self.bbox_callback)
            self.depth_sub = rospy.Subscriber('/face_depth', Float32, self.depth_data_callback)
        else:
            rospy.loginfo("Using gazebo camera")
            if self.got_camera_info:
                self.image_sub = rospy.Subscriber('/camera_face/color/image_raw', 
                                                Image, 
                                                self.image_callback)
        # if self.got_camera_info:
        #     self.image_sub = rospy.Subscriber('/camera_face/color/image_raw', 
        #                                     Image, 
        #                                     self.image_callback)
        # self.image_sub = None
        
        self.face_depth = 1.0
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        self.setup_transforms()
            
    def create_rotation_matrix(self, axis, angle):
        """
        주어진 축과 각도로 회전 행렬 생성
        angle: 라디안 단위의 회전 각도
        """
        if axis.lower() == 'x':
            return rotation_matrix(angle, [1, 0, 0])
        elif axis.lower() == 'y':
            return rotation_matrix(angle, [0, 1, 0])
        elif axis.lower() == 'z':
            return rotation_matrix(angle, [0, 0, 1])
        else:
            raise ValueError(f"Invalid rotation axis: {axis}")

    def setup_transforms(self):
        rot_mat = np.eye(4)
        if self.rot_params:
            for r_key in sorted(self.rot_params.keys()):  
                axis, angle = self.rot_params[r_key]
                r_mat = self.create_rotation_matrix(axis, angle)
                rot_mat = np.matmul(rot_mat, r_mat)
        else:
            rospy.logwarn("No rotation parameters found in YAML, using default rotation")
            align_rot_y = rotation_matrix(-np.pi/2, [0, 1, 0])
            rot_x = rotation_matrix(-np.pi/2, [1, 0, 0])
            rot_z = rotation_matrix(-np.pi/2, [0, 0, 1])
            rot_mat = np.matmul(np.matmul(align_rot_y, rot_z), rot_x)
        # rot_x_pi = rotation_matrix(np.pi, [1, 0, 0])
        
        # rot_x_minus_pi_2 = rotation_matrix(-np.pi/2, [1, 0, 0])
        # rot_z_minus_pi_2 = rotation_matrix(-np.pi/2, [0, 0, 1])
        
        # # 카메라의 Z축을 로봇의 X축과 정렬
        # align_rot_y = rotation_matrix(-np.pi/2, [0, 1, 0])
        # align_rot_z = rotation_matrix(np.pi, [0, 0, 1])
        # align_rot = np.matmul(align_rot_z, align_rot_y)
        
        # rot_mat = np.matmul(np.matmul(align_rot, rot_z_minus_pi_2), rot_x_minus_pi_2)
        
        # translation
        trans_mat = np.eye(4)
        trans_mat[0:3, 3] = [self.trans_x, self.trans_y, self.trans_z]
        
        self.transform_mat = np.matmul(trans_mat, rot_mat)
        self.inv_transform_mat = np.linalg.inv(self.transform_mat)

    def transform_point(self, point_camera):
        point_h = np.array([point_camera[0], point_camera[1], point_camera[2], 1.0])
        point_trunk = np.matmul(self.inv_transform_mat, point_h)
        
        return point_trunk[:3]

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            
            self.image_width = msg.width
            self.image_height = msg.height
            
            self.got_camera_info = True
            
            self.image_sub = rospy.Subscriber('/camera_face/color/image_raw', 
                                            Image, 
                                            self.image_callback)
            
            self.camera_info_sub.unregister()
            

    def depth_data_callback(self, msg):
        self.face_depth = msg.data
                
    def bbox_callback(self, msg):
        if not self.got_camera_info:
            return
            
        try:
            x, y, w, h = msg.data
            
            center_x = x + w/2
            center_y = y + h/2
            
            cam_x = (center_x - self.image_width/2) / self.fx
            cam_y = (center_y - self.image_height/2) / self.fy
            
            # 여기를 손봐야 함. 
            depth = self.face_depth
            print("depth is", depth)

            point_camera = [
                depth,    
                cam_x * depth, 
                cam_y * depth  
            ]
            
            point_trunk = self.transform_point(point_camera)
            
            dx = point_trunk[0]
            dy = point_trunk[1]
            dz = point_trunk[2]
            
            yaw = np.arctan2(dy, dx)  
            pitch = np.arctan2(dz, np.sqrt(dx*dx + dy*dy))
                            
            self.publish_pose(point_trunk, yaw, pitch)
            self.publish_marker(point_trunk)
            self.publish_bbox(x, y, w, h)
            
        except Exception as e:
            rospy.logerr(f"Error processing bbox: {str(e)}")

    def image_callback(self, msg):
        if not self.got_camera_info:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 이미지가 상하좌우 반전되어서 들어옴 
            cv_image = cv2.flip(cv_image, -1)  
            
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                self.publish_bbox(x, y, w, h)
                
                center_x = x + w/2
                center_y = y + h/2
                
                cam_x = ((center_x - self.image_width/2) / self.fx) 
                cam_y = ((center_y - self.image_height/2) / self.fy) 
                
                depth = 1.0

                point_camera = [
                    depth,    
                    cam_x * depth, 
                    cam_y * depth  
                ]
                
                point_trunk = self.transform_point(point_camera)
                
                dx = point_trunk[0]
                dy = point_trunk[1]
                dz = point_trunk[2]
                
                yaw = -np.arctan2(dy, dx)
                pitch = np.arctan2(dz, np.sqrt(dx*dx + dy*dy))
                                
                self.publish_pose(point_trunk, yaw, pitch)
                self.publish_marker(point_trunk)
                
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(cv_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

                cv2.imshow('Object Detection', cv_image)
                cv2.waitKey(1)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")

    def publish_marker(self, point_trunk):
        marker = Marker()
        marker.header.frame_id = "base"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0
        
        dx = point_trunk[0]
        dy = point_trunk[1]
        dz = point_trunk[2]
        
        yaw = np.arctan2(dy, dx)
        pitch = np.arctan2(dz, np.sqrt(dx*dx + dy*dy))
        q = quaternion_from_euler(0, pitch, yaw)
        
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]
        
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        marker.scale.x = dist 
        marker.scale.y = 0.05 
        marker.scale.z = 0.05
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)
        
    def publish_bbox(self, x, y, w, h):
        bbox_msg = Float32MultiArray()
        
        bbox_msg.layout.dim = [MultiArrayDimension()]
        bbox_msg.layout.dim[0].label = "bbox"
        bbox_msg.layout.dim[0].size = 4
        bbox_msg.layout.dim[0].stride = 4
        
        bbox_msg.data = [float(x), float(y), float(w), float(h)]
        
        self.bbox_pub.publish(bbox_msg)
        
    def publish_pose(self, point_trunk, yaw, pitch):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = "trunk"
        pose_msg.header.stamp = rospy.Time.now()
        
        pose_msg.pose.position.x = point_trunk[0]
        pose_msg.pose.position.y = point_trunk[1]
        pose_msg.pose.position.z = point_trunk[2]
        
        yaw = - yaw
        q = quaternion_from_euler(0, pitch, yaw)
        pose_msg.pose.orientation.x = q[0]
        pose_msg.pose.orientation.y = q[1]
        pose_msg.pose.orientation.z = q[2]
        pose_msg.pose.orientation.w = q[3]
        
        self.pose_pub.publish(pose_msg)


if __name__ == '__main__':
    try:
        tracker = ObjectTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass