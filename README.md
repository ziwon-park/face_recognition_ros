## Object Following Robot : Perception Package

### 실행 방법

```
$ roslaunch gazebo_recognition gazebo_recognition_node.launch
```

- 카메라 변경할 경우

```/launch/gazebo_recognition_node.launch```에서 
```
<rosparam command="load" file="$(find gazebo_recognition)/config/gazebo.yaml" />
```

여기 yaml 파일을 바꿔서 카메라 파라미터와 tf 정보를 수정할 수 있다. (하단 '카메라가 변경될 경우' 참조)

### Details

- transformation 수행하는 부분
    - ```src/gazebo_recognition_node.py```의 ```def setup_transforms```
- 카메라 파라미터, tf 정보
    - 카메라가 변경될 경우 해당 정보를 yaml에서 관리하도록 되어 있음. 
    - 현재 : ```/config/gazebo.yaml``` 


### 발행되고 있는 토픽

- ```/detected_object```
    - rviz 시각화용 마커
    - 유형 : visualization_msgs/Marker
- ```/detected_object_bbox```
    - bounding box의 위치 (x,y,w,h)
    - 유형 : std_msgs/Float32MultiArray


```
ziwon@ziwon-MacBookPro:~$ rostopic echo /detected_object_bbox
layout:
  dim:
    -
      label: "bbox"
      size: 4
      stride: 4
  data_offset: 0
data: [446.0, 381.0, 29.0, 29.0]
---
```

- ```/detected_object_pose```
    - 마커가 가리키는 방향 (roll, pitch, yaw) 값
    - 유형 : geometry_msgs/PoseStamped

```
ziwon@ziwon-MacBookPro:~$ rostopic echo /detected_object_pose
header:
  seq: 1
  stamp:
    secs: 110
    nsecs: 188600000
  frame_id: "trunk"
pose:
  position:
    x: -0.02555907986146712
    y: -0.7215
    z: -0.03349024553617212
  orientation:
    x: 0.00041030251555473527
    y: -0.02317189134679053
    z: 0.017699376251172343
    w: 0.9995747231615942
---
```

### 카메라가 변경될 경우

현재 gazebo 카메라가 상하좌우 반전되어 들어오고 있어서, 다음과 같은 부분들이 그에 맞게 수정되어 있다. 

```
# 현재 코드 (반전된 이미지용)

cam_x = -((center_x - self.image_width/2) / self.fx)   
cam_y = -((center_y - self.image_height/2) / self.fy) 

point_camera = [
    depth,
    -cam_x * depth,  
    -cam_y * depth  
]

yaw = -np.arctan2(dy, dx)
```

따라서 정상적인 카메라를 사용할 경우 cv2.flip 부분을 제거하고 마이너스 부호를 없애 줘야 한다. 

### 아직 미완인 부분

- gazebo 상 이미지가 아니라 realsense로 얼굴 인식한 결과 받아서 pitch, yaw 생성하는 부분
