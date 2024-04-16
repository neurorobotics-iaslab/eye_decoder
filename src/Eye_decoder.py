import rospy
import cv2 as cv
import mediapipe as mp
import numpy as np
from eye_decoder.msg import Eye
from utils import blinking_ratio, distance_nose
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

class Eye_decoder:
    def __init__(self):
        # fixed variables
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                                                            min_detection_confidence=0.5, 
                                                            min_tracking_confidence=0.5)

        self.RIGHT_EYE_POINTS = [ 33, 160, 159, 158, 133, 153, 145, 144]
        self.LEFT_EYE_POINTS  = [362, 385, 386, 387, 263, 373, 374, 380]

        self.LEFT_IRIS =  [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        self.NOSE = [1, 168]
        
        self.seq = 0
        
    def configure(self):
        rospy.init_node('eye_decoder', anonymous=True)
        self.pub = rospy.Publisher('cvsa/eye', Eye, queue_size=1)

        self.cam_source = rospy.get_param('eye_decoder/cam_source', 0)
        self.cap = cv.VideoCapture(self.cam_source)

        r = rospy.get_param('eye_decoder/rate', 256)
        self.rate = rospy.Rate(r)

        self.blink_threshold = rospy.get_param('eye_decoder/blink_threshold', 0.52)
        self.count_frame_blinking = 0
        
        self.show_frame = rospy.get_param('eye_decoder/show_frame', True)


    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()

            if not ret:
                rospy.WARN('[ERROR] camera not open correctly')
                break

            self.detection(frame)
            
            self.rate.sleep()

        self.cap.release()
        cv.destroyAllWindows()
        

    def publish(self, frame_with_points, frame, center_eyes, distances2nose, blinking, nose_points):
        msg = Eye()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'eye_decoder'
        msg.header.seq = self.seq
        self.seq += 1
        
        msg.left_pupil.x = center_eyes[0][0]  
        msg.left_pupil.y = center_eyes[0][1]
        
        msg.right_pupil.x = center_eyes[1][0]
        msg.right_pupil.y = center_eyes[1][1]
        
        msg.image = CvBridge().cv2_to_imgmsg(frame, encoding='bgr8')
        msg.image_with_points = CvBridge().cv2_to_imgmsg(frame_with_points, encoding='bgr8')
        
        msg.distance_left = distances2nose[0]
        msg.distance_right = distances2nose[1]
        
        msg.count_frame_blinking = self.count_frame_blinking
        msg.blink = blinking
        
        for i in range(0, len(nose_points)):
            msg.nose_points.append(Point(x=nose_points[i][0], y=nose_points[i][1], z=0))
        
        self.pub.publish(msg)
        

    def detection(self, frame):
        frame_clone = frame.copy()
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = self.mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )

            eyes_aspect_ratio = blinking_ratio(mesh_points_3D, self.LEFT_EYE_POINTS, self.RIGHT_EYE_POINTS) 

            if eyes_aspect_ratio <= self.blink_threshold:
                self.count_frame_blinking += 1
                blinking = True
            else:
                blinking = False

            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[self.LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[self.RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)  # Left iris
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA) # Right iris
                
            distances2nose = distance_nose([center_left, center_right], mesh_points[self.NOSE])
            
            for point in mesh_points[self.NOSE]:
                cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            
            
            self.publish(frame, frame_clone, [center_left, center_right], distances2nose, blinking, mesh_points[self.NOSE])

            if self.show_frame:
                f = frame.copy()
                cv.putText(f, f"Blinks: {self.count_frame_blinking}", (30, 50), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.imshow("Eye Detector", f)
                cv.waitKey(1)