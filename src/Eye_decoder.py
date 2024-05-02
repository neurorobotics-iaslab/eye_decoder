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
        
        self.FACE = [10, 152, 234, 454] # up, down, left, right
        
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
        while True:
            ret, frame = self.cap.read()
            frame = cv.flip(frame, 0)
            
            if rospy.is_shutdown():
                self.cap.release()
                cv.destroyAllWindows()
                rospy.signal_shutdown("Shutting down ROS node")
                break

            if not ret:
                rospy.WARN('[ERROR] camera not open correctly')
                break

            self.detection(frame)
            
            self.rate.sleep()

        
        
    def publish(self, frame, center_eyes, l_radius, r_radius, distances2nose, blinking, nose_points):
        msg = Eye()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'eye_decoder'
        msg.header.seq = self.seq
        self.seq += 1
        
        msg.left_pupil.x = center_eyes[0][0]  
        msg.left_pupil.y = center_eyes[0][1]
        
        msg.right_pupil.x = center_eyes[1][0]
        msg.right_pupil.y = center_eyes[1][1]
        
        msg.right_radius = r_radius
        msg.left_radius = l_radius
        
        msg.face_image = CvBridge().cv2_to_imgmsg(frame, encoding='bgr8')
        
        msg.distance_left = distances2nose[0]
        msg.distance_right = distances2nose[1]
        
        msg.count_frame_blinking = self.count_frame_blinking
        msg.blink = blinking
        
        for i in range(0, len(nose_points)):
            msg.nose_points.append(Point(x=nose_points[i][0], y=nose_points[i][1], z=0))
        
        self.pub.publish(msg)
        

    def detection(self, frame):
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
            tmp_c_l = np.array([l_cx, l_cy], dtype=np.int32)
            tmp_c_r = np.array([r_cx, r_cy], dtype=np.int32)
            
            face_region = np.array(mesh_points[self.FACE], np.int32)
            face_frame = frame[np.min(face_region[:, 1]):np.max(face_region[:, 1]), np.min(face_region[:, 0]):np.max(face_region[:, 0])].copy()
            
            center_left = np.array([tmp_c_l[0] - np.min(face_region[:, 0]), tmp_c_l[1] - np.min(face_region[:, 1])], dtype=np.int32)
            center_right = np.array([tmp_c_r[0] - np.min(face_region[:, 0]), tmp_c_r[1] - np.min(face_region[:, 1])], dtype=np.int32)
                
            distances2nose = distance_nose([center_left, center_right], mesh_points[self.NOSE])
            
            n_p = np.array(mesh_points[self.NOSE], np.int32)
            nose_points = np.array([[n_p[i][0] - np.min(face_region[:, 0]), n_p[i][1] - np.min(face_region[:, 1])] for i in range(0, len(n_p))], dtype=np.int32)
            
            self.publish(frame, [center_left, center_right], l_radius, r_radius, distances2nose, blinking, nose_points)

            if self.show_frame:
                f = face_frame.copy()
                #cv.circle(f, tuple(center_left), int(l_radius), (0, 255, 0), 2)
                #cv.circle(f, tuple(center_right), int(r_radius), (0, 255, 0), 2)
                #print(f'disance left: {distances2nose[0]}, distance right: {distances2nose[1]}')
                #print(f'blinking: {blinking}, count_frame_blinking: {self.count_frame_blinking}')
                #print(f'left eye: {center_left}, right eye: {center_right}')
                cv.imshow("Eye Detector", f)
                cv.waitKey(1)