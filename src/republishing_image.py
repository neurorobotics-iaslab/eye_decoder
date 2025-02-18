#!/usr/bin/env python
import rospy
from eye_decoder.msg import Eye
from sensor_msgs.msg import Image

def callback(msg):
    pub.publish(msg.face_image)  # Extract and republish the image

rospy.init_node("face_image_republisher")
pub = rospy.Publisher("/face_image", Image, queue_size=10)  # Create a publisher for the image
rospy.Subscriber("/cvsa/eye", Eye, callback)  # Subscribe to your custom message
rospy.spin()