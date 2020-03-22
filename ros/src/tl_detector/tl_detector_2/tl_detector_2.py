#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml

from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector_2')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.camera_image = None
        self.waypoint_tree = None
        self.lights = []
        self.state = None

        self.debug_publisher = rospy.Publisher('/detected_light_images', Image, queue_size=10)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        self.bridge = CvBridge()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

 
    def image_cb(self, msg):
        self.has_image = True
        self.camera_image = msg
        
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, 'bgr8')
        cv_image = cv2.rectangle(cv_image, (100, 100), (200, 200), (0, 255, 0), thickness=3)
        #cv_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        self.debug_publisher.publish(cv_msg)

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
