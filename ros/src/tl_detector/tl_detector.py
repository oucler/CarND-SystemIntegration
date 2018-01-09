#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import math
import numpy as np

STATE_COUNT_THRESHOLD = 3
tlStates = ['RED','YELLOW','GREEN','UNKNOWN']
LOOKAHEAD_WPS = 200

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        # Subscribes to:
        #               /current_pose : determining the current location of the vehicle
        #               /base_waypoints : entire waypoints
        #               /vehicle_traffic : images to determine the traffic light state
        self.sub_current_pose = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.sub_base_waypoints = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        self.sub_traffic_lights = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        self.sub_image_color = rospy.Subscriber('/image_color', Image, self.image_cb)
        # Publish traffic_waypoints
        self.tl_waypoint_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        # Retrieve traffic_light_config
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        # Variable definitions
        self.pose = None
        self.position = None
        self.orientation = None
        self.theta = None
        self.waypoints = None
        self.wlen = 0
        self.lights = []
        self.row = None
        self.col = None
        self.camera_image = None
        self.updateRate = 10
        self.init = True
        self.nwp = None
        self.tlWaypoints = []
        self.ntlwp = None
        self.last_wp = -1
        self.bridge = CvBridge()
        self.light_classifier = None
        self.listener = tf.TransformListener()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.state_count = 0
        self.attribute = "NONE"
        self.has_image = False
        self.path = rospy.get_param('~model_path')
        self.camera_topic = rospy.get_param('~camera_topic')
        if self.path == "NONE":
            self.light_classifier = None
        elif self.path.find("GAN") > 0:
            self.row = 600
            self.col = 800
            self.light_classifier = TLClassifier(self.path)
        self.loop()

    def loop(self):
        rate = rospy.Rate(self.updateRate)
        while not rospy.is_shutdown():
            if not self.init:
                self.nwp = self.getNextWaypoint(self.pose)
                self.ntlwp = self.getNextTLWaypoint(LOOKAHEAD_WPS)
                if self.ntlwp is not None and self.sub_image_color is None:
                    self.sub_image_color = rospy.Subscriber(self.camera_topic, Image, self.image_cb)
                elif self.ntlwp is None and self.sub_image_color is not None:
                    self.sub_image_color.unregister()
                    self.sub_image_color = None
                    self.last_wp = -1
                    self.tl_waypoint_pub.publish(Int32(self.last_wp))
            elif self.light_classifier is None:
                self.tl_waypoint_pub.publish(Int32(-1))
            rate.sleep()

    def getNextWaypoint(self,pose):
        """
        This function calculate number of waypoints
        :param: pose --> position of the car
        :return: nwp --> Number of waypoints
        """
        # big enough from the waypoints
        dist = 10000
        nwp = 0
        location = pose.position
        # Labda used to calculate the distance
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)

        #Calculating closest waypoints
        for i in range(len(self.waypoints)):
            d1 = dl(location, self.waypoints[i].pose.pose.position)
            if dist > d1:
                nwp = i
                dist = d1
        #
        x = self.waypoints[nwp].pose.pose.position.x
        y = self.waypoints[nwp].pose.pose.position.y
        heading = np.arctan2((y-location.y), (x-location.x))
        angle = np.abs(self.theta-heading)
        if angle > np.pi/4.:
            nwp += 1
            if nwp >= len(self.waypoints):
                nwp = 0
        return nwp

    def getNextTLWaypoint(self,num):
        """
        Find the closest waypoints to the traffic lights
        :param num: Number of ahead waypoints
        :return:
        """
        tlwp = None
        light = None
        for i in range(len(self.tlWaypoints)):
            # Ensuring Traffic Light waypoints ahead
            if self.nwp < self.tlWaypoints[i]:
                tlwp = self.tlWaypoints[i] - self.nwp
                if tlwp < num - 2:
                    # Checking a condition if the traffic lights within 100 meters
                    if self.distance(self.waypoints, self.nwp, (self.nwp + tlwp) % self.wlen) < 100:
                        light = tlwp
        return light

    def distance(self, waypoints, wp1, wp2):
        """
        Calculating distance between wp1 and wp2
        :param waypoints:
        :param wp1: First waypoint
        :param wp2: Second waypoint
        :return: distantace
        """
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def pose_cb(self, msg):
        """
        Retrieving position, orientation, and calculating theta
        :param msg: has pos message which has position and orientation information
        :return: None
        """
        self.pose = msg.pose
        self.position = msg.pose.position
        self.orientation = msg.pose.orientation
        euler = tf.transformations.euler_from_quaternion([
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w])
        self.theta = euler[2]
        if self.light_classifier is not None:
            if self.light_classifier.predict is None:
                print ("Info: Initializing TRAFFIC LIGHT DETECTOR")
        else:
            print ("Warning: NO TRAFFIC LIGHT DETECTOR !!!")

    def waypoints_cb(self, msg):
        """
        Assign waypoints to local class variable self.waypoints
        for later calculations
        :param waypoints:
        :return: None
        """

        if self.waypoints is None:
            self.waypoints = []
            for waypoint in msg.waypoints:
                self.waypoints.append(waypoint)
        self.wlen = len(self.waypoints)

        # Traffic Lights waypoint population
        self.initTLWaypoints()

    def initTLWaypoints(self):
        """
        Finding out the closest waypoints to the traffic lights
        :return: None
        """
        # Labda function for distance
        dl = lambda a, b: math.sqrt((a.x - b[0]) ** 2 + (a.y - b[1]) ** 2)
        for i in range(len(self.config['stop_line_positions'])):
            dist = 100000
            tlwp = 0
            for j in range(len(self.waypoints)):
                d1 = dl(self.waypoints[j].pose.pose.position, self.config['stop_line_positions'][i])
                if dist > d1:
                    tlwp = j
                    dist = d1
            self.tlWaypoints.append(tlwp)

    def traffic_cb(self, msg):
        """
        Directly assigning msg to self.lights list
        :param msg:
        :return: None
        """
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.row = int(msg.height)
        self.col = int(msg.width)
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if state == TrafficLight.GREEN and light_wp is not None:
                light_wp = -light_wp
            elif state == TrafficLight.UNKNOWN:
                light_wp = -1
            #light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.tl_waypoint_pub.publish(Int32(light_wp))
        else:
            self.tl_waypoint_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        self.init = False

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if (not self.has_image):
            self.prev_light_loc = None
            return TrafficLight.RED

            # fixing convoluted camera encoding...
        if hasattr(self.camera_image, 'encoding'):
            self.attribute = self.camera_image.encoding
            if self.camera_image.encoding == '8UC3':
                self.camera_image.encoding = "rgb8"
        else:
            self.camera_image.encoding = 'rgb8'
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Traffic Light Classification
        if self.light_classifier is not None:
            classification = self.light_classifier.get_classification(cv_image)
        else:
            classification = TrafficLight.UNKNOWN
        print ("Traffic Light State: ", tlStates[classification])
        return classification

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if self.init:
            state = self.get_light_state(0)
            return -1, TrafficLight.UNKNOWN
        elif self.ntlwp:
            state = self.get_light_state(self.ntlwp)
            # state = TrafficLight.RED
            return self.ntlwp, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
