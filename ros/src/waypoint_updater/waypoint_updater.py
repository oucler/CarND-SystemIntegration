#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight, Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number

# states
INIT = 0
GO = 1
MPS = 0.44704


class WaypointUpdater(object):
    def __init__(self):
        # Declaration of Subscriber and Publisher

        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        self.sub_waypoints = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        # NOTE: comment out until we get traffic lights working...
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Waypoint, self.obstacle_cb)

        # TODO: For testing - comment out when we have /traffic_waypoint working...
        # rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_light_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # WaypointerUpdater class variable declaration
        self.pose = None
        self.position = None
        self.orientation = None
        self.theta = None
        self.waypoints = None
        self.updateRate = 2  # Update rate in second
        self.cwp = None
        self.final_waypoints = []
        self.cruz_control = None
        self.ideal_velocity = 4.*0.44704
        self.current_linear_velocity = 0.
        self.current_angular_velocity = 0.
        # Start the loop
        self.loop()

    def loop(self):
        """
        Calculating nextWaypoints then publish the finalWaypoints.
        Ros publish rate/second the waypoints.
        :return: None
        """
        rate = rospy.Rate(self.updateRate)
        while not rospy.is_shutdown():
            if self.waypoints and self.theta:
                self.cwp = self.getNextWaypointIndex()
                self.getNextWaypoints(LOOKAHEAD_WPS)
                self.publish()
            rate.sleep()

    def pose_cb(self, msg):
        """
        :param msg: Reading current_pose message assigned to class variables.
                    pose message has position[x,y,z] and orientation[x,y,z,w].
                    For more information please type rosmsg info styx_msgs/Lane
        :return: None
        """
        # self.i += 1
        self.pose = msg.pose
        self.position = msg.pose.position
        self.orientation = msg.pose.orientation
        # Transforming [x,y,z,w] in cartesian to [s,d,theta] in frenet coordinates
        euler = tf.transformations.euler_from_quaternion([
            self.orientation.x,
            self.orientation.y,
            self.orientation.z,
            self.orientation.w])
        self.theta = euler[2]

    def velocity_cb(self, msg):
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z

    def getNextWaypointIndex(self,position=None):
        """
        Finding out which position is the next among the self.waypoints.
        :return: NextWaypoint (cwp)
        """
        # Assigning a value big enough for distance and looping until finding the smallest distance.
        # and storing the index of the smallest distance.
        if position is None:
            position = self.position
        dist = 100000.
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        cwp = 0
        for i in range(len(self.waypoints)):
            d1 = dl(position, self.waypoints[i].pose.pose.position)
            if dist > d1:
                cwp = i
                dist = d1
        return cwp


    # Transform from global Cartesian x,y to local car coordinates x,y
    # where lx is pointing to the positive x axis and ly is deviation from the car's path
    def getLocalXY(self, theta, x, y):
        # convert to local coordinates
        vx = x - self.position.x
        vy = y - self.position.y
        lx = vx * np.cos(self.theta) + vy * np.sin(self.theta)
        ly = -vx * np.sin(self.theta) + vy * np.cos(self.theta)
        return lx, ly

    # Transform from local Cartesian x,y to global car coordinates x,y
    # where lx is pointing to the positive x axis and ly is deviation from the car's path
    def getWorldXY(self, theta, lx, ly):
        # convert back to global coordinates
        x = lx * np.cos(theta) - ly * np.sin(theta) + self.position.x
        y = lx * np.sin(theta) + ly * np.cos(theta) + self.position.y
        return x, y

    def getNextWaypoints(self, number):
        """
        :param number: Number of waypoints, this populates final_waypoints
        :return: None
        """
        # Initializing variables
        self.final_waypoints = []
        vptsx = []
        vptsy = []
        vptsd = []
        wlen = len(self.waypoints)
        velpolypoint = None
        wpd = 0.
        p0 = self.waypoints[self.cwp].pose.pose.position

        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(number):
            x = self.waypoints[(self.cwp + i) % wlen].pose.pose.position.x
            y = self.waypoints[(self.cwp + i) % wlen].pose.pose.position.y
            lx, ly = self.getLocalXY(self.theta, x, y)
            vptsx.append(lx)
            vptsy.append(ly)
            p1 = self.waypoints[(self.cwp + i) % wlen].pose.pose.position
            ld = dl(p0, p1)
            vptsd.append(wpd + ld)
            p0 = p1
            wpd += ld

        # Calculating cross track error (cte) for steering
        steerpoly = np.polyfit(np.array(vptsx), np.array(vptsy), 3)
        polynomial = np.poly1d(steerpoly)

        # Calculating trajectory
        for i in range(len(vptsx)):
            scte = polynomial([vptsx[i]])[0]
            # we are off by more than 2 meters per second!
            if self.current_linear_velocity > (self.ideal_velocity + 2.):
                # make hard correction.
                self.ideal_velocity *= 0.25

            p = Waypoint()
            p.pose.pose.position.x = self.waypoints[(self.cwp + i) % wlen].pose.pose.position.x
            p.pose.pose.position.y = self.waypoints[(self.cwp + i) % wlen].pose.pose.position.y
            p.pose.pose.position.z = self.waypoints[(self.cwp + i) % wlen].pose.pose.position.z
            p.pose.pose.orientation.x = self.waypoints[(self.cwp + i) % wlen].pose.pose.orientation.x
            p.pose.pose.orientation.y = self.waypoints[(self.cwp + i) % wlen].pose.pose.orientation.y
            p.pose.pose.orientation.z = self.waypoints[(self.cwp + i) % wlen].pose.pose.orientation.z
            p.pose.pose.orientation.w = self.waypoints[(self.cwp + i) % wlen].pose.pose.orientation.w
            p.twist.twist.linear.x = self.ideal_velocity
            p.twist.twist.linear.y = 0.
            p.twist.twist.linear.z = 0.
            p.twist.twist.angular.x = 0.
            p.twist.twist.angular.y = 0.
            p.twist.twist.angular.z = scte
            self.final_waypoints.append(p)

    def waypoints_cb(self, msg):
        """
        :param msg: Reading message from /base_waypoints publisher.
                    Published base_waypoints are stored in class variable self.waypoints.
        :return: None
        """
        # Check to see if there are waypoints in the msg.
        if self.waypoints is None:
            # Ensure that list doesn't have anything assigned initially.
            self.waypoints = []
            # Loop the msg.waypoints
            for waypoint in msg.waypoints:
                # need to create a new waypoint array so not to overwrite old one
                p = Waypoint()
                p.pose.pose.position.x = waypoint.pose.pose.position.x
                p.pose.pose.position.y = waypoint.pose.pose.position.y
                p.pose.pose.position.z = waypoint.pose.pose.position.z
                p.pose.pose.orientation.x = waypoint.pose.pose.orientation.x
                p.pose.pose.orientation.y = waypoint.pose.pose.orientation.y
                p.pose.pose.orientation.z = waypoint.pose.pose.orientation.z
                p.pose.pose.orientation.w = waypoint.pose.pose.orientation.w
                p.twist.twist.linear.x = waypoint.twist.twist.linear.x
                p.twist.twist.linear.y = waypoint.twist.twist.linear.y
                p.twist.twist.linear.z = waypoint.twist.twist.linear.z
                p.twist.twist.angular.x = waypoint.twist.twist.angular.x
                p.twist.twist.angular.y = waypoint.twist.twist.angular.y
                p.twist.twist.angular.z = waypoint.twist.twist.angular.z
                self.waypoints.append(p)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message.
        pass


    def traffic_light_cb(self, msg):
        # TODO: Callback for /vehicle/traffic_lights message
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        wlen = len(waypoints)
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1 % wlen].pose.pose.position, waypoints[i % wlen].pose.pose.position)
            wp1 = i
        return dist

    def publish(self):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(lane)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
