from yaw_controller import YawController
from lowpass import LowPassFilter
import numpy as np
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args):
        """
        Controller Class to calculate throttle, brake and steer
        :param args:
        :param kwargs:
        """
        ## Get all the information that you can from the dbw_node - Clean this up
        if len(args) >= 1:
            self.vehicle_mass = args[0]
        else:
            self.vehicle_mass = 1736.35

        if len(args) >= 2:
            self.fuel_capacity = args[1]
        else:
            self.fuel_capacity = 13.5

        if len(args) >=3:
            self.brake_deadband = args[2]
        else:
            self.brake_deadband = 0.1

        if len(args) >= 4:
            self.decel_limit = args[3]
        else:
            self.decel_limit = -5

        if len(args) >= 5:
            self.accel_limit = args[4]
        else:
            self.accel_limit = 1

        if len(args) >= 6:
            self.wheel_radius = args[5]
        else:
            self.wheel_radius = 0.2413
        if len(args) >= 7:
            self.wheel_base = args[6]
        else:
            self.wheel_base = 2.8498

        if len(args) >= 8:
            self.steer_ratio = args[7]
        else:
            self.steer_ratio = 14.8

        if len(args) >= 9:
            self.max_lat_accel = args[8]
        else:
            self.max_lat_accel = 3.

        if len(args) >= 10:
            self.max_steer_angle = args[9]
        else:
            self.max_steer_angle = 8

        # Use PID control for throttle and brake - if throttle is negative then use brake, else brake remains 0
        Kp = 0.1
        Ki = 0.0001
        Kd = 0.05

        self.throttle_pid = PID(Kp, Ki, Kd, self.decel_limit, self.accel_limit)

    def control(self, linear_velocity, angular_velocity, current_velocity, deltaT, dbw_is_on):
        """

        :param args: At a minimum will need the linear_velocity, angular_velocity and current_velocity.
        Look at yaw_controller for more details about each param.
        :param kwargs:
        :return:
        """
        throttle = steer = brake = 0
        # TODO: Change the arg, kwarg list to suit your needs
        if dbw_is_on:

            brake = 0

            # Use YawController to calculate the steering angle
            yc = YawController(self.wheel_base, self.steer_ratio, 0, self.max_lat_accel, self.max_steer_angle)

            # Calculate the steering angle - Start off with just simple calculation
            steer = yc.get_steering(linear_velocity, angular_velocity, current_velocity)

            # Use the throttle PID to figure out the next
            velocity_error = linear_velocity - current_velocity
            throttle = self.throttle_pid.step(velocity_error, deltaT)

            # # Bound it based on ROSBAG output for actual car
            # if throttle > 0.025:
            #     throttle = 0.025
            #     brake = 0

            # Looser bound for running simulation
            if throttle > 0.2:
                throttle = 0.2
                brake = 0

            # If throttle is negative, then brake needs to be activated
            if throttle < 0:
                brake = -throttle
                throttle = 0
                if brake > 0.2:
                    brake = 0.2

        else:
            # Control is now with the driver, so reset the PID controller
            self.throttle_pid.reset()

        return throttle, brake, steer
