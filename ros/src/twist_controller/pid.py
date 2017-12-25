
MIN_NUM = float('-inf')  # Need to change these values
MAX_NUM = float('inf')   # Need to change these values


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        self.int_val = self.last_int_val = self.last_error = 0.

    def reset(self):
        """
        Resets the integral error
        :return:
        """
        self.int_val = 0.0
        self.last_int_val = 0.0

    def step(self, error, sample_time):
        """

        :param error: The error after that particular actuation step
        :param sample_time: time interval
        :return: value (actuation) after applying PID control to it
        """
        self.last_int_val = self.int_val

        integral = self.int_val + error * sample_time;
        derivative = (error - self.last_error) / sample_time;

        y = self.kp * error + self.ki * self.int_val + self.kd * derivative;
        val = max(self.min, min(y, self.max))

        if val > self.max:
            val = self.max
        elif val < self.min:
            val = self.min
        else:
            self.int_val = integral
        self.last_error = error

        return val

    def twiddle(self):
        """Implement an algorithm to optimize PID control parameters """
        pass