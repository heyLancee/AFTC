import numpy as np

class PController:
    def __init__(self, kp=1, dt=0.1):
        self.kp = kp
        self.dt = dt

        self.kps = []

    def compute(self, p_error):
        output = self.kp * p_error
        return output

    def update(self, kp):
        self.kp = kp
        self.kps.append(kp)

    def reset(self):
        self.kps.clear()


class PDController(PController):
    def __init__(self, kp=1, kd=0.1, dt=0.1):
        super().__init__(kp, dt)
        self.kd = kd

        self.kds = []

    def compute(self, p_error, d_error):
        output = self.kp * p_error + self.kd * d_error
        return output

    def update(self, kp, kd):
        kp = -np.abs(kp)
        kd = -np.abs(kd) * 2
        self.kp = kp
        self.kd = kd

        self.kps.append(kp)
        self.kds.append(kd)

    def reset(self):
        self.kps.clear()
        self.kds.clear()


class PIController(PController):
    def __init__(self, kp=1, ki=0.1, dt=0.1):
        super().__init__(kp, dt)
        self.ki = ki
        self.integral = 0.0

        self.kis = []

    def compute(self, p_error, i_error):
        self.integral += i_error * self.dt
        output = self.kp * p_error + self.ki * self.integral
        return output

    def update(self, kp, ki):
        self.kp = kp
        self.ki = ki

        self.kps.append(kp)
        self.kis.append(ki)

    def reset(self):
        self.integral = 0.0

        self.kps.clear()
        self.kis.clear()


class PIDController(PDController, PIController):
    def __init__(self, kp=1, ki=0.1, kd=0.1, dt=0.1):
        PDController.__init__(self, kp, kd, dt)
        PIController.__init__(self, kp, ki, dt)

    def compute(self, p_error, i_error, d_error):
        self.integral += i_error * self.dt
        output = self.kp * p_error + self.ki * self.integral + self.kd * d_error
        return output
    
    def update(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.kps.append(kp)
        self.kis.append(ki)
        self.kds.append(kd)
    
    def reset(self):
        self.integral = 0.0

        self.kps.clear()
        self.kis.clear()
        self.kds.clear()
