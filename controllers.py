import numpy as np


class Controller(object):
    def __init__(self):
        super().__init__()

    def select_action(self, *args, **kwargs):
        raise NotImplementedError
    
    def set_params(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_params(self):
        raise NotImplementedError


class SunPointController(Controller):
    def __init__(self, kp, kd):
        super().__init__()

        self.Kp = kp
        self.Kd = kd

    def set_params(self, *args, **kwargs):
        kp, kd = args

        self.Kp = kp
        self.Kd = kd

    def get_params(self):
        return self.Kp, self.Kd

    def select_action(self, *args, **kwargs):
        s, sd, omega = args
        # u = -Kp*(s - sd)xs - Kd*omega
        u = - self.Kp * np.cross((s - sd).flatten(), s.flatten()) - self.Kd * omega.flatten()
        return u
