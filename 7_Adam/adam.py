import numpy as np

# define adam optimizer

class Adam(object):
    def __init__(self, step_size = 0.001, pho1 = 0.9, pho2 = 0.999, eps = 1e-8):
        '''
        step_size -  update parameter
        pho1 - exponential decay rate for first moment
        pho2 - exponential decay rate for second moment
        eps - small constant for numerical stablilization
        '''
        self.eps = eps
        self.step_size = step_size
        self.pho1 = pho1
        self.pho2 = pho2
        self.time_step = 0  # time step
        self.initated = False

    def __call__(self, g):
        self.time_step += 1
        if not self.initated:
            self.s = np.zeros_like(g) # first moment
            self.r = np.zeros_like(g)  # second moment
        # update moment
        self.s = self.pho1 * self.s + (1-self.pho1) * g
        self.r = self.pho2 * self.r + (1-self.pho2) * np.multiply(g, g)
        # correct bias
        self.s  = self.s / (1 - self.pho1 ** self.time_step)
        self.r = self.r / (1- self.pho2 ** self.time_step)
        dw = -self.step_size * self.s / (np.sqrt(self.r) + self.eps)
        return dw

class AdamInitializer(object):
    def __init__(self, step_size = 0.001, pho1 = 0.9, pho2 = 0.999, eps = 1e-8):
        '''
        step_size - update parameter
        pho1 - exponential decay rate for first moment
        pho2 - exponential decay rate for second moment
        eps - small constant for numerical stablilization
        '''
        self.eps = eps
        self.step_size = step_size
        self.pho1 = pho1
        self.pho2 = pho2

    def __call__(self):
        return Adam(self.step_size, self.pho1 , self.pho2 , self.eps)

