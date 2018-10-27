import numpy as np
from .topology_interface import Topology


class RectangleTopology(Topology):
    
    
    def __init__(self, params):
        """
        This topology requires two parameters to scale the width and length of 
        the rectangle. These parameters are passed directly to the finite 
        element model.
        """
        self.topology = np.ones((params['nelx'], params['nely']))
        self.ind_size = 2
        self.is_connected = True
        self.connectivity_penalty = 0
        self.a = params['a0']
        self.b = params['b0']
        self._a0 = params['a0']
        self._b0 = params['b0']


    def update_topology(self, xs):

        scale_a = xs[0] if xs[0] > 0.05 else 0.05
        scale_b = xs[1] if xs[1] > 0.05 else 0.05
        self.a = self._a0 * scale_a
        self.b = self._b0 * scale_b
    