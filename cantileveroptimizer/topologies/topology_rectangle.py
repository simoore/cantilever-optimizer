import numpy as np


class RectangleTopology(object):

    def __init__(self, params):
        """
        This topology requires two parameters to scale the width and length of
        the rectangle. These parameters are passed directly to the finite
        element model.
        """
        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']

        # Public Attributes.
        self.topology = np.ones(self._dim_elems)
        self.ind_size = 2
        self.is_connected = True
        self.connectivity_penalty = 0
        self.a = params['a0']
        self.b = params['b0']
        self.xtip = 0
        self.ytip = 0

    def update_topology(self, xs):

        scale_a = xs[0] if xs[0] > 0.05 else 0.05
        scale_b = xs[1] if xs[1] > 0.05 else 0.05
        self.a = self._a0 * scale_a
        self.b = self._b0 * scale_b
        nelx, nely = self._dim_elems
        self.xtip = self.a * nelx
        self.ytip = 2 * self.b * (nely - 0.05)

    def get_params(self):
        return (self.topology, self.a, self.b, self.xtip, self.ytip)
