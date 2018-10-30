import numpy as np


class NewSteppedTopology(object):

    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']

        # The topology is formulated on a normalised mesh where elements are
        # 1x1 units. The aspect ratio of the elements is defined by (a,b) and
        # is only taken into account in the finite element model.
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5

        # Public attributes.
        self.a = self._a0
        self.b = self._b0
        self.ind_size = 4
        self.topology = None
        self.connectivity_penalty = 0.0
        self.is_connected = True
        self.xtip = 0
        self.ytip = 0

    def get_params(self):
        return (self.topology, self.a, self.b, self.xtip, self.ytip)

    def update_topology(self, xs):

        # Range of xs is [-1,1], make range [0,1].
        xss = [0.5*x + 0.5 for x in xs]

        nelx, nely = self._dim_elems
        p0 = 1 + (nely - 1)*xss[0]
        p1 = (nelx - 1)*xss[1]
        p2 = p1*xss[2]
        p3 = p1 + (nelx - 1 - p1)*xss[3]

        in1x = (p1 <= self._x) & (self._x <= nelx)
        in1y = (0 <= self._y) & (self._y <= p2)
        in2x = (p3 <= self._x) & (self._x <= nelx)
        in2y = (p2 <= self._y) & (self._y <= p0)
        topology = (in1x & in1y) | (in2x & in2y)

        self.topology = np.vstack((topology, np.flipud(topology)))

        #ytip_norm = round(p2)
        ytip_norm = np.amax(np.argwhere(topology[-1,:])) + 1
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (ytip_norm - 0.05)
