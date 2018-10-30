import numpy as np


class PowerTopology(object):

    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']
        self._t0 = 20
        self._init_grid()

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
        return (self.topology,
                self.a,
                self.b,
                self.xtip,
                self.ytip)


    def update_topology(self, xs):

        # Range of xs is [-1,1], make range [0,1].
        xss = [0.5*x + 0.5 for x in xs]

        scale = xss[2] if xss[2] > 0.05 else 0.05
        self.a = self._a0 * scale
        self.b = self._b0 * scale

        scale_t = xss[1] if xss[1] > 0.05 else 0.05
        t = self._t0 * scale_t

        p = 10 ** xs[0]
        nelx, nely = self._dim_elems
        y1 = nely / nelx**p * (self._x + t) ** p
        xx = np.where(self._x - t < 0, np.full(self._x.shape, 0), self._x - t)
        y2 = nely / nelx**p * (xx) ** p

        in1 = self._y <= y1 + t
        in2 = self._y >= y2 - t
        topology = (in1 & in2)
        topology[nelx - 1, nely - 1] = 1  # Tip element.

        self.topology = np.vstack((topology, np.flipud(topology)))
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (nely - 0.05)


    def _init_grid(self):

        # The topology is formulated on a normalised mesh where elements are
        # 1x1 units. The aspect ratio of the elements is defined by (a,b) and
        # is only taken into account in the finite element model.
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
