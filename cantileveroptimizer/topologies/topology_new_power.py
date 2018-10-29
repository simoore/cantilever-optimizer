import numpy as np


class NewPowerTopology(object):
    
    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        
        # Initialize the grid.
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
        
        # Public attributes.
        self.a = params['a0']
        self.b = params['b0']
        self.ind_size = 3
        self.topology = None
        self.connectivity_penalty = 0.0
        self.is_connected = True
        self.xtip = nelx
        self.ytip = nely
        
    def get_params(self):
        return (self.topology, self.a, self.b, self.xtip, self.ytip)
        
    def update_topology(self, xs):
        
        # Range of xs is [-1,1], make range [0,1].
        xss = [0.5*x + 0.5 for x in xs]
        
        nelx, nely = self._dim_elems
        p = 10 ** xs[0]
        t = 2 + (nelx - 2) * xss[2]
        tip = 2 + (nely - 2)  * xss[1]
        y1 = tip / nelx**p * self._x ** p
        xx = np.where(self._x - t < 0, np.full(self._x.shape, 0), self._x - t)
        y2 = tip / nelx**p * (xx) ** p - t
        
        in1 = self._y <= y1
        in2 = self._y >= y2
        topology = (in1 & in2)
        self.topology = np.vstack((topology, np.flipud(topology)))
        
        # Set tip locations.
        ytip_norm = np.amax(np.argwhere(topology[-1,:])) + 1
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (ytip_norm - 0.05)
