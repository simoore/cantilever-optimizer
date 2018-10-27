import numpy as np
from .topology_interface import Topology


class NewSplitTopology(Topology):
    
    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        self.a = params['a0']
        self.b = params['b0']
        self.ind_size = 8
        self.is_connected = True
        self.connectivity_penalty = 0
        self.topology = None
        
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
        
    
    def update_topology(self, xs):
        
        # Range of xs is [-1,1], make range [0,1].
        xss = [0.5*x + 0.5 for x in xs]
        
        nelx, nely = self._dim_elems
        p1 = 1 + (nely - 1)*xss[0]
        p2 = (nelx - 1) - (nelx - 1)*xss[1]
        p3 = (p1 - 1) - (p1 - 1)*xss[2]
        p4 = p2 + (nelx - 1 - p2)*xss[3]
        p5 = p4 + 1 + (nelx - (p4 + 1))*xss[4]
        p6 = p5 + (nelx - p5)*xss[5]
        p7 = p3*xss[6]
        p8 = p4*xss[7]
        
        in1x = (p2 <= self._x) & (self._x <= nelx)
        in1y = (p3 <= self._y) & (self._y <= p1)
        in2x = (p4 <= self._x) & (self._x <= p5)
        in2y = (p7 <= self._y) & (self._y <= p3)
        in3x = (p8 <= self._x) & (self._x <= p6)
        in3y = (0 <= self._y) & (self._y <= p7)
        topology = (in1x & in1y) | (in2x & in2y) | (in3x & in3y)

        self.topology = np.vstack((topology, np.flipud(topology)))
        
        
        ytip_norm = round(p1)
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (ytip_norm - 0.05)
        #self.xtip = 1e6 * (2 * self.a * nelx)
        #self.ytip = 1e6 * (2 * self.b * (nely - 0.05))
        
        
        