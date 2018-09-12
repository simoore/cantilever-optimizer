import numpy as np


class RegularSplitTopology(object):
    
    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']
        self.ind_size = 8
        
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
        
    
    
    def update_topology(self, xs):
        
        # Range of xs is [-1,1], make range [0,1].
        xss = [0.5*x + 0.5 for x in xs]
        
        scale = xss[7] if xss[7] > 0.05 else 0.05
        self.a = self._a0 * scale
        self.b = self._b0 * scale
        
        nelx, nely = self._dim_elems
        p1 = (nelx - 1) - (nelx - 1)*xss[0]
        p2 = (nely - 1) - (nely - 1)*xss[1]
        p3 = p1 + (nelx - 1 - p1)*xss[2]
        p4 = p3 + 1 + (nelx - (p3 + 1))*xss[3]
        p5 = p4 + (nelx - p4)*xss[4]
        p6 = p2*xss[5]
        p7 = p3*xss[6]
        
        in1x = (p1 <= self._x) & (self._x <= nelx)
        in1y = (p2 <= self._y) & (self._y <= nely)
        in2x = (p3 <= self._x) & (self._x <= p4)
        in2y = (p6 <= self._y) & (self._y <= p2)
        in3x = (p7 <= self._x) & (self._x <= p5)
        in3y = (0 <= self._y) & (self._y <= p6)
        topology = (in1x & in1y) | (in2x & in2y) | (in3x & in3y)

        self.topology = np.vstack((topology, np.flipud(topology)))
        self.is_connected = True
        self.connectivity_penalty = 0
        #cantilever = microfem.Cantilever(topology, self.a, self.b)
        #microfem.plot_topology(cantilever)
        
        self.xtip = 1e6 * (2 * self.a * nelx)
        self.ytip = 1e6 * (2 * self.b * (nely - 0.05))
        
        
        