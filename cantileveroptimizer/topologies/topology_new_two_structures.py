import numpy as np


class NewTwoStructuresTopology(object):

    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']
        self.ind_size = 8
        
        # Initialize the grid.
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
        
        # Public attributes.
        self.a = params['a0']
        self.b = params['b0']
        self.ind_size = 12
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
        
        p1 = (nelx - 3) * xss[0]
        p2 = p1 + 1 + (nelx - 2 - p1 - 1) * xss[1]
        p3 = 0
        p4 = 1 + (nely - 2 - 1) * xss[2]
        p5 = (p2 - 1) * xss[3]
        p6 = p2 + (nelx - 2 - p2) * xss[4]
        p7 = p4
        p8 = p4 + 1 + (nely - 1 - p4 - 1) * xss[5]
        p9 = (p6 - 1) * xss[6]
        p10 = nelx
        p11 = p8
        p12 = p8 + 1 + (nely - p8 - 1) * xss[7] 
        p13 = p6 + 1 + (nelx - p6 - 1) * xss[8]
        p14 = nelx
        p15 = p8
        p16 = 1 + (p4 - 1 - 1) * xss[9] 
        p17 = p2 + 1 + (nelx - p2 - 1) * xss[10]
        p18 = nelx
        p19 = p16
        p20 = 1 + (p16 - 1) * xss[11]
        
        in1 = (self._x > p1) & (self._x < p2) & (self._y > p3) & (self._y < p4)
        in2 = (self._x > p5) & (self._x < p6) & (self._y > p7) & (self._y < p8)
        in3 = (self._x > p9) & (self._x < p10) & (self._y > p11) & (self._y < p12)
        in4 = (self._x > p13) & (self._x < p14) & (self._y > p16) & (self._y < p15)
        in5 = (self._x > p17) & (self._x < p18) & (self._y > p20) & (self._y < p19)
        
        topology = in1 | in2 | in3 | in4
        if p14 - p13 > 0.5:
            topology = topology | in5
        self.topology = np.vstack((topology, np.flipud(topology)))
        
        # Set tip locations.
        ytip_norm = np.amax(np.argwhere(topology[-1,:])) + 1
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (ytip_norm - 0.05)
        