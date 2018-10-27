import numpy as np


class NewVShaped(object):
    
    def __init__(self, params):
        """
        params['nelx']: The number of elements in the x-direction for one-half
                        of the cantilever. Symmetry is applied which doubles
                        the number of elements in the x-direction.
        params['nely']: The number of elements from the base to the tip of the
                        cantilever.
        """
        self._dim_elems = (params['nelx'], params['nely'])
        self._init_grid()
        
        # Public attributes.
        self.a = params['a0']
        self.b = params['b0']
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
        
        offset = 2
        nelx, nely = self._dim_elems
        p1 = (nelx - offset)*xss[0]
        p2 = offset + (nely - offset)*xss[1]
        p3 = p1 + offset + (nelx - p1 - offset)*xss[2]
        p4 = (p2 - offset)*xss[3]
        
        def line(y2, y1, x2, x1):
            m = (y2 - y1)/(x2 - x1)
            b = -m*x1 + y1
            return m, b
        
        m1, b1 = line(p2, 0, nelx - 1, p1)
        m2, b2 = line(p4, 0, nelx, p3)
        in1 = self._y <= m1*self._x + b1
        in2 = self._y >= m2*self._x + b2
        topology = (in1 & in2)
        
        self.topology = np.vstack((topology, np.flipud(topology)))
        
        ytip_norm = np.amax(np.argwhere(topology[-1,:])) + 1
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (ytip_norm - 0.05)


    def _init_grid(self):
        
        # The topology is formulated on a normalised mesh where elements are
        # 1x1 units. The aspect ratio of the elements is defined by (a,b) and
        # is only taken into account in the finite element model.
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
