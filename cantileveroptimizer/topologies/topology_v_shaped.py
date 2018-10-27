import numpy as np
from .topology_interface import Topology


class RegularVShaped(Topology):
    
    def __init__(self, params):
        """
        params['nelx']: The number of elements in the x-direction for one-half
                        of the cantilever. Symmetry is applied which doubles
                        the number of elements in the x-direction.
        params['nely']: The number of elements from the base to the tip of the
                        cantilever.
        """
        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']
        self._init_grid()
        
        # Public attributes.
        self.a = self._a0
        self.b = self._b0
        self.ind_size = 4
        self.topology = None
        self.connectivity_penalty = 0.0
        self.is_connected = True
        
        
    def update_topology(self, xs):
        
        # Range of xs is [-1,1], make range [0,1].
        xss = [0.5*x + 0.5 for x in xs]
        
        scale = xss[-1] if xss[-1] > 0.05 else 0.05
        self.a = self._a0 * scale
        self.b = self._b0 * scale
        
        nelx, nely = self._dim_elems
        p1 = (nelx - 2)*xss[0]
        p2 = p1 + 2 + (nelx - p1 - 2)*xss[1]
        p3 = (nely - 2)*xss[2]
        
        def line(y2, y1, x2, x1):
            m = (y2 - y1)/(x2 - x1)
            b = -m*x1 + y1
            return m, b
        
        m1, b1 = line(nely - 1, 0, nelx - 1, p1)
        m2, b2 = line(p3, 0, nelx, p2)
        in1 = self._y <= m1*self._x + b1
        in2 = self._y >= m2*self._x + b2
        topology = (in1 & in2)
        topology[nelx - 1, nely - 1] = 1  # Tip element.

        self.topology = np.vstack((topology, np.flipud(topology)))
        
        
    def _init_grid(self):
        
        # The topology is formulated on a normalised mesh where elements are
        # 1x1 units. The aspect ratio of the elements is defined by (a,b) and
        # is only taken into account in the finite element model.
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
