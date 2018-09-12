import numpy as np
from topology_interface import Topology


class NewRectangleTopology(Topology):
    
    
    def __init__(self, params):
        """
        This topology requires two parameters to scale the width and length of 
        the rectangle. These parameters are passed directly to the finite 
        element model.
        """
        nelx = params['nelx']
        nely = params['nely']
        self._dim_elems = (nelx, nely)
        
        # Initialize the grid. Topology is formed of a grid of normalized 1x1
        # mesh elements.
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        self._x = ii.T + 0.5
        self._y = jj.T + 0.5
        
        # Public attributes.
        self.topology = None
        self.is_connected = True
        self.connectivity_penalty = 0
        self.a = params['a0']
        self.b = params['b0']
        self.ind_size = 2
        self.xtip = 1e6 * (2 * self.a * nelx)
        self.ytip = 1e6 * (2 * self.b * (nely - 0.05))


    def update_topology(self, xs):

        # Input x in range [-1, 1], xss in range [0, 1].
        xss = [0.5*x + 0.5 for x in xs]
        
        # The rectangle is made of at least one element.
        nelx, nely = self._dim_elems
        p1 = (nelx - 1)*xss[0]
        p2 = 1 + (nely - 1)*xss[1]  
        
        in1x = (p1 <= self._x) & (self._x <= nelx)
        in1y = (p2 >= self._y)
        topology = (in1x & in1y)
        self.topology = np.vstack((topology, np.flipud(topology)))
        
        ytip_norm = round(p2)
        self._update_tip_location(ytip_norm)
        
        
    def _update_tip_location(self, ytip_normalized):
        """
        The tip location is along the center of the cantilever, near the top
        of the last element along the central axis.
        ytip_normalized: The position of the tip given in a normalized range
                         [0, nely].
        """
        nelx, nely = self._dim_elems
        self.xtip = 1e6 * (2 * self.a * nelx)
        self.ytip = 1e6 * (2 * self.b * (ytip_normalized - 0.05))
        
        