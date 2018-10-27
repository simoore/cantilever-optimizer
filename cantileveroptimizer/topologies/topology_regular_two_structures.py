import numpy as np
from .topology_interface import Topology


class RegularTwoStructureTopology(Topology):
    """
    Public Attributes
    -----------------
    self.ind_size
    self.a
    self.b
    self.topology
    self.is_connected
    self.connectivity_penalty
    """
    
    def __init__(self, params):

        self._dim_elems = (params['nelx'], params['nely'])
        self._a0 = params['a0']
        self._b0 = params['b0']
        self.ind_size = 8
    
    
    @staticmethod
    def _dimension(val, scale, max_, min_):
        
        vali = round(float(scale) * val)
        #print(vali, val, min_, max_)
        vali = max_ if vali > max_ else min_ if vali < min_ else vali
        return vali


    def update_topology(self, xs):
        """
        This method takes the design variables of the structure to produce the
        discretized topology that is used for finite element modeling.
        """
        
        scale_a = xs[0] if xs[0] > 0.05 else 0.1
        #scale_b = xs[1] if xs[1] > 0.05 else 0.05
        self.a = self._a0 * scale_a
        self.b = self._b0 * scale_a
        #self.b = self._b0 * scale_b
        
        #print(xs)
        nelx, nely = self._dim_elems
        #p1 = self._dimension(xs[2], nelx, 1)
        #p2 = self._dimension(xs[3], nelx - p1, 1)
        #p3 = self._dimension(xs[4], nelx - p1, 1)
        #p4 = self._dimension(xs[5], nely, 1)
        #p5 = self._dimension(xs[6], nely - p4, 0)
        #p6 = self._dimension(xs[7], nely - p4 - p5, 0)
        p1 = self._dimension(xs[2], nelx, nelx - 1, 1)
        p2 = self._dimension(xs[3], nelx, nelx - p1, 1)
        p3 = self._dimension(xs[4], nelx, nelx - p1, 1)
        p4 = self._dimension(xs[5], nely, nely, 1)
        p5 = self._dimension(xs[6], nely, nely - p4, 0)
        p6 = self._dimension(xs[7], nely, nely - p4 - p5, 0)
        
        #print(nelx, nely)
        #print(p1, p2, p3, p4, p5, p6)
        
        side = np.ones((p1, nely))
        top = np.ones((nelx - p1, p4))
        end_void = np.zeros((nelx - p1, nely - p4 - p5 - p6))
        mass_void = np.zeros((nelx - p1 - p2, p6))
        mass = np.ones((p2, p6))
        link_void = np.zeros((nelx - p1 - p3, p5))
        link = np.ones((p3, p5))
        piece1 = np.vstack((mass_void, mass))
        piece2 = np.vstack((link_void, link))
        piece3 = np.hstack((end_void, piece1, piece2, top))
        topology = np.vstack((side, piece3))
        
        self.topology = np.vstack((topology, np.flipud(topology)))
        self.connectivity_penalty = 0
        self.is_connected = True

   