import unittest
import numpy as np
from topology_radial_level_set import RadialLevelSetTopology



class TestRadialLevelSetTopology(unittest.TestCase):
    
    def test_all(self):

        c_param = 1e-15
        rlst = RadialLevelSetTopology(2, 2, 3, 4, 5e-6, 5e-6, c_param)
        self.assertTrue(rlst._a == 5)
        
        
        # Test initialization of element coordinates.
        xc = np.array([5, 15, 25, 5, 15, 25, 5, 15, 25, 5, 15, 25])
        yc = np.array([5, 5, 5, 15, 15, 15, 25, 25, 25, 35, 35, 35])
        self.assertTrue(np.all(xc == rlst._xelems))
        self.assertTrue(np.all(yc == rlst._yelems))
        
        
        # Test initialization of knot coordinates.
        xd = np.array([10, 20, 10, 20])
        yd = np.array([40/3, 40/3, 80/3, 80/3])
        self.assertTrue(np.all(xd == rlst._xcoords))
        self.assertTrue(np.all(yd == rlst._ycoords))
        
        
        # Test initialization of hmat.
        amat = np.zeros((4, 4))
        for i, j in np.ndindex(4, 4):
            r2 = (xd[i] - xd[j]) ** 2 + (yd[i] - yd[j]) ** 2
            amat[i, j] = np.sqrt(r2 + c_param ** 2)
        pmat = np.array([[1, 10, 40/3], [1, 20, 40/3], 
                         [1, 10, 80/3], [1, 20, 80/3]])
        zmat = np.zeros((3, 3))
        hmat = np.vstack((np.hstack((amat, pmat)), np.hstack((pmat.T, zmat))))
        self.assertTrue(np.all(hmat == rlst._hmat))
        
        
        # Test initialization of gmat.
        amat = np.zeros((12, 4))
        pmat = np.zeros((12, 3))
        for i, j in np.ndindex(12, 4):
            r2 = (xc[i] - xd[j]) ** 2 + (yc[i] - yd[j]) ** 2
            amat[i, j] = np.sqrt(r2 + c_param ** 2)
        for i in range(12):
            pmat[i, 0] = 1
            pmat[i, 1] = xc[i]
            pmat[i, 2] = yc[i]
        gmat = np.hstack((amat, pmat))
        self.assertTrue(np.all(gmat == rlst._gmat))
        
        
        # Lets test rashape puts coordinates of elements back in correct 
        # position.
        xcc = np.atleast_2d(xc).T.reshape(rlst._dim_elems, order='F')
        ycc = yc.reshape(rlst._dim_elems, order='F')
        xccc = np.array([[5, 5, 5, 5], [15, 15, 15, 15], [25, 25, 25, 25]])
        yccc = np.array([[5, 15, 25, 35], [5, 15, 25, 35], [5, 15, 25, 35]])
        self.assertTrue(np.all(xcc == xccc))
        self.assertTrue(np.all(ycc == yccc))
        
        # Lets test a topology that is biased towards x=0.
        # Remember the tip of the AFM cantilever is added afte the fact.
        f = np.array([0.1, 0.25, -0.2, -0.3])
        t1 = np.array([[1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 0, 0]])
        rlst.update_topology(f)
        self.assertTrue(np.all(t1 == rlst.topology))
        
        
        # Lets test a topology that is biased towards y=0.
        # Remember the tip of the AFM cantilever is added afte the fact.
        f = np.array([0.1, -0.25, 0.2, -0.3])
        t2 = np.array([[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
        rlst.update_topology(f)
        self.assertTrue(np.all(t2 == rlst.topology))
        
        
if __name__ == '__main__':
    unittest.main()
