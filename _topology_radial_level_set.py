import numpy as np
import scipy.linalg as linalg


class RadialLevelSetTopology(object):
    """
    References:
        
    Radial basis functions and level set method for structural topology 
    optimization, by Shengyin Wang and Michael Yu Wang, in Numerical Methods in 
    Engineering Vol. 65(12) 2005.
    
    Level-set methods for structural topology optimization: a review, by van 
    Dijk, N. P.; Maute, K.; Langelaar, M. & van Keulen, F., in Structural and 
    Multidisciplinary Optimization, 2013, 48, 437-472  
    """
    
    def __init__(self, nknx, nkny, nelx, nely, a, b, mq_basis=True):
        """
        Knots are the location at which the level-set interpolation function is
        defined exactly. The level-set interpolation function is defined by a 
        uniformly spaced rectangular grid of knots.
        
        :param nknx:    The number of knots in the x-direction.
        :param nkny:    The number of knots in the y-direction.
        :param nelx:    The number of elements in the x-direction.
        :param nely:    The number of elements in the y-direction.
        :param a:       Half the width of an element in metres.
        :param b:       Half the length of an element in metres.
        mq_basis : bool
            If true mq_spline basis functions are employed in the level set 
            formulation. Else gaussian basis functions are employed.
        """
        
        #self._cparam = c_param
        self._dim_elems = (nelx, nely)
        self._dim_knots = (nknx, nkny)
        self._a = a * 1e6
        self._b = b * 1e6
        self._topology = None
        
        # Initialise the coordinates of the knots and elements
        self._init_knot_coords()
        self._init_elem_coords()
        
        basis = self._mq_spline if mq_basis is True else self._gaussian
        
        # Initialise H matrix.
        pmat = self._aux(self._xcoords, self._ycoords)
        amat = basis(self._xcoords, self._ycoords)
        row1 = np.hstack((amat, pmat))
        row2 = np.hstack((pmat.T, np.zeros((3, 3))))
        self._hmat = np.vstack((row1, row2))
        
        # Initialise G matrix.
        pmat = self._aux(self._xelems, self._yelems)
        amat = basis(self._xelems, self._yelems)
        self._gmat = np.hstack((amat, pmat))
        
    
    @property
    def topology(self):
        return self._topology
    
    @property
    def ind_size(self):
        return self._dim_knots[0] * self._dim_knots[1]
    
    
    def update_topology(self, xs):
        """
        This method takes the design variables of the structure to produce the
        discretized topology that is used for finite element modeling.
        
        :param xs:  1d ndarray containing the interpolation data values at the 
                    knot locations.
        """
        
        xs_col = np.atleast_2d(xs).T
        f = np.vstack((xs_col, np.zeros((3, 1))))
        alpha = linalg.solve(self._hmat, f)
        topology = self._direct_mapping(alpha)
        self._topology = topology
        
    
    def _mq_spline(self, x, y):
        """
        Evaluates the basis functions for points (x,y). 
        
        :param x:   1d ndarray with the x-coordinates of points to evaluate 
                    the basis functions at.
        :param y:   1d ndarray with the y-coordinates of points to evaluate
                    the basis functions at.
        :returns:   A matrix with the evaluations of the basis functions. The
                    rows are evaluated for each (x,y) pair and the columns are
                    associated with each basis function.
        """
        # Put the corresponding (x,y) values and basis function parameters 
        # into a matrix form.
        n_vals = x.shape[0]
        n_basis = self._xcoords.shape[0]
        
        xmat = np.tile(np.atleast_2d(x).T, (1, n_basis))
        ximat = np.tile(self._xcoords, (n_vals, 1))
        ymat = np.tile(np.atleast_2d(y).T, (1, n_basis))
        yimat = np.tile(self._ycoords, (n_vals, 1))
        
        # Evaluate the basis functions.
        norm_squared = (xmat - ximat) ** 2 + (ymat - yimat) ** 2
        return np.sqrt(norm_squared + self._cparam ** 2)
    
    
    def _gaussian(self, x, y):
        """
        Evaluates the basis functions for points (x,y). 
        
        :param x:   1d ndarray with the x-coordinates of points to evaluate 
                    the basis functions at.
        :param y:   1d ndarray with the y-coordinates of points to evaluate
                    the basis functions at.
        :returns:   A matrix with the evaluations of the basis functions. The
                    rows are evaluated for each (x,y) pair and the columns are
                    associated with each basis function.
        """
        # Put the corresponding (x,y) values and basis function parameters 
        # into a matrix form.
        n_vals = x.shape[0]
        n_basis = self._xcoords.shape[0]
        
        xmat = np.tile(np.atleast_2d(x).T, (1, n_basis))
        ximat = np.tile(self._xcoords, (n_vals, 1))
        ymat = np.tile(np.atleast_2d(y).T, (1, n_basis))
        yimat = np.tile(self._ycoords, (n_vals, 1))
        
        # Evaluate the basis functions.
        norm_squared = (xmat - ximat) ** 2 + (ymat - yimat) ** 2
        return np.exp(norm_squared / (self._dparam ** 2))
    
    
    def _aux(self, x, y):
        
        a = np.ones((x.shape[0], 1))
        b = np.atleast_2d(x).T
        c = np.atleast_2d(y).T
        pmat = np.hstack((a, b, c))
        return pmat
    
    
    def _direct_mapping(self, alpha):
        """
        The simplest way to map the level-set function to a discretized 
        topology is to take the level-set function value at the center of the 
        element and compare it to the threshold to detect whether the element
        is void or solid.
        """
        
        lsf = self._gmat @ alpha
        lsf_2d = lsf.reshape(self._dim_elems, order='F')
        topology = lsf_2d < 0
        return topology
    
    
    def _init_knot_coords(self):
        """ 
        Calculate the coordinates of the knots.
        """
        
        nelx, nely = self._dim_elems
        nknx, nkny = self._dim_knots
        dx = 2 * self._a * nelx / (nknx + 1)
        dy = 2 * self._b * nely / (nkny + 1)
        xc = np.linspace(dx, nknx*dx, nknx)
        yc = np.linspace(dy, nkny*dy, nkny)
        xv, yv = np.meshgrid(xc, yc)
        self._xcoords = xv.ravel()
        self._ycoords = yv.ravel()
        self._dparam = min((dx, dy))
        self._cparam = 1 / min((dx, dy))
    
    
    def _init_elem_coords(self):
        """
        Computes the center coordinate of the elements in the finite elemement
        mesh.
        """
        
        nelx, nely = self._dim_elems
        dx = 2 * self._a
        dy = 2 * self._b
        xc = np.arange(self._a, nelx*dx, dx)
        yc = np.arange(self._b, nely*dy, dy)
        xv, yv = np.meshgrid(xc, yc)
        self._xelems = xv.ravel()
        self._yelems = yv.ravel()
        