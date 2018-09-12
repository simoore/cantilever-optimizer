import numpy as np
from analysis_connectivity import connectivity_penalization


class CompactTopology(object):
    """
    References:
        
    Radial basis functions and level set method for structural topology 
    optimization, by Shengyin Wang and Michael Yu Wang, in Numerical Methods in 
    Engineering Vol. 65(12) 2005.
    
    Level-set methods for structural topology optimization: a review, by van 
    Dijk, N. P.; Maute, K.; Langelaar, M. & van Keulen, F., in Structural and 
    Multidisciplinary Optimization, 2013, 48, 437-472  
    """
    
    def __init__(self, dim_knots, dim_elems, a, b, penalty_factors, ratio):
        """
        Knots are the location at which the level-set interpolation function is
        defined exactly. The level-set interpolation function is defined by a 
        uniformly spaced rectangular grid of knots.
        
        Parameters
        ----------
        dim_knots : tuple
            Tuple with the number of knots in the x- and y-directions.
        dim_elems : tuple 
            Tuple with the number of elements in the x- and y-direction.
        a : float 
            Half the width of an element in metres.
        b : float 
            Half the length of an element in metres.
        penalty_factors : tuple 
            Tuple with the two penalty factors for an  unconnected topology.
        """
        
        self._dim_elems = dim_elems  # (nelx, nely)
        self._dim_knots = dim_knots  # (nknx, nkny)
        self._a = a * 1e6
        self._b = b * 1e6
        self._ind_size = dim_knots[0] * dim_knots[1]
        self._pcon1, self._pcon2 = penalty_factors
        self._ratio = ratio
        
        # Initialise the coordinates of the knots and elements
        self._init_knot_coords()
        self._init_elem_coords()
        
        self._gmat = self._gaussian(self._xelems, self._yelems)


    @property
    def topology(self):
        return self._topology
    
    
    @property
    def ind_size(self):
        return self._ind_size
    
    
    @property
    def is_connected(self):
        return self._is_connected
    
    
    @property
    def connectivity_penalty(self):
        return self._connectivity_penalty
        
        
    def update_topology(self, xs):
        """
        This method takes the design variables of the structure to produce the
        discretized topology that is used for finite element modeling.
        
        :param xs:  1d ndarray containing the interpolation data values at the 
                    knot locations.
        """
        
        alpha = np.atleast_2d(xs).T
        lsf = self._gmat @ alpha
        lsf_2d = lsf.reshape(self._dim_elems, order='F')
        topology = lsf_2d > 0
        self._topology = self._apply_regularization(topology)
        n_isle, dist = connectivity_penalization(self._topology)
        self._connectivity_penalty = self._pcon1 * n_isle + self._pcon2 * dist
        self._is_connected = True if n_isle == 0 else False


    def _gaussian(self, x, y):
        """
        Evaluates the basis functions for points (x,y). 
        
        Parameters
        ----------
        x : 1d ndarray
            The x-coordinates of points to evaluate the basis functions at.
        y : 1d ndarray 
            The y-coordinates of points to evaluate the basis functions at.
        
        Returns
        -------
        : 2d ndarray
            A matrix with the evaluations of the basis functions. The rows are 
            evaluated for each (x,y) pair and the columns are associated with 
            each basis function.
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
        basis = self._kparam * np.exp(norm_squared * self._eparam)
        return basis
    

    def _init_knot_coords(self):
        """ 
        Calculate the coordinates of the knots.
        """
        
        nelx, nely = self._dim_elems
        nknx, nkny = self._dim_knots
        dx = 2 * self._a * nelx / (nknx - 1)
        dy = 2 * self._b * nely / (nkny - 1)
        xc = np.linspace(0, (nknx-1)*dx, nknx)
        yc = np.linspace(0, (nkny-1)*dy, nkny)
        xv, yv = np.meshgrid(xc, yc)
        self._xcoords = xv.ravel()
        self._ycoords = yv.ravel()
        
        h = max((dx, dy))   # Distance between knots
        s = self._ratio * h # Measure of support size
        e = 0.25 * s
        self._kparam = 1 / (e*np.sqrt(2*np.pi))
        self._eparam = -0.5 / (e ** 2)
        
        print()
        print('--- Compact Topology Parameterization ---')
        print('Knot X-Separation: %g' % dx)
        print('Knot Y-Separation: %g' % dy)
        print('Support Size: %g' % s)
    
    
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
        
        
    @staticmethod    
    def _apply_regularization(topology, mask=None, symmetry=True, tip=True):
        """
        The topology is an arbitary 2D design. This function applies a mask to 
        remove elements in void zones, mirrors the topology to enfornce 
        symmetry and adds elements at elements at the tip.
        """

        if mask is not None:
            new_topology = np.logical_and(topology, mask) 
        else:
            new_topology = topology
            
    
        if symmetry is True:
            new_topology = np.vstack((new_topology, np.flipud(new_topology)))
        
        # The tip is located along the center axis of the cantilever, half an
        # element back (y-direction) from the end of the design space.
        # xtip = self._a * nelx
        # ytip = 2 * nely * self._b - self._b
        if tip is True:
            nelx, nely = new_topology.shape
            if nelx % 2 == 0:
                new_topology[round(nelx/2-1), nely-1] = True
                new_topology[round(nelx/2), nely-1] = True
            else:
                new_topology[round((nelx-1)/2),nely-1] = True
                
        return new_topology
        