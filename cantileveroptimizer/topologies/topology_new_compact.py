import numpy as np
from ..analysis_connectivity import connectivity_penalization


class NewCompactTopology(object):
    """
    References:
        
    Radial basis functions and level set method for structural topology 
    optimization, by Shengyin Wang and Michael Yu Wang, in Numerical Methods in 
    Engineering Vol. 65(12) 2005.
    
    Level-set methods for structural topology optimization: a review, by van 
    Dijk, N. P.; Maute, K.; Langelaar, M. & van Keulen, F., in Structural and 
    Multidisciplinary Optimization, 2013, 48, 437-472  
    
    Since the tip location can change, the 'wings' of the structure are not 
    removed from this structure. If the wings are inappropriate, use a regular
    parameterization instead. Or use the original compact topology which fixes
    the location of the tip. 
    """

    # Add fixed tip location to this class as an option.

    def __init__(self, params):
        """
        Knots are the location at which the level-set interpolation function is
        defined exactly. The level-set interpolation function is defined by a 
        uniformly spaced rectangular grid of knots.
        """
        self._dim_elems = (params['nelx'], params['nely'])
        self._dim_knots = (params['nknx'], params['nkny'])
        self._a = 1
        self._b = 1
        self._ind_size = self._dim_knots[0] * self._dim_knots[1]
        self._pcon1 = params['pcon1']
        self._pcon2 = params['pcon2']
        self._ratio = params['support_ratio']

        # Initialise the coordinates of the knots and elements
        self._init_knot_coords()
        self._init_elem_coords()
        self._gmat = self._gaussian(self._xelems, self._yelems)

        self._areal = params['a0']
        self._breal = params['b0']
        self._xtip, self._ytip = 0, 0
        self._topology = None
        self._connectivity_penalty = 0
        self._is_connected = False

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

    @property
    def a(self):
        return self._areal

    @property
    def b(self):
        return self._breal

    @property
    def xtip(self):
        return self._xtip

    @property
    def ytip(self):
        return self._ytip

    def update_topology(self, xs):
        """
        This method takes the design variables of the structure to produce the
        discretized topology that is used for finite element modeling.
        
        :param xs:  1d ndarray containing the interpolation data values at the 
                    knot locations.
        """
        # Topology generation.
        alpha = np.atleast_2d(np.array(xs)).T
        lsf = self._gmat @ alpha
        lsf_2d = lsf.reshape(self._dim_elems, order='F')
        topology = lsf_2d > 0
        self._topology = self._apply_regularization(topology)

        # Connectivity penalization.
        n_isle, dist = connectivity_penalization(self._topology)
        self._connectivity_penalty = self._pcon1 * n_isle + self._pcon2 * dist
        self._is_connected = True if n_isle == 0 else False

        # Tip location.
        if self._is_connected is True:
            nelx, nely = self._dim_elems
            ytip_norm = np.amax(np.argwhere(topology[-1, :])) + 1
            self._xtip = 2 * self.a * nelx
            self._ytip = 2 * self.b * (ytip_norm - 0.05)

    def get_params(self):
        return (self._topology,
                self._areal,
                self._breal,
                self._xtip,
                self._ytip)

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
        xc = np.linspace(0, (nknx - 1) * dx, nknx)
        yc = np.linspace(0, (nkny - 1) * dy, nkny)
        xv, yv = np.meshgrid(xc, yc)
        self._xcoords = xv.ravel()
        self._ycoords = yv.ravel()

        h = max((dx, dy))  # Distance between knots
        s = self._ratio * h  # Measure of support size
        e = 0.25 * s
        self._kparam = 1 / (e * np.sqrt(2 * np.pi))
        self._eparam = -0.5 / (e ** 2)

        print()
        print('--- Compact Topology Parameterization ---')
        print('Knot X-Separation: %g' % dx)
        print('Knot Y-Separation: %g' % dy)
        print('Support Size: %g' % s)
        print('Number of Parameters: %d' % self._ind_size)

    def _init_elem_coords(self):
        """
        Computes the center coordinate of the elements in the finite elemement
        mesh.
        """

        nelx, nely = self._dim_elems
        dx = 2 * self._a
        dy = 2 * self._b
        xc = np.arange(self._a, nelx * dx, dx)
        yc = np.arange(self._b, nely * dy, dy)
        xv, yv = np.meshgrid(xc, yc)
        self._xelems = xv.ravel()
        self._yelems = yv.ravel()

    @staticmethod
    def _apply_regularization(topology, symmetry=True):
        """
        The topology is an arbitary 2D design. This function applies a mask to 
        remove elements in void zones, mirrors the topology to enfornce 
        symmetry and adds elements at elements at the tip.
        """

        if symmetry is True:
            topology = np.vstack((topology, np.flipud(topology)))

        return topology
