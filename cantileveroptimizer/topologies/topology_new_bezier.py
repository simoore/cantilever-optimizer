from math import floor
import numpy as np
from collections import namedtuple


Bezier = namedtuple('Bezier', 'eu1 eu2 ex1 ex2 ey1 ey2 rx ry')


class NewBezierTopology(object):
    """
    This topology optimization factory automatrically applies symmetry and 
    connectivity between the base and tip are guaranteed. Compared to the 
    original bezier curve, this class allows the tip to move and doesn't scale
    the size of the finite elements to enforce feature sizes.
    
    Currently, the bezier curves can move beyond the tip, this might need to be 
    restricted.
    
    Public Attributes
    -----------------
    self.ind_size : int
        The number of parameters that describe the topology.
    self.topology : 2d binary ndarray
        A mesh that descibes whether an element is solid or void.
    self.is_connected : bool
        True if the topology is connected.
    self.connection_penalty : float
        A suggest penalization value to use for unconstrained optimization
        algorithms if the topology is unnconnected.
    """
    
    def __init__(self, params):
        
        super().__init__()
        self._num_curves = params['ncurves']
        self._dim_elems = (params['nelx'], params['nely'])
        self._crop = params['crop']
        
        # Public attributes.
        self.a = params['a0']
        self.b = params['b0']
        self.ind_size = 9 * self._num_curves + 2
        self.topology = None
        self.connectivity_penalty = 0.0
        self.is_connected = True
        self.xtip = 0
        self.ytip = 0
        
        self._mscale, self._bscale = self._init_scaling()
        self._x, self._y = self._init_grid()
        
        
    def update_topology(self, xs):
        """
        Coverts the bezier cantilever chromosome to the cantilever topology 
        to be processed by a finite element analysis.
        """
        
        # To place the tip at the top of the domain, the number of elements in 
        # the y-direction need to be reduced. Steps to do this include:
        #   - Reduce the size of the half matrix (optional).
        #   - Change the saturation limits in the _fill_mesh function.
        # OR
        #   - Get tip location at the end and delete wings.
        
        half = np.zeros(self._dim_elems)
        fixed = self._chromosome_scaling(xs)
                
        # For each bezier curve, add the end points to the mesh, then call the 
        # subdivion function which samples the curve and adds the sampled 
        # points to the mesh.
        for i in range(self._num_curves):
            c_slice = slice(9 * i, 9 * i + 11)
            x1, y1, t1, x2, y2, t2, x3, y3, t3, x4, y4 = fixed[c_slice]
            rx = [x1, x2, x3, x4]
            ry = [y1, y2, y3, y4]
            b = Bezier(0, 1, floor(x1), floor(x4), floor(y1), floor(y4), rx, ry)
            t = [floor(t1), floor(t2), floor(t3)]
            self._bezier_subdivision(b, half, t)
            self._fill_mesh(half, 0, floor(x1), floor(y1), t)
            self._fill_mesh(half, 1, floor(x4), floor(y4), t)
        self.topology = np.vstack((np.flipud(half), half))
        
        # Set tip locations.
        nelx, nely = self._dim_elems
        ytip_norm = np.amax(np.argwhere(half[0,:])) + 1
        if self._crop is True:
            self.topology[:, ytip_norm-1] = (np.sum(self.topology[:, ytip_norm-1:nely], axis=1) > 0)
            self.topology[:, ytip_norm:nely] = 0
        self.xtip = 2 * self.a * nelx
        self.ytip = 2 * self.b * (ytip_norm - 0.05)

               
    def _fill_mesh(self, half, u, x, y, t):
        """
        Adds elements to the mesh. The parameters (x,y,t) are floats for 
        optimization and the floor() function should be applied before calling 
        this function.
        
        Parameters
        ----------
        half : 2D binary ndarray
            A binary matrix representing half the topology of the cantilever.
        u : float
            The parametric coordinate of the points to fill.
        x : int
            The x-coordinate of the point to fill.
        y : int
            The y-coordinate of the point to fill.
        t : list of int
            The thickness parameters for each section of the bezier curve. 
            This is given as [t1, t2, t3]. t1 is the thickness parameter for u 
            in (0,0.333), t2 for (0.333,0.666) and t3 for (0.666,1). It is the 
            number of elements to add around the one given.
        """
        
        # If an element outside the mesh is targeted, the closed element inside 
        # the mesh is returned instead.
        def f(z, top): return 0 if z < 0 else (top - 1) if z >= top else z
        
        # Number of elements to add in all directions based on 'u'.
        tt = t[0] if u < 0.333 else t[1] if u < 0.666 else t[2]
        tt = tt if tt >= 1 else 1
        
        # Add elements to the mesh.
        nelx, nely = self._dim_elems
        for i in range(x - tt, x + tt + 1):
            for j in range(y - tt, y + tt + 1):
                half[f(i, nelx), f(j, nely)] = 1
                
    
    def _bezier_subdivision(self, b, half, t):

        if (abs(b.ex1 - b.ex2) <= 1) and (abs(b.ey1 - b.ey2) <= 1):
            return
        u = 0.5 * (b.eu1 + b.eu2)
        w = 1 - u
        coef = [w * w * w, 3 * u * w * w, 3 * u * u * w, u * u * u]
        x = floor(sum([x * y for x, y in zip(coef, b.rx)]))
        y = floor(sum([x * y for x, y in zip(coef, b.ry)]))
        self._fill_mesh(half, u, x, y, t)
        b1 = Bezier(b.eu1, u, b.ex1, x, b.ey1, y, b.rx, b.ry)
        b2 = Bezier(u, b.eu2, x, b.ex2, y, b.ey2, b.rx, b.ry)
        self._bezier_subdivision(b1, half, t)
        self._bezier_subdivision(b2, half, t)
    
    
    def _chromosome_scaling(self, chromosome):
        """
        All design variables are in the range [-1,1]. These need to be scaled
        appropriately. In addition, certain variables need to be fixed to 
        ensure connectivity.
        """
        
        scaled = self._mscale * chromosome + self._bscale
        _, nely = self._dim_elems
        
        # Add excluded parameters back in before topology generation.
        # The y-coord of the first point is at the base.
        # The x-coord of the last point is at the tip.
        scaled[1] = 0    
        scaled[-2] = 0
        
        return scaled
    
    
    def _init_scaling(self):
        """
        Generate the lower bound and upper bound for parameters. This is used
        for initialization of the structure. Note that some stochastic 
        algorithms cannot place constraints on the design variables therefore
        this topology factory must apply the constraints when generating the 
        topology.
        """

        min_t = 1
        max_t = 5
        nelx, nely = self._dim_elems
        lb = np.zeros(self.ind_size)
        ub = np.zeros(self.ind_size)
        lower_slice = [0, 0, min_t]
        upper_slice = [nelx, nely, max_t]
        for i in range(self.ind_size):
            lb[i] = lower_slice[i % 3]
            ub[i] = upper_slice[i % 3]

        mscale = 0.5 * (ub - lb)
        bscale = mscale + lb
        
        return mscale, bscale
    
    
    def _init_grid(self):
        """
        The topology is formulated on a normalised mesh where elements are
        1x1 units. The aspect ratio of the elements is defined by (a,b) and
        is only taken into account in the finite element model.
        """
        
        nelx, nely = self._dim_elems
        ii, jj = np.meshgrid(np.arange(nelx), np.arange(nely))
        xx = ii.T + 0.5
        yy = jj.T + 0.5
        return xx, yy
    