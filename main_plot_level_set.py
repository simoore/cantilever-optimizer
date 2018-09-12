import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def main():
    
    filename = 'solutions/prelim_3_gau_ls_ratio_3-design.npy'
    p = PlotCompactBasisFunctions(filename)
    p.plot()
    

class PlotCompactBasisFunctions(object): 
    
    def __init__(self, filename):

    
        self.nelx = 20
        self.nely = 40
        self.nknx = 5
        self.nkny = 10
        self.a = 5e-6
        self.b = 5e-6
        self.filename = filename
        self.ratio = 2
        
        self._dim_elems = (self.nelx, self.nely)
        self._dim_knots = (self.nknx, self.nkny)
        #self._dim_nodes = (self.nelx + 1, self.nely + 1)
        self._a = self.a * 1e6
        self._b = self.b * 1e6
        self._ind_size = self.nknx * self.nkny
        
        # Initialise the coordinates of the knots and elements
        self._init_knot_coords()
        self._init_node_coords()
        self._init_basis_parameters()
        
        self._gmat = self._gaussian(self._xnodes, self._ynodes)
    
    
    def plot(self):
        
        xs = np.load(self.filename)
        xs = np.ones(50)
        #print(xs)
        z_1d = self._gmat @ xs
        x_2d = self._xnodes.reshape(self._dim_nodes, order='F')
        y_2d = self._ynodes.reshape(self._dim_nodes, order='F')
        z_2d = z_1d.reshape(self._dim_nodes, order='F')
        zz_2d = np.zeros(z_2d.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_2d, y_2d, z_2d, cmap=cm.viridis)
        ax.plot_surface(x_2d, y_2d, zz_2d, color='black')
        ax.set_xlabel('Boundary')


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
        
    
    def _init_node_coords(self):
        
        nelx, nely = self._dim_elems
        dx = 2 * self._a
        dy = 2 * self._b
        xc = np.arange(0, nelx*dx, 0.1*dx)
        yc = np.arange(0, nely*dy, 0.1*dy)
        #print(xc, yc)
        self._dim_nodes = (len(xc), len(yc))
        xv, yv = np.meshgrid(xc, yc)
        self._xnodes = xv.ravel()
        self._ynodes = yv.ravel()
        
        
    def _init_basis_parameters(self):
        
        nelx, nely = self._dim_elems
        nknx, nkny = self._dim_knots
        dx = 2 * self._a * nelx / (nknx + 1)
        dy = 2 * self._b * nely / (nkny + 1)
        r = self.ratio      # Support size ratio
        h = max((dx, dy))   # Distance between knots
        s = r*h             # Measure of support size
        e = 0.25 * s
        self._kparam = 1 / (e*np.sqrt(2*np.pi))
        self._eparam = -0.5 / (e ** 2)
        
        
if __name__ == '__main__':
    
    main()
    