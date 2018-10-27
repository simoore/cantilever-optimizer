import sys

civ_path = 'D:\\doc\\Google Drive\\Github\\micro-fem'
if civ_path not in sys.path:
    sys.path.append(civ_path)
    
import numpy as np
import microfem


class FrequencyProblem(object):
    
    def __init__(self, params, topology_factory):
        
        self.f0 = params['f0']
        self.a = 5e-6
        self.b = 5e-6
        self.topology_factory = topology_factory
        self.material = microfem.SoiMumpsMaterial()
    
    
    @property
    def ind_size(self):
        return (self.topology_factory.ind_size + 1)
    
    
    @property
    def name(self):
        return '--- Frequency Placement Optimization ---'
    
    
    def objective_function(self, xs):
        """
        Evaluates the objective function. The objective function calculates 
        the ratio between the dynamic stiffness of the flexural modes with
        respect to the first.
        
        Parameters
        ----------
        xs : list of floats      
            The optimization design variables for a solution.
            
        Returns
        -------
        : float       
            The evaluated objective for the solution.
        """
        top_vars, scale_var = xs[0:-1], xs[-1]
        self.topology_factory.update_topology(np.array(top_vars))
        
        if self.topology_factory.is_connected is True:
            
            topology = self.topology_factory.topology
            a = self.a * abs(scale_var)
            b = self.b * abs(scale_var)
            cantilever = microfem.Cantilever(topology, a, b)
            fem = microfem.PlateFEM(self.material, cantilever)
            w, _, _ = fem.modal_analysis(1)
            f = np.asscalar(np.sqrt(w) / (2*np.pi))
            cost = abs(f - self.f0)
            
            return (cost,)
        
        return (self.topology_factory.connectivity_penalty,)
    
    
    def console_output(self, xopt, image_file):
        
        top_vars, scale_var = xopt[0:-1], xopt[-1]
        self.topology_factory.update_topology(top_vars)
        topology = self.topology_factory.topology
        a = self.a * abs(scale_var)
        b = self.b * abs(scale_var)
        print('The element dimensions are (um): %gx%g' % (2e6*a, 2e6*b))
        cantilever = microfem.Cantilever(topology, a, b)
        microfem.plot_topology(cantilever, image_file)
        
        if self.topology_factory.is_connected is True:
            
            fem = microfem.PlateFEM(self.material, cantilever)
            w, _, vall = fem.modal_analysis(1)
            f = np.asscalar(np.sqrt(w) / (2 * np.pi))
            print('The first modal frequency is (Hz): %g' % f)
            microfem.plot_mode(fem, vall[:, 0])
