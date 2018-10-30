import numpy as np
import microfem


class FrequencyProblem(object):
    
    def __init__(self, params, topology_factory):
        
        self.f0 = params['f0']
        self.topology_factory = topology_factory
        self.material = microfem.SoiMumpsMaterial()
    
    
    @property
    def ind_size(self):
        return (self.topology_factory.ind_size)
    
    
    @property
    def name(self):
        return '--- Frequency Placement Optimization ---'
    
    
    def objective_function(self, xs):

        self.topology_factory.update_topology(xs)
        
        if self.topology_factory.is_connected is True:
            
            params = self.topology_factory.get_params()
            cantilever = microfem.Cantilever(*params)
            fem = microfem.PlateFEM(self.material, cantilever)
            w, _, _ = fem.modal_analysis(1)
            f = np.asscalar(np.sqrt(w) / (2*np.pi))
            cost = abs(f - self.f0)
            
            return (cost,)
        
        return (self.topology_factory.connectivity_penalty,)
    
    
    def console_output(self, xopt, image_file):
        
        self.topology_factory.update_topology(xopt)
        params = self.topology_factory.get_params()
        cantilever = microfem.Cantilever(*params)
        microfem.plot_topology(cantilever, image_file)
        
        if self.topology_factory.is_connected is True:
            
            fem = microfem.PlateFEM(self.material, cantilever)
            w, _, vall = fem.modal_analysis(1)
            f = np.asscalar(np.sqrt(w) / (2 * np.pi))
            print('The first modal frequency is (Hz): %g' % f)
            microfem.plot_mode(fem, vall[:, 0])
