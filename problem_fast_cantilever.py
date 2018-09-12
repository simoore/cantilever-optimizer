import sys

civ_path = 'D:\\doc\\Google Drive\\Github\\micro-fem'
if civ_path not in sys.path:
    sys.path.append(civ_path)
    
import numpy as np
import microfem
from analysis_plate_displacement import PlateDisplacement
from analysis_mode_identification import ModeIdentification


class FastCantileverProblem(object):
    """
    Public Attributes
    -----------------
    self.ind_size
    self.name
    """
    
    def __init__(self, params, topology_factory):
        
        self.k1 = params['k1']
        self.topology_factory = topology_factory
        self.material = microfem.SoiMumpsMaterial()
        self.ind_size = self.topology_factory.ind_size
        self.name = '--- Fast Cantilever Optimization ---'
        
        
    def objective_function(self, xs):

        self.topology_factory.update_topology(xs)
        
        if self.topology_factory.is_connected is True:
            
            params = self.topology_factory.get_params()
            cantilever = microfem.Cantilever(*params)
            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = PlateDisplacement(fem, coords).get_operator()
            mode_ident = ModeIdentification(fem, cantilever)
            
            try:
                w, _, vall = fem.modal_analysis(1)
            except RuntimeError:
                print('singular')
            kuu = fem.get_stiffness_matrix(free=False)
            f1 = np.asscalar(np.sqrt(w) / (2*np.pi))
            phi1 = vall[:, [0]]
            wtip1 = np.asscalar(opr @ phi1)
            kfunc = lambda p, w: np.asscalar(p.T @ kuu @ p / w ** 2)
            k1 = kfunc(phi1, wtip1)
            type1 = mode_ident.is_mode_flexural(phi1) 
            
            if type1 is False:
                cost = 1e8
            else:
                cost = -f1*1e-6 if k1 < self.k1 else k1
            
            return (cost,)
        
        return (self.topology_factory.connectivity_penalty,)
    
    
    def console_output(self, xopt, image_file):
        

        self.topology_factory.update_topology(xopt)
        topology = self.topology_factory.topology
        a = self.topology_factory.a
        b = self.topology_factory.b
        print('The element dimensions are (um): %gx%g' % (2e6*a, 2e6*b))
        xtip = self.topology_factory.xtip
        ytip = self.topology_factory.ytip
        cantilever = microfem.Cantilever(topology, a, b, xtip, ytip)
        microfem.plot_topology(cantilever, image_file)
        
        if self.topology_factory.is_connected is True:
            
            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = PlateDisplacement(fem, coords).get_operator()
            mode_ident = ModeIdentification(fem, cantilever)
            
            w, _, vall = fem.modal_analysis(3)
            freq = np.sqrt(w) / (2*np.pi)
            kuu = fem.get_stiffness_matrix(free=False)
            phis = [vall[:, [i]] for i in range(3)]
            wtips = [opr @ p for p in phis]
            kfunc = lambda p, w: np.asscalar(p.T @ kuu @ p / w ** 2)
            ks = [kfunc(p, w) for p, w in zip(phis, wtips)]
            types = [mode_ident.is_mode_flexural(p) for p in phis]
            
            tup = ('Disp', 'Freq (Hz)', 'Stiffness', 'Flexural')
            print('\n    %-15s %-15s %-15s %-10s' % tup)
            for i in range(3):
                tup = (i, wtips[i], freq[i], ks[i], str(types[i]))
                print('%-2d: %-15g %-15g %-15g %-10s' % tup)

            for i in range(3):
                microfem.plot_mode(fem, vall[:, i])
