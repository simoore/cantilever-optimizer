import sys

civ_path = 'D:\\doc\\Google Drive\\Github\\micro-fem'
if civ_path not in sys.path:
    sys.path.append(civ_path)
    
import numpy as np
import microfem
from analysis_plate_displacement import PlateDisplacement
from analysis_mode_identification import ModeIdentification


class OffResonanceProblem(object):
    
    def __init__(self, params, topology_factory):
        
        self.topology_factory = topology_factory
        self.material = microfem.SoiMumpsMaterial()
        self.n_modes = 1
        self.ind_size = self.topology_factory.ind_size
        self.name = '--- Off Resonance Cantilever Optimization ---'

    
    def objective_function(self, xs):
        
        self.topology_factory.update_topology(xs)
        a = self.topology_factory.a
        b = self.topology_factory.b

        if self.topology_factory.is_connected is True:
            
            topology = self.topology_factory.topology
            cantilever = microfem.Cantilever(topology, a, b)
            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = PlateDisplacement(fem, coords).get_operator()
            mode_ident = ModeIdentification(fem, cantilever)
            
            w, _, vall = fem.modal_analysis(self.n_modes)
            kuu = fem.get_stiffness_matrix(free=False)
            phi = vall[:, [0]] 
            wtip = opr @ phi
            f1 = np.asscalar(np.sqrt(w) / (2*np.pi))
            k1 = np.asscalar(phi.T @ kuu @ phi / wtip ** 2)
            type_ = mode_ident.is_mode_flexural(phi)
            cost = (-f1, k1) if type_ is True else (1e8, 1e8)
            return cost
        
        return (self.topology_factory.connectivity_penalty,
                self.topology_factory.connectivity_penalty)
    
    
    def console_output(self, xopt, image_file):
        
        n_modes = self.n_modes + 1
        self.topology_factory.update_topology(xopt)
        a = self.topology_factory.a
        b = self.topology_factory.b
        topology = self.topology_factory.topology
        print('The element dimensions are (um): %gx%g' % (2e6*a, 2e6*b))
        cantilever = microfem.Cantilever(topology, a, b)
        microfem.plot_topology(cantilever, image_file)
        
        if self.topology_factory.is_connected is True:
            
            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = PlateDisplacement(fem, coords).get_operator()
            mode_ident = ModeIdentification(fem, cantilever)
            
            w, _, vall = fem.modal_analysis(n_modes)
            freq = np.sqrt(w) / (2*np.pi)
            kuu = fem.get_stiffness_matrix(free=False)
            phis = [vall[:, [i]] for i in range(n_modes)]
            wtips = [opr @ p for p in phis]
            kfunc = lambda p, w: np.asscalar(p.T @ kuu @ p / w ** 2)
            ks = [kfunc(p, w) for p, w in zip(phis, wtips)]
            types = [mode_ident.is_mode_flexural(p) for p in phis]
            
            tup = ('Disp', 'Freq (Hz)', 'Stiffness', 'Flexural')
            print('\n    %-15s %-15s %-15s %-10s' % tup)
            for i in range(n_modes):
                tup = (i, wtips[i], freq[i], ks[i], str(types[i]))
                print('%-2d: %-15g %-15g %-15g %-10s' % tup)

            for i in range(n_modes):
                microfem.plot_mode(fem, vall[:, i])
                