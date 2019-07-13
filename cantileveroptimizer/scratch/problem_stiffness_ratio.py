import sys

civ_path = 'D:\\doc\\Google Drive\\Github\\micro-fem'
if civ_path not in sys.path:
    sys.path.append(civ_path)
    
import microfem
import numpy as np
from analysis_plate_displacement import PlateDisplacement
from analysis_mode_identification import ModeIdentification


class StiffnessRatioProblem(object):
    
    def __init__(self, params, topology_factory):
        
        self.a = 5e-6
        self.b = 5e-6
        self.topology_factory = topology_factory
        self.material = microfem.SoiMumpsMaterial()
        self.n_modes = params['nmodes']
        self.n_ratio = params['nratio']
    
    
    @property
    def ind_size(self):
        return (self.topology_factory.ind_size)
    
    
    @property
    def name(self):
        return '--- Stiffness Ratio Optimization ---'
    
    def objective_function(self, xs):
        pass
#        self.fem.update_mesh(cantilever)
#        coords = (cantilever.xtip, cantilever.ytip)
#        pd = PlateDisplacement(self.fem, coords)
#        opr = pd.get_operator()
#        mode_ident = ModeIdentification(self.fem, cantilever)
#        
#        # Perform modal analysis and retrieve the stiffness matrix.
#        w, _, vall = self.fem.modal_analysis(self.n_modes)
#        kuu = self.fem.get_stiffness_matrix(free=False)
#        
#        # Analyze the first mode.
#        phi1 = vall[:, [0]]
#        wtip1 = opr @ phi1
#        k1 = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
#        
#        cost, ratio_index, ratios = 0, 0, [2, 3, 4]
#        for i in range(1, self.n_modes):
#            if ratio_index < (self.n_ratio-1):
#                phi = vall[:, [i]]
#                if mode_ident.is_mode_flexural(phi) == True:
#                    wtip = opr @ phi
#                    k = np.asscalar(phi.T @ kuu @ phi / wtip ** 2)
#                    cost += (k/k1 - ratios[ratio_index]) ** 2
#                    ratio_index += 1
#                    
#        return (cost,)
    
    
    def console_output(self, xopt, image_file):
        
        self.topology_factory.update_topology(xopt)
        topology = self.topology_factory.topology
        cantilever = microfem.Cantilever(topology, self.a, self.b)
        microfem.plot_topology(cantilever, image_file)
        
        if self.topology_factory.is_connected is True:
            
            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = PlateDisplacement(fem, coords).get_operator()
            mode_ident = ModeIdentification(fem, cantilever)
            
            w, _, vall = fem.modal_analysis(self.n_modes)
            freq = np.sqrt(w) / (2*np.pi)
            kuu = fem.get_stiffness_matrix(free=False)
            phis = [vall[:, [i]] for i in range(self.n_modes)]
            wtips = [opr @ p for p in phis]
            ks = [np.asscalar(p.T @ kuu @ p / w ** 2) for p, w in zip(phis, wtips)]
            costs = [k / ks[0] for k in ks]
            types = [mode_ident.is_mode_flexural(p) for p in phis]
            
            tup = ('Disp', 'Freq (Hz)', 'Stiffness', 'Ratio', 'Flexural')
            print('\n    %-15s %-15s %-15s %-15s %-10s' % tup)
            for i in range(self.n_modes):
                tup = (i, wtips[i], freq[i], ks[i], costs[i], str(types[i]))
                print('%-2d: %-15g %-15g %-15g %-15g %-10s' % tup)

            for i in range(self.n_modes):
                microfem.plot_mode(fem, vall[:, i])
                