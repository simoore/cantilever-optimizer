import numpy as np
import microfem


class BimodalProblem(object):
    """
    Public Attributes
    -----------------
    self.ind_size
    self.name
    """

    def __init__(self, params, topology_factory):

        self.f1 = params['f1']
        self.f2 = params['f2']
        self.k1 = params['k1']
        self.k2 = params['k2']
        self.topology_factory = topology_factory
        self.material = microfem.SoiMumpsMaterial()
        self.n_modes = 3
        self.ind_size = self.topology_factory.ind_size
        self.name = '--- Bimodal Cantilever Optimization ---'

        funcs = {'stiffness_placement': self.stiffness_placement,
                 'frequency_minimization': self.frequency_minimization,
                 'stiffness_minimization': self.stiffness_minimization,
                 'stiffness_maximization': self.stiffness_maximization,
                 'frequency_maximization': self.frequency_maximization,
                 'frequency_placement': self.frequency_placement,
                 'k1_k2_place': self.k1_k2_place,
                 'k1_place_k2_min': self.k1_place_k2_min}

        if params['type'] not in funcs:
            raise ValueError('Bimodal type parameter invalid.')
        self.evaluation = funcs[params['type']]


    def objective_function(self, xs):

        self.topology_factory.update_topology(xs)

        if self.topology_factory.is_connected is True:

            params = self.topology_factory.get_params()
            cantilever = microfem.Cantilever(*params)
            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = microfem.PlateDisplacement(fem, coords).get_operator()
            mode_ident = microfem.ModeIdentification(fem, cantilever, 'plate')

            w, _, vall = fem.modal_analysis(self.n_modes)
            kuu = fem.get_stiffness_matrix(free=False)
            fs = np.sqrt(w) / (2*np.pi)
            phis = [vall[:, [i]] for i in range(self.n_modes)]
            wtips = [opr @ p for p in phis]
            kfunc = lambda p, w: np.asscalar(p.T @ kuu @ p / w ** 2)
            ks = [kfunc(p, w) for p, w in zip(phis, wtips)]
            types = [mode_ident.is_mode_flexural(p) for p in phis]

            if types[0] is False:
                cost = 1e8
            elif types[1] is True:
                cost = self.evaluation(fs[0], fs[1], ks[0], ks[1])
            elif types[2] is True:
                cost = self.evaluation(fs[0], fs[2], ks[0], ks[2])
            else:
                cost = 2e7

            return (cost,)

        return (self.topology_factory.connectivity_penalty,)


    def stiffness_placement(self, f1, f2, k1, k2):
        return abs(f1 - self.f1)/self.f1 + abs(k2 - self.k2)/self.k2


    def frequency_placement(self, f1, f2, k1, k2):
        return abs(f1 - self.f1)/self.f1 + abs(f2 - self.f2)/self.f2


    def frequency_minimization(self, f1, f2, k1, k2):
        return f2/self.f1 if f1 > self.f1 else 1e6-f1/self.f1


    def frequency_maximization(self, f1, f2, k1, k2):
        return -f2/self.f1 if f1 < self.f1 else f1/self.f1


    def stiffness_minimization(self, f1, f2, k1, k2):
        return k2 if f1 > self.f1 else 1e3*k2


    def stiffness_maximization(self, f1, f2, k1, k2):
        return -k2 if f1 < self.f1 and k2 < self.k2 else k2


    def k1_k2_place(self, f1, f2, k1, k2):
        return abs(k1 - self.k1)/self.k1 + abs(k2 - self.k2)/self.k2


    def k1_place_k2_min(self, f1, f2, k1, k2):
        return k2/self.k1 + abs(k1 - self.k1)/self.k1


    def console_output(self, xopt, image_file):


        self.topology_factory.update_topology(xopt)
        tup = (2*self.topology_factory.a, 2*self.topology_factory.b)
        print('The element dimensions are (um): %gx%g' % tup)
        params = self.topology_factory.get_params()
        cantilever = microfem.Cantilever(*params)
        microfem.plot_topology(cantilever, image_file)

        if self.topology_factory.is_connected is True:

            fem = microfem.PlateFEM(self.material, cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = microfem.PlateDisplacement(fem, coords).get_operator()
            mode_ident = microfem.ModeIdentification(fem, cantilever, 'plate')

            w, _, vall = fem.modal_analysis(self.n_modes)
            freq = np.sqrt(w) / (2*np.pi)
            kuu = fem.get_stiffness_matrix(free=False)
            phis = [vall[:, [i]] for i in range(self.n_modes)]
            wtips = [opr @ p for p in phis]
            kfunc = lambda p, w: np.asscalar(p.T @ kuu @ p / w ** 2)
            ks = [kfunc(p, w) for p, w in zip(phis, wtips)]
            types = [mode_ident.is_mode_flexural(p) for p in phis]

            tup = ('Disp', 'Freq (Hz)', 'Stiffness', 'Flexural')
            print('\n    %-15s %-15s %-15s %-10s' % tup)
            for i in range(self.n_modes):
                tup = (i, wtips[i], freq[i], ks[i], str(types[i]))
                print('%-2d: %-15g %-15g %-15g %-10s' % tup)

            for i in range(self.n_modes):
                microfem.plot_mode(fem, vall[:, i])
