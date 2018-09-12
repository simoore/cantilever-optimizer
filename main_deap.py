import sys

civ_path = 'D:\\doc\\Google Drive\\Github\\micro-fem'
if civ_path not in sys.path:
    sys.path.append(civ_path)

import random
import numpy as np
from time import time
from deap import creator, base, tools, algorithms
import microfem
from ruamel.yaml import YAML
import pprint
import os

from analysis_plate_displacement import PlateDisplacement
from analysis_mode_identification import ModeIdentification
from topology_radial_level_set import RadialLevelSetTopology
from topology_bezier import BezierTopology
from topology_compact_v2 import CompactTopology
from problem_frequency import FrequencyProblem


def main():
    
    filename = 'solutions-bimodal/prelim-1a.yaml'
    params = load_parameters(filename)
    opt = TopologyOptimizer(params)
    #opt.execute()
    #opt.load_solution()
    return opt

        
def load_parameters(filename):
    
    with open(filename, 'r') as f:
        yaml = YAML(typ='safe')
        params = yaml.load(f)
        params['tag'] = os.path.splitext(os.path.basename(f.name))[0]
        params['dir'] = os.path.dirname(f.name)
        
        print('--- Parameters ---')
        pprint.pprint(params)
        return params


class TopologyOptimizer(object):
    
    def __init__(self, params):
        
        #self.nknx = params['nknx']
        #self.nkny = params['nkny']
        
        
        #self.top_method = params['top_method']
        self.a = 5e-6
        self.b = 5e-6
        
        
        #self.num_curves = params['ncurves']
        #self.n_modes = params['nmodes']
        #self.support_ratio = params['support_ratio']
        #self.n_ratio = params['nratio']
        
        self.tag = params['tag']
        self.dir = params['dir']
        self.ngen = params['generations']
        self.nind = params['num_individuals']
        self.cxpb = 0.5
        self.mutpb = 0.2
        #self.pcon1 = 1e6
        #self.pcon2 = 10
        
        
        #material = microfem.SoiMumpsMaterial()
        #self.topology_factory = self.init_topology_factory()
        #self.fem = microfem.PlateFEM(material, self.a, self.b)
        self.exe_time = 0
        self.init_topology(params)
        self.init_problem(params)
        #self.init_simple_genetic_algorithm()
        
    
    def init_problem(self, params):
          
        problem_class = params['problem_class']
        problem_params = params['problem_params']
        
        if problem_class == 'frequency_placement':
            self.problem = FrequencyProblem(problem_params)
        else:
            raise ValueError('Non-existent problem class.')
    
    
    def init_topology(self, params):
        
        topology_class = params['topology_class']
        topology_params = params['topology_params']
        
        if topology_class == 'mq_spline':
            self.topology_factory = RadialLevelSetTopology(topology_params)
        elif topology_class == 'gaussian':
            self.topology_factory = CompactTopology(topology_params)
        elif topology_class == 'bezier':
            self.topology_factory = BezierTopology(topology_params)
        else:
            raise ValueError('Non-existent topology class.')

        
       
    ###########################################################################
    # Console logging functions.
    ###########################################################################
    def to_console_init(self):
        
        print()
        print('--- AFM Cantilever Optimization ---')

        
    def to_console_final(self, xopt):
        
        print()
        print('--- Solution Characteristics ---')
        if self.exe_time != 0:
            print('Time (s): %g' % (self.exe_time))
        
        self.topology_factory.update_topology(xopt)
        topology = self.topology_factory.topology
        cantilever = microfem.Cantilever(topology, self.a, self.b)
        fn = ''.join((self.dir, '/', self.tag, '-image.png'))
        microfem.plot_topology(cantilever, fn)
        
        if self.topology_factory.is_connected is True:
            
            self.fem.update_mesh(cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            opr = PlateDisplacement(self.fem, coords).get_operator()
            mode_ident = ModeIdentification(self.fem, cantilever)
            
            w, _, vall = self.fem.modal_analysis(self.n_modes)
            freq = np.sqrt(w) / (2*np.pi)
            kuu = self.fem.get_stiffness_matrix(free=False)
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
                microfem.plot_mode(self.fem, vall[:, i])

        
    ###########################################################################
    # Execution and analysis.
    ###########################################################################
    def execute(self):
        
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        if os.path.isfile(fn):
            print('Solution already exists. Delete .npy file to continue.')
            return
        
        self.to_console_init()
        args = (self.population, self.toolbox)
        kwargs = {'cxpb': self.cxpb, 
                  'mutpb': self.mutpb, 
                  'ngen': self.ngen, 
                  'stats': self.stats, 
                  'halloffame': self.hof, 
                  'verbose': True}        
        tic = time()
        _, self.log = algorithms.eaSimple(*args, **kwargs) 
        toc = time()
        self.exe_time = toc - tic
        self.save_solution()
        self.save_records()
        self.to_console_final(self.hof[0])
        
        
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
        
        self.topology_factory.update_topology(np.array(xs))
        topology = self.topology_factory.topology
        
        if self.topology_factory.is_connected is True:
            
            cost = self.problem.objective_function(topology)
            
            # Initialize new FEM and other analyses.
            cantilever = microfem.Cantilever(topology, self.a, self.b)
            self.fem.update_mesh(cantilever)
            coords = (cantilever.xtip, cantilever.ytip)
            pd = PlateDisplacement(self.fem, coords)
            opr = pd.get_operator()
            mode_ident = ModeIdentification(self.fem, cantilever)
            
            # Perform modal analysis and retrieve the stiffness matrix.
            w, _, vall = self.fem.modal_analysis(self.n_modes)
            kuu = self.fem.get_stiffness_matrix(free=False)
            
            # Analyze the first mode.
            phi1 = vall[:, [0]]
            wtip1 = opr @ phi1
            k1 = np.asscalar(phi1.T @ kuu @ phi1 / wtip1 ** 2)
            
            cost, ratio_index, ratios = 0, 0, [2, 3, 4]
            for i in range(1, self.n_modes):
                if ratio_index < (self.n_ratio-1):
                    phi = vall[:, [i]]
                    if mode_ident.is_mode_flexural(phi) == True:
                        wtip = opr @ phi
                        k = np.asscalar(phi.T @ kuu @ phi / wtip ** 2)
                        cost += (k/k1 - ratios[ratio_index]) ** 2
                        ratio_index += 1
                        
            return (cost,)
        
        return (self.topology_factory.connectivity_penalty,)
    
    
    ###########################################################################
    # Load/save data.
    ###########################################################################
    def save_solution(self):
        
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        np.save(fn, self.hof[0])    
        fn = ''.join((self.dir, '/', self.tag, '-design.txt'))
        np.savetxt(fn, self.hof[0])
        
        
    def load_solution(self):
        
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        self.to_console_final(np.load(fn))
    
    
    def save_records(self):
        
        header = 'Iteration, Objective Value'
        fn = ''.join((self.dir, '/', self.tag, '-records.txt'))
        mat = [(r['gen'], r['min']) for r in self.log]
        data = np.array(mat)
        np.savetxt(fn, data, delimiter=',', header=header)
        
        
    ###########################################################################
    # Initialize components of the optimizer.
    ###########################################################################
#    def init_topology_factory(self):
#        
#        dim_knots = (self.nknx, self.nkny)
#        dim_elems = (self.nelx, self.nely)
#        c_penal = (self.pcon1, self.pcon2)
#        
#        if self.top_method == 'mq_spline':
#            tparams = (dim_knots, dim_elems, self.a, self.b, True)
#            top_factory = RadialLevelSetTopology(*tparams)
#        elif self.top_method == 'gaussian':
#            tparams = (dim_knots, dim_elems, self.a, self.b, c_penal, 
#                       self.support_ratio)
#            top_factory = CompactTopology(*tparams)
#        elif self.top_method == 'bezier':
#            tparams = (self.num_curves, self.nelx, self.nely)
#            top_factory = BezierTopology(*tparams)
#        else:
#            top_factory = None
#            
#        return top_factory
    
    
    def init_simple_genetic_algorithm(self):
        
        ind_size = self.topology_factory.ind_size
        self.toolbox = base.Toolbox()
        
        if hasattr(creator, 'FitnessMin') is False:
            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
            
        if hasattr(creator, 'Individual') is False:
            creator.create('Individual', list, fitness=creator.FitnessMin)
        
        ind_size = self.topology_factory.ind_size
        att = lambda : random.uniform(-1.0, 1.0)
        ind = lambda : tools.initRepeat(creator.Individual, att, n=ind_size)
        pop = lambda n: tools.initRepeat(list, ind, n=n)
        mut = lambda xs: tools.mutGaussian(xs, mu=0, sigma=1, indpb=0.1)
        
        self.toolbox.register('evaluate', self.objective_function)
        self.toolbox.register('mate', tools.cxTwoPoint)
        self.toolbox.register('mutate', mut)
        self.toolbox.register('select', tools.selTournament, tournsize=3)
        
        self.population = pop(n=self.nind)
        self.hof = tools.HallOfFame(1)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register('min', np.min)
        
    
if __name__ == '__main__':
    
    np.set_printoptions(precision=3)
    opt = main()
