"""
Version 2: incorporates a new way of templating optimization problems.
Version 3: rework how multiobjective optimization is incorporated.
"""
import random
import numpy as np
import array
from time import time
from deap import creator, base, tools, algorithms
from ruamel.yaml import YAML
import pprint
import os
import matplotlib.pyplot as plt
from topology_bezier_v2 import BezierTopology
from topology_compact_v2 import CompactTopology
from topology_rectangle import RectangleTopology
from topology_regular_two_structures import RegularTwoStructureTopology
from topology_regular_split import RegularSplitTopology
from topology_regular_stepped import RegularSteppedTopology
from topology_v_shaped import RegularVShaped
from topology_power import PowerTopology
from topology_new_rectangle import NewRectangleTopology
from topology_new_split import NewSplitTopology
from topology_new_v_shaped import NewVShaped
from topology_new_stepped import NewSteppedTopology
from topology_new_power import NewPowerTopology
from topology_new_two_structures import NewTwoStructuresTopology
from topology_new_bezier import NewBezierTopology
from topology_new_compact import NewCompactTopology
from problem_frequency import FrequencyProblem
from problem_bimodal import BimodalProblem
from problem_stiffness_ratio import StiffnessRatioProblem
from problem_off_resonance import OffResonanceProblem
from problem_fast_cantilever import FastCantileverProblem


def main():
    
    #filename = 'solutions-bimodal/regular-3f.yaml
    #filename = 'solutions-bimodal/prelim-1f.yaml'
    #filename = 'solutions-stiffness-ratio/prelim-4h.yaml'
    #filename = 'solutions-off-resonance/prelim-1a.yaml'
    filename = 'solutions-fast-cantilevers/new-compact-1a.yaml'
    params = load_parameters(filename)
    opt = TopologyOptimizer(params)
    opt.execute()
    #opt.load_solution()
    #opt.load_front()
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
        
        random.seed()
        self.tag = params['tag']
        self.dir = params['dir']
        self.exe_time = 0
        self.init_topology(params)
        self.init_problem(params)
        if params['algorithm_class'] == 'single':
            self.save_pareto_front = False
            self.init_algorithm(params)
        elif params['algorithm_class'] == 'multiobjective':
            self.save_pareto_front = True 
            self.init_multiobjective(params)
        else:
            raise ValueError('Non-existent algorithm class.')
    
    
    ###########################################################################
    # Initialize components of the optimizer.
    ###########################################################################       
    def init_algorithm(self, params):
                
        toolbox = base.Toolbox()
        ind_size = self.problem.ind_size
        ngen = params['generations']
        nind = params['num_individuals']
        cxpb, mutpb, lb, ub, mu, lam = 0.7, 0.2, -1.0, 1.0, nind, nind
                
        if hasattr(creator, 'FitnessMin') is False: 
            creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
            
        if hasattr(creator, 'Individual') is False:
            kw0 = {'typecode': 'd', 'fitness': creator.FitnessMin}
            creator.create('Individual', array.array, **kw0)
            
        atr = lambda: [random.uniform(lb, ub) for _ in range(ind_size)]
        ind = lambda: tools.initIterate(creator.Individual, atr)
        population = [ind() for _ in range(nind)]
        
        kw1 = {'low': lb, 'up': ub, 'eta': 20.0, 'indpb': 1.0/ind_size}
        mut = lambda xs: tools.mutPolynomialBounded(xs, **kw1)
        kw2 = {'low': lb, 'up': ub, 'eta': 20.0}
        crs = lambda i1, i2: tools.cxSimulatedBinaryBounded(i1, i2, **kw2)
        sel = lambda p, n: tools.selTournament(p, n, tournsize=3)
        
        toolbox.register('evaluate', self.problem.objective_function)
        toolbox.register('mate', crs)
        toolbox.register('mutate', mut)
        toolbox.register('select', sel)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min)
        self.hof = tools.HallOfFame(5)
        
        args = (population, toolbox, mu, lam, cxpb, mutpb, ngen)
        kw3 = {'stats': stats, 'halloffame': self.hof, 'verbose': True}
        self.algorithm = lambda: algorithms.eaMuPlusLambda(*args, **kw3)
        
        
    def init_multiobjective(self, params):
                
        toolbox = base.Toolbox()
        ind_size = self.problem.ind_size
        ngen = params['generations']
        nind = params['num_individuals']
        cxpb, mutpb, lb, ub, mu, lam = 0.7, 0.2, -1.0, 1.0, nind, 2*nind
                
        if hasattr(creator, 'FitnessMin') is False: 
            creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
            
        if hasattr(creator, 'Individual') is False:
            kwargs = {'typecode': 'd', 'fitness': creator.FitnessMin}
            creator.create('Individual', array.array, **kwargs)
            
        atr = lambda: [random.uniform(lb, ub) for _ in range(ind_size)]
        ind = lambda: tools.initIterate(creator.Individual, atr)
        population = [ind() for _ in range(nind)]
        
        kwargs = {'low': lb, 'up': ub, 'eta': 20.0, 'indpb': 1.0/ind_size}
        mut = lambda xs: tools.mutPolynomialBounded(xs, **kwargs)
        kwargs = {'low': lb, 'up': ub, 'eta': 20.0}
        crs = lambda a, b: tools.cxSimulatedBinaryBounded(a, b, **kwargs)
        sel = tools.selNSGA2
        
        toolbox.register('evaluate', self.problem.objective_function)
        toolbox.register('mate', crs)
        toolbox.register('mutate', mut)
        toolbox.register('select', sel)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min', np.min, axis=0)
        self.hof = tools.ParetoFront()
        
        args = (population, toolbox, mu, lam, cxpb, mutpb, ngen)
        kwargs = {'stats': stats, 'halloffame': self.hof, 'verbose': True}
        self.algorithm = lambda: algorithms.eaMuPlusLambda(*args, **kwargs)


    def init_problem(self, params):
          
        problem_class = params['problem_class']
        problem_params = (params['problem_params'], self.topology_factory)
        
        funcs = {'frequency_placement': FrequencyProblem,
                 'bimodal': BimodalProblem,
                 'stiffness_ratio': StiffnessRatioProblem,
                 'off_resonance': OffResonanceProblem,
                 'fast_cantilever': FastCantileverProblem}
        
        if problem_class not in funcs:
            raise ValueError('Non-existent problem class.')
        self.problem = funcs[problem_class](*problem_params)
                 
    
    def init_topology(self, params):
        
        funcs = {'gaussian': CompactTopology,
                 'bezier': BezierTopology,
                 'rectangle': RectangleTopology,
                 'two_structure': RegularTwoStructureTopology,
                 'split': RegularSplitTopology,
                 'stepped': RegularSteppedTopology,
                 'vshaped': RegularVShaped,
                 'power': PowerTopology,
                 'new-rectangle': NewRectangleTopology,
                 'new-split': NewSplitTopology,
                 'new-vshaped': NewVShaped,
                 'new-stepped': NewSteppedTopology,
                 'new-power': NewPowerTopology,
                 'new-two-structures': NewTwoStructuresTopology,
                 'new-bezier': NewBezierTopology,
                 'new-compact': NewCompactTopology}
        
        topology_class = params['topology_class']
        topology_params = params['topology_params']
        
        if topology_class not in funcs:
            raise ValueError('Non-existent topology class.')
        self.topology_factory = funcs[topology_class](topology_params)

        
    ###########################################################################
    # Console logging functions.
    ###########################################################################
    def to_console_init(self):
        
        print()
        print(self.problem.name)
        print('Number of Parameters: %d' % self.problem.ind_size)

        
    def to_console_final(self, xopt):
        
        print()
        print('--- Solution Characteristics ---')
        if self.exe_time != 0:
            print('Time (s): %g' % (self.exe_time))
        fn = ''.join((self.dir, '/', self.tag, '-image.png'))
        self.problem.console_output(xopt, fn)
        

    ###########################################################################
    # Execution and analysis.
    ###########################################################################
    def execute(self):
        
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        if os.path.isfile(fn):
            print('Solution already exists. Delete .npy file to continue.')
            return
        
        self.to_console_init()      
        tic = time()
        self.pop, self.log = self.algorithm()
        toc = time()
        self.exe_time = toc - tic
        self.save_solution_all()
        self.save_records()
        if self.save_pareto_front is True:
            self.save_front()
        self.to_console_final(self.hof[0])
        
        
    ###########################################################################
    # Load/save data.
    ###########################################################################
    def save_solution_all(self):
        
        if len(self.hof) == 1:
            data = self.hof[0]
        else:
            data = np.vstack(self.hof).T
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        np.save(fn, data)    
        fn = ''.join((self.dir, '/', self.tag, '-design.txt'))
        np.savetxt(fn, data)
        
        
    def load_solution(self, index=0):
        
        fn = ''.join((self.dir, '/', self.tag, '-design.npy'))
        data = np.load(fn)
        if len(data.shape) == 1:
            self.to_console_final(data)
        else:
            self.to_console_final(data[:, index])
   
    
    def save_front(self):     
        
        front = np.array([ind.fitness.values for ind in self.hof])
        fn = ''.join((self.dir, '/', self.tag, '-front.npy'))
        np.save(fn, front)
        
        
    def load_front(self):
        
        fn = ''.join((self.dir, '/', self.tag, '-front.npy'))
        front = np.load(fn)
        plt.scatter(front[:,0], front[:,1], c="b")
        
        
    def save_records(self):
        
        header = 'Iteration, Objective Value(s)'
        fn = ''.join((self.dir, '/', self.tag, '-records.txt'))
        mat = [np.hstack((r['gen'], r['min'])) for r in self.log]
        data = np.array(mat)
        np.savetxt(fn, data, delimiter=',', header=header)
        
        
if __name__ == '__main__':
    
    np.set_printoptions(precision=3)
    opt = main()
