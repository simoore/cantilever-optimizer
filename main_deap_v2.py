"""
Version 2: incorporates a new way of templating optimization problems.
"""
import random
import numpy as np
import array
from time import time
from deap import creator, base, tools, algorithms
from ruamel.yaml import YAML
import pprint
import os
from topology_radial_level_set import RadialLevelSetTopology
from topology_bezier import BezierTopology
from topology_compact_v2 import CompactTopology
from problem_frequency import FrequencyProblem
from problem_bimodal import BimodalProblem


def main():
    
    filename = 'solutions-bimodal/test.yaml'
    params = load_parameters(filename)
    opt = TopologyOptimizer(params)
    opt.execute()
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
        
        random.seed()
        self.tag = params['tag']
        self.dir = params['dir']
        self.exe_time = 0
        self.init_topology(params)
        self.init_problem(params)
        self.init_algorithm(params)
    
    
    ###########################################################################
    # Initialize components of the optimizer.
    ###########################################################################       
    def init_algorithm(self, params):
        
        if params['algorithm_class'] not in ['simple', 'multiobjective']:
            raise ValueError('Non-existent algorithm class.')
        
        toolbox = base.Toolbox()
        ngen = params['generations']
        nind = params['num_individuals']
        cxpb = 0.5 if params['algorithm_class'] == 'simple' else 0.9
        lb, ub = -1.0, 1.0
        ind_size = self.problem.ind_size
        
        if nind % 4 != 0:
            raise ValueError('Number of individuals must be multiple of four')
        
        if hasattr(creator, 'FitnessMin') is False: 
            creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
            
        if hasattr(creator, 'Individual') is False:
            creator.create('Individual', array.array, typecode='d', fitness=creator.FitnessMin)
            
        atr = lambda: [random.uniform(lb, ub) for _ in range(ind_size)]
        ind = lambda: tools.initIterate(creator.Individual, atr)
        population = [ind() for _ in range(nind)]
        
        if params['algorithm_class'] == 'simple':
            self.hof = tools.HallOfFame(1)
            mut = lambda xs: tools.mutGaussian(xs, mu=0, sigma=1, indpb=0.1)
            crs = tools.cxTwoPoint
            sel = lambda p, n: tools.selTournament(p, n, tournsize=3)
        else:
            self.hof = tools.ParetoFront()
            mut = lambda xs: tools.mutPolynomialBounded(xs, low=lb, up=ub, eta=20.0, indpb=1.0/ind_size)
            crs = lambda ind1, ind2: tools.cxSimulatedBinaryBounded(ind1, ind2, low=lb, up=ub, eta=20.0)
            sel = tools.selNSGA2
        
        toolbox.register('evaluate', self.problem.objective_function)
        toolbox.register('mate', crs)
        toolbox.register('mutate', mut)
        toolbox.register('select', sel)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean, axis=0)
        stats.register('std', np.std, axis=0)
        stats.register('min', np.min, axis=0)
        stats.register('max', np.max, axis=0)
        
        args = (population, toolbox)
        kwargs = {'cxpb': cxpb, 'ngen': ngen, 'stats': stats} 
        kwargs['halloffame'] = self.hof
        if params['algorithm_class'] == 'simple':
            kwargs['mutpb'] = 0.2 
            kwargs['verbose'] = True 
            self.algorithm = lambda: algorithms.eaSimple(*args, **kwargs)
        else:
            self.algorithm = lambda: self.multiobjective(*args, **kwargs) 

    
    def init_problem(self, params):
          
        problem_class = params['problem_class']
        problem_params = (params['problem_params'], self.topology_factory)
        
        if problem_class == 'frequency_placement':
            self.problem = FrequencyProblem(*problem_params)
        elif problem_class == 'bimodal':
            self.problem = BimodalProblem(*problem_params)
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
        self.save_solution()
        self.save_records()
        self.to_console_final(self.hof[0])
        
        
    @staticmethod
    def multiobjective(pop, toolbox, ngen, cxpb, stats, halloffame):
        """
        This code is based of the NSGA-II example from:
        https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
        """
        logbook = tools.Logbook()
        logbook.header = 'gen', 'evals', 'std', 'min', 'avg', 'max'
        
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        
        halloffame.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)

        # Begin the generational process
        for gen in range(1, ngen):
            
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= cxpb:
                    toolbox.mate(ind1, ind2)
                
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
    
            # Select the next generation population
            pop = toolbox.select(pop + offspring, len(pop))
            halloffame.update(pop)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
    
        return pop, logbook
        
        
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
        
        header = 'Iteration, Objective Value(s)'
        fn = ''.join((self.dir, '/', self.tag, '-records.txt'))
        mat = [np.hstack((r['gen'], r['min'])) for r in self.log]
        data = np.array(mat)
        np.savetxt(fn, data, delimiter=',', header=header)
        
        
if __name__ == '__main__':
    
    np.set_printoptions(precision=3)
    opt = main()
