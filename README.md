# cantilever-optimizer

This code is designed for the topology optimization of cantilever structures
targeted for use in atomic force microscopy. The code is structured to account
for a range of different optimization problems and topology parameterizations.
The genetic algorithm is used to search for the optimal solution.

To use the software, first edit the `main()` function in `main_deap.py` (shown 
below). Set the `filename` variable to point to a YAML configuration file which 
describes the problem. Then execute the script. To re-examine previous 
solutions, comment out the line `opt.execute()` and uncomment the line 
`opt.load_solution()`.

```
def main():
    
    filename = 'examples/fast-new-compact.yaml'
    params = load_parameters(filename)
    opt = TopologyOptimizer(params)
    opt.execute()
    #opt.load_solution()
    return opt
```

At completion of the optimization routine, the following files are output for 
the above example.

* `fast-new-compact-design.npy` -- The parameters of the optimal solution in a
    binary format.
* `fast-new-compact-design.txt` -- The parameters of the optimal solution in a
    text format.
* `fast-new-compact-image.png` -- An image of the optimal topology.
* `fast-new-compact-records.txt` -- Lists the cost function at each iteration 
    of the optimization routine.

To re-execute the optimization the `.npy` output from the previous execution
of the script must be deleted. The YAML configuration file for an optimization 
problem is shown below.

```
---
generations: 60
num_individuals: 300
topology_class: new-compact
topology_params:
    support_ratio: 3.0
    nknx: 5
    nkny: 10
    nelx: 40
    nely: 80
    pcon1: 1.0e7
    pcon2: 1.0e3
    a0: 5.0
    b0: 5.0
problem_class: fast_cantilever
problem_params:
    k1: 10.0
...
```

The parameters are:
* `generations` -- The number of iterations of the genetic algorithm to 
    execute.
* `num_individuals` -- The number of solutions per iteration to examine in the 
    genetic algorithm.
* `topology_class` -- A keyword linked to a particular topology 
    parameterization.
* `topology_params` -- A dictionary of parameters assocaited with a given 
    topology.
* `problem_class` -- A keyword linked to the optimization problem.
* `probelm_params` -- A set of parameters associated with a given problem.

A set of examples are provided in the example folder for the three problems
and sixteen topolgies already provided with this code. To add additional 
problems and topologies create the class with the appropriate interface and 
modify the functions in `main_deap.py` called `init_problem()` and 
`init_topology()` to link the classes with a keyword.
* ``

# Problems

## API for Problem Classes

#### Public Attributes

`self.name` 

A string containing the name of the problem.

`self.ind_size`

The number of optimization parameters for the problem.

#### Functions

`objective_function(self, xs)`

`xs` is a rank 1 numpy array containing the parameters of optimizationnnnn

Constructor:


## fast_cantilever

The fast cantilever seeks to find the highest resonance frequency of the
first mode (if it is flexural) for a given stiffness constraint.

The parameters are:
* `k1` -- The stiffness constraint.

## bimodal

Bimodal cantilevers examine the dynamics of the first three modes and seek to
set the frequency and/or stiffness of the first two flexural modes. If there
is only one flexural mode the cost function is penalized. The parameters are
as follows.

* `stiffness_placement` -- Places the stiffness of mode 1 and mode 2 at given
    values.
* `frequency_placement` -- Place the frequency of mode 1 and mode 2 at given
    values.
* `frequency_minimization` -- Minimines the frequency of mode 2 for a given 
    frequency of mode 1.

## frequency_placement

This problem seeks to place the frequency of the first mode of the cantilever.

#### Parameters

* `f0` -- The frequency setpoint of the first mode.

# Topologies

## API for Topology Classes

## new-compact

Parameterizes the topology with a level set method. Unconnected structures
are possible.

The parameters are:
* `support_ratio` -- 3.0
* `nknx` -- The number of basis functions in the x-direction.
* `nkny` -- The number of basis functions in the y-direction.
* `nelx` -- Half the number of elements in the x-direction.
* `nely` -- The number of elements in the y-direction.
* `pcon1` -- 1.0e7
* `pcon2` -- 1.0e3
* `a0` -- half the width of an element in the x-direction (um).
* `b0` -- half the width of an element in the y-direction (um).

# Extensions

* Add option to seed genetic algorithm to produce consistent designs.
* Use zope.interface to enforce design by contract for topologies and problems.
