# cantilever-optimizer

This code is designed for the topology optimization of cantilever structures
targeted for use in atomic force microscopy. The code is structure to account
for a range of different optimization problems and topology parameterizations.
The genetic algorithm is used to search for the optimal solution.

To use the software, first edit the `main()` function in `main_deap.py`. Set 
the `filename` variable to point to a YAML configuration file which describes
the problem. Then execute the script. To re-examine previous solutions, 
comment out the line `opt.execute()` and uncomment the line 
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

To re-execute the optimization the `.npy` output from the previous execution
of the script must be deleted. For the above example the file is 
`examples/fast-new-compact-design.npy`. The YAML configuration file for a
optimization problem is shown below.

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
* `generations` -- The number of iterations of the genetic algorithm to execute.
* `num_individuals` -- The number of solutions per iteration to examine in the 
    genetic algorithm.
* `topology_class` -- A keyword linked to a particular topology parameterization.
* `topology_params` -- A dictionary of parameters assocaited with a given 
    topology.
* `problem_class` -- A keyword linked to the optimization problem.
* `probelm_params` -- A set of parameters associated with a given problem.

A set of examples are provided in the example folder for the three problems
and eight topolgies already provided with this code. To add additional problems
and topologies create the class with the appropriate interface and modify
the functions in `main_deap.py` called `init_problem()` and `init_topology()`
to link the classes with a keyword.

After the problem has executed, the following files are generated
* ``

# Problems

Problems prefixed by `new-` have a fixed mesh size but the cantilever tip
can change locations. Otherwise, the tip location is fixed and the mesh size
changes.


## API for Problem Classes

Public Attributes:
* `self.name` -- A string containing the name of the problem.
* `self.ind_size` -- The number of optimization parameters for the problem.

Functions:
* 

Constructor:


## fast_cantilever

The fast cantilever seeks to find the highest resonance frequency of the
first mode (if it is flexural) for a given stiffness constraint.

The parameters are:
* `k1` -- The stiffness constraint.

## The Bimodal Cantilever


## The Frequency


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
