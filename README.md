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
* `generations` - The number of iterations of the genetic algorithm to 
    execute.
* `num_individuals` - The number of solutions per iteration to examine in the 
    genetic algorithm.
* `topology_class` - A keyword linked to a particular topology 
    parameterization.
* `topology_params` - A dictionary of parameters assocaited with a given 
    topology.
* `problem_class` - A keyword linked to the optimization problem.
* `probelm_params` - A set of parameters associated with a given problem.

A set of examples are provided in the example folder for the three problems
and sixteen topolgies already provided with this code. To add additional 
problems and topologies create the class with the appropriate interface and 
modify the functions in `main_deap.py` called `init_problem()` and 
`init_topology()` to link the classes with a keyword.
* ``

# Problems

Problem classes define and compute the cost function used to evaluate a
solution. 


## Interface for Problem Classes

Use the following interface to added additional problems to the code.

#### Public Attributes

* `self.name` - A string containing the name of the problem.
* `self.ind_size` - The number of optimization parameters for the problem.

#### Functions

`objective_function(self, xs)` - This function is called to evaluate the cost
    function. `xs` is a rank 1 numpy array containing the optimization 
    parameters. Returns the evaluated cost function.
`console_output(self, xopt, image_file)` - This function prints the details
    of a solution to the console. `xopt` is the optimization parameters.
    `image_file` is a string containing a filename to save an image of the 
    topology. 

#### Constructor

`__init__(self, params, topology_factory)` - The constructor recieves the
    dictionary `params` which contains the `problem_params` from the YAML
    config file. `topology_factory` is the object that will generate the 
    parameters for the finite element model.


## Built-in Problem: fast_cantilever

The fast cantilever seeks to find the highest resonance frequency of the
first mode (if it is flexural) for a given stiffness constraint.

The parameters are:
* `k1` - The stiffness constraint.


## Built-in Problem: bimodal

Bimodal cantilevers examine the dynamics of the first three modes and seek to
set the frequency and/or stiffness of the first two flexural modes. If there
is only one flexural mode the cost function is penalized. The parameters are
as follows.

The parameters are:
* `f1` - Frequency of the first flexural mode.
* `f2` - Frequency of the second flexural mode.
* `k1` - Stiffness of the first flexural mode.
* `k2` - Stiffness of the second flexural mode.
* `type` - Selects from several different cost functions.

The type parameter can be one of the following values. Not all cost functions
use all the parameters above.
* `stiffness_placement` - Places the frequency of mode 1 and the stiffness of
    mode 2 at given values.
* `frequency_placement` - Place the frequency of mode 1 and mode 2 at given
    values.
* `frequency_minimization` - Minimines the frequency of mode 2 for a given 
    frequency of mode 1.
* `stiffness_minimization` - Minimizes the stiffness of mode 2 for a given
    frequency of mode 1.
* `stiffness_maximization` - Maximizes the stiffness of mode 2 for a given
    frequency of mode 1.
* `frequency_maximization` - Maximizes the frequency of mode 2 for a given
    frequency of mode 1.
* `k1_k2_place` - Places the stiffness of mode 1 and mode 2 at given
    values.
* `k1_place_k2_min` - Places the stiffness of mode 1 and minimizes the
    stiffness of mode 2.s
    
    
## Built-in Problem: frequency_placement

This problem seeks to place the frequency of the first mode of the cantilever.

The parameters are:
* `f0` -- The frequency setpoint of the first mode.


# Topologies

Topology classes take the a set of optimization parameters and produce the
parameters for the finite element model to analyze the structure.

## Interface for Topology Classes

#### Public Attributes

* `self.topology` - A binary matrix describing the mesh of the structure.
* `self.ind_size` - The number of parameters defining the topology.
* `self.a` - Half the element width (x-direction) in um.
* `self.b` - Half the element length (y-direction) in um.
* `self.xtip` - The mode shapes are normalized at this point (x-coord).
* `self.ytip` - The mode shapes are normalized at this point (y-coord).
* `self.is_connected` - True if the structure is valid.
* `self.connectivity_penalty` - A value to return as the cost function if
    unconnected.
    
#### Functions

* `update_topology(self, xs)` - Updates the public attributes based on the
    new optimization parameters `xs`.
* `get_params(self)` - Returns a tuple of parameters that can be passed 
    directly to the microfem Cantilever class.

#### Constructor

* `__init__(self, params)` - `params` is a dictionary containing the parameters
    from the YAML config file.


## Topology Classes

The following keywords in the YAML config file select different topology 
parameterizations.

* `gaussian`
* `bezier`
* `rectangle`
* `two_structure`
* `split`
* `stepped`
* `vshaped`
* `power`
* `new-rectangle`
* `new-split`
* `new-vshaped`
* `new-stepped`
* `new-power`
* `new-two-structures`
* `new-bezier`
* `new-compact`

All classes have the following parameters:
* `a0` - half the width of an element in the x-direction (um).
* `b0` - half the width of an element in the y-direction (um).
* `nelx` - Half the number of elements in the x-direction.
* `nely` - The number of elements in the y-direction.

The `new-compact` and `gaussian` classes have the following additional 
parameters:
* `support_ratio` -- 3.0
* `nknx` - The number of basis functions in the x-direction.
* `nkny` - The number of basis functions in the y-direction.
* `pcon1` - 1.0e7
* `pcon2` - 1.0e3

The `gaussian` class has the following addition parameter:
* `mask` - (None | mask1 | mask2) - Masks remove elements from the top corners
    of the design space.
    
The `bezier` and `new-bezier` classes have the following additional parameter:
* `ncurves` - The number of curves to parameterize the topology.

The `new-bezier` class has the following additional parameter:
* `crop` - If true elements whose y-coordinate are above the tip are cropped.


# Extensions

* Add option to seed genetic algorithm to produce consistent designs.
* Use zope.interface to enforce design by contract for topologies and problems.
