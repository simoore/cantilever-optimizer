import sys

civ_path = 'D:\\doc\\Google Drive\\Github\\micro-fem'
if civ_path not in sys.path:
    sys.path.append(civ_path)
    
import numpy as np
import microfem
#from topology_regular_stepped import RegularSteppedTopology
#from topology_bezier_v2 import BezierTopology
#from topology_new_bezier import NewBezierTopology
#from topology_v_shaped import RegularVShaped
#from topology_power import PowerTopology
#from topology_rectangle_new import NewRectangleTopology
#from topology_split_new import NewSplitTopology
#from topology_new_v_shaped import NewVShaped
#from topology_new_stepped import NewSteppedTopology
#from topology_new_power import NewPowerTopology
#from topology_new_two_structures import NewTwoStructuresTopology
from topology_new_compact import NewCompactTopology


#params = {'nelx': 30, 'nely': 60, 'a0': 10e-6, 'b0':10e-6}
#top = RegularSteppedTopology(params)

#params = {'nelx': 30, 'nely': 60, 'a0': 10e-6, 'b0':10e-6, 'ncurves': 3}
#top = BezierTopology(params)
#top = NewBezierTopology(params)

#params = {'nelx': 30, 'nely': 60, 'a0': 10e-6, 'b0':10e-6}
#top = RegularVShaped(params)

#params = {'nelx': 30, 'nely': 60, 'a0': 10e-6, 'b0':10e-6}
#top = PowerTopology(params)

#params = {'nelx': 30, 'nely': 60, 'a0': 10e-6, 'b0':10e-6}
#top = PowerTopology(params)

#params = {'nelx': 30, 'nely': 60, 'a0': 5e-6, 'b0':5e-6}
#top = NewRectangleTopology(params)
#top = NewSplitTopology(params)
#top = NewSteppedTopology(params)
#top = NewPowerTopology(params)
#top = NewTwoStructuresTopology(params)

params = {'support_ratio': 3.0,
          'nknx': 5,
          'nkny': 10,
          'nelx': 40,
          'nely': 80,
          'pcon1': 1.0e7,
          'pcon2': 1.0e3,
          'a0': 5.0e-6,
          'b0': 5.0e-6}
top = NewCompactTopology(params)


for i in range(1):
    xs = 2*np.random.random(top.ind_size) - 1.0
    top.update_topology(xs)
    cantilever = microfem.Cantilever(*top.get_params())
    #print(xs)
    print(top.xtip, top.ytip)
    microfem.plot_topology(cantilever)

