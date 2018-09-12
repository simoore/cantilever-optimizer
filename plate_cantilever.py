import microfem

class PlateCantilever(microfem.Cantilever):
    
    def __init__(self, topology, a, b):
        
        name = 'Cantilever for stiffness optimization.'
        nelx, nely = topology.shape
        xtip = 1e6 * (a * nelx)
        ytip = 1e6 * (2 * nely * b - b)
        super().__init__(topology, a, b, xtip, ytip, topology, name)
        