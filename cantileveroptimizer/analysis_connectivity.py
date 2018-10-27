import numpy as np
from collections import deque


def is_connected(topology):
    """
    Determines if a topology is a connected structure.
    
    Parameters
    ----------
    topology : 2d binary ndarray 
        A binary matrix representing the topology. A value of 1 is a solid 
        element, 0 is a void element.
        
    Returns
    -------
    : bool          
        True if the structure is connected else false.
    """
    
    # The fixed end of the structure is along the x-axis (y=0), which is the 
    # first column of the binary matrix represeneting the topology.
    width, _ = topology.shape
    source = None
    for ii in range(width):
      if topology[ii, 0] == 1 and source is None:
          source = (ii, 0)
    
    # Structure is not connected to the fixed boundary.
    if source is None:
        return False
    
    # The structure is connected if the number of discovered nodes equals the
    # toal number of nodes in the topology.
    n_nodes = np.sum(topology)
    discovered = depth_first_search(topology, source)
    n_connected = np.sum(discovered)
    if n_nodes == n_connected:
        return True
    return False
    

def connectivity_penalization(topology):
    """
    The article:
    
    Multi-objective topology optimization of multi-component continuum 
    structures via a Kriging-interpolated level set approach: 
    David Guirguis, Karim Hamza, Mohamed Aly, Hesham Hegazi, Kazuhiro Saitou:
    2015, Volume 51, Issue 3, pp 733–748,
    
    presents a penalization function for applying an unconstrained optimization
    to the constrainted optimization problem. This function calculates the two 
    metrics required, the number of floating islands in the structure, and the
    distance of each island away from the connected structure.
    
    Parameters
    ----------
    topology : 2d binary ndarray
        The matrix describing the topology of the cantilever structure. The 
        fixed edge is along the left-hand column. A solid element is denoted 
        with 1 and a void element is denoted with 0.
    
    Returns
    -------
    n_islands : int
        The number of disconnected regions in the topology.
    distance_metric : int
        From a base element, this is the length of shortest path to all
        regions in the topology.
    """
    
    width, _ = topology.shape
    source = None
    for ii in range(width):
      if topology[ii, 0] == 1 and source is None:
          source = (ii, 0)
       
    if source is not None:
        n_islands = connectivity_regions(topology) - 1
        distances = dijkstra(topology, source)
        distance_metric = np.sum(np.unique(topology * distances))
    else:
        n_islands = connectivity_regions(topology)
        distances = dijkstra(topology, (0, 0))
        distance_metric = np.sum(np.unique(topology * distances))

    return n_islands, distance_metric


def dijkstra(graph, source):
    """
    This function applies Dijkstra's algorithm to the graph to find the 
    shortest path to all nodes. The graph in this code is a binary 2D image.
    Each pixel is a node. The edge between two black (1) pixels is 0, else it
    is 1 from or to any white (0) pixels.
    
    Parameters
    ----------
    graph : 2d binary ndarray
        The binary image that represents that is intepreted as a graph.
    source : tuple
        (i, j) is the coordinate of the source pixel.
        
    Returns
    -------
    distances : 2d ndarray
        A matrix of distances from the source node.
    """
    width, length = graph.shape
    n_visited = 0
    n_nodes = np.prod(graph.shape)
    visited = np.zeros(graph.shape)
    distances = np.full(graph.shape, np.inf)
    candidates = []
    
    current = source
    current_distance = 0
    distances[current] = current_distance
    
    f = lambda i, j: [(i+1, j), (i-1,j), (i,j+1), (i, j-1)]
    g = lambda i, j: (0 <= i < width and 0 <= j < length)
    h = lambda i, j: [(ii, jj) for (ii, jj) in f(i, j) if g(ii, jj)]
    
    while n_visited < n_nodes:
        
        for neighbour in h(*current):
            if visited[neighbour] == 0:
                distance = 1 if graph[neighbour] == 0 or graph[current] == 0 else 0
                new_distance = current_distance + distance
                if distances[neighbour] == float('inf'):
                    candidates.append(neighbour)
                if distances[neighbour] > new_distance:
                    distances[neighbour] = new_distance
            
        n_visited += 1
        visited[current] = 1
        distances[current] = current_distance
        
        if n_visited < n_nodes:
            values = [(c, distances[c]) for c in candidates]
            index, (current, current_distance) = min(enumerate(values), key=lambda x: x[1][1])
            del candidates[index]
            
    return distances
            
            
def connectivity_regions(topology):
    """
    Detects the number of disconnected regions in the topology.
    
    Parameters
    ----------
    topology : 2d binary ndarray
        A 2d binary matrix describing which elements are solid and void.
        
    Returns
    -------
    nr : int
        Number of regions in the topology.
    """
    
    label = 1
    image = np.zeros(topology.shape)
    for i, j in np.ndindex(topology.shape):
        if topology[i, j] == 1 and image[i, j] == 0:
            region = depth_first_search(topology, (i, j))
            image = image + label * region
            label += 1
    nr = label - 1
    return nr
    
    
def connectivity_hinges(topology):
    """
    Hinges are topological features consisting of a single element appended
    on the edge of the structure. Sometimes this type of feature is 
    undersirable. This function detects the number of hinges in the topology
    such that the topology can be penalized in an optimation problem.
    
    Parameters
    ----------
    topology : 2D binary ndarray
        This 2D binary matrix describes the topology of the cantilever.
    
    Returns
    -------
    int
        The number of hinges in the topology.
    """
    wd, ln = topology.shape
    f = lambda x, y: topology[x, y] if 0 <= x < wd and 0 <= y < ln else 0
    g = lambda x, y: f(x + 1, y) + f(x - 1, y) + f(x, y + 1) + f(x, y - 1)
    gmat = np.array([[g(i, j) for j in range(ln)] for i in range(wd)])
    nh = np.sum(np.logical_and(topology, gmat == 1))
    return nh


def depth_first_search(graph, node):
    """
    Starting with a given node, the graph is serached for all connected nodes.
    It is depth first because the nodes that are most recently discovered are
    then analyzed to find more connected nodes.
    
    Parameters
    ----------
    graph : 2d binary ndarray
        Nodes in the graph are represented by 1. Four edge connectivity is 
        used to describe the edges in the graph.
    node : tuple 
        The node (i,j) from which to start the search.
        
    Returns
    -------
    discovered : 2d binary ndarray
        A binary matrix where discovered nodes are given a value of 1.
    """
    
    discovered = np.zeros(graph.shape)
    stack = deque()
    stack.append(node)
    
    # The function h(i, j) returns the nodes adjacent to node (i, j) in a list.
    width, length = graph.shape
    f = lambda i, j: [(i+1, j), (i-1,j), (i,j+1), (i, j-1)]
    g = lambda i, j: (0 <= i < width and 0 <= j < length)
    h = lambda i, j: [(ii, jj) for (ii, jj) in f(i, j) if g(ii, jj)]
    
    while stack:
        v = stack.pop()
        if discovered[v] == 0:
            discovered[v] = 1
            for n in h(*v):
                if graph[n] == 1:
                    stack.append(n)
    return discovered
    