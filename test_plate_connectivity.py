import unittest
import numpy as np
from plate_analysis import is_connected
from plate_analysis import connectivity_hinges
from plate_analysis import dijkstra
from plate_analysis import connectivity_penalization


class TestConnectivityChecker(unittest.TestCase):
    
    t2 = np.array([[0, 0, 0, 1, 1, 0],
                   [1, 1, 1, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [1, 1, 1, 1, 1, 1],
                   [0, 0, 0, 0, 1, 0]])
    
    def test_all(self):

        t1 = np.array([[0, 0, 0, 1, 1, 0],
                       [1, 1, 1, 1, 0, 0],
                       [0, 0, 1, 0, 1, 0],
                       [1, 1, 1, 1, 1, 1],
                       [0, 0, 0, 0, 1, 0]])
    
        self.assertTrue(is_connected(t1) == True)
        self.assertTrue(is_connected(self.t2) == False)
        
        nh = connectivity_hinges(t1)
        
        self.assertTrue(nh == 6)
        
        
    def test_dijkstra(self):
        
        
        e2 = np.array([[0, 1, 2, 1, 1, 2],
                       [1, 1, 1, 1, 2, 3],
                       [2, 2, 2, 2, 3, 4],
                       [3, 3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 3, 4]])
    
        distances = dijkstra(self.t2, (0, 0))
        self.assertTrue(np.all(distances == e2))
        
        ni, dm = connectivity_penalization(self.t2)
        self.assertTrue(ni == 1)
        self.assertTrue(dm == 2)
        
if __name__ == '__main__':
    unittest.main()
