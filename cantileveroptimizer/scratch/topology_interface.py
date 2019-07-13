class TopologyMeta(type):
         
    def __call__(cls, *args, **kwargs):
        
        obj = type.__call__(cls, *args, **kwargs)
        obj.check_attributes()
        return obj
    
    
class Topology(object, metaclass=TopologyMeta):
    """
    This class checks that all requried functions and attributes are defined 
    in any subclass.
    """
    attributes = ('a', 
                  'b', 
                  'topology', 
                  'ind_size', 
                  'is_connected', 
                  'connectivity_penalty',
                  'xtip',
                  'ytip') 
    
    def check_attributes(self):
        
        for a in Topology.attributes:
            if a not in self.__dict__:
                tup = (a, type(self).__name__)
                message = 'Attribute \'{}\' missing from {} class.'
                message = message.format(*tup)
                raise NotImplementedError(message)
    
    
    def update_topology(self, xs):
        
        tup = ('update_topology', type(self).__name__)
        message = 'Function \'{}\' missing from {} class.'
        message = message.format(*tup)
        raise NotImplementedError(message)
        
        
    def get_params(self):
        return (self.topology, self.a, self.b, self.xtip, self.ytip)
    