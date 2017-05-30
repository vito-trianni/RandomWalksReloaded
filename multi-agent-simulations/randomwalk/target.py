"""
@author: vtrianni and cdimidov
"""
import random, math, copy
import numpy as np
from pysage import pysage


########################################################################################
## Pysage Target
########################################################################################

##########################################################################
# factory to dynamically create agents
class TargetFactory:
    factories = {}
    def add_factory(id, target_factory):
        TargetFactory.factories[id] = target_factory
    add_factory = staticmethod(add_factory)

    def create_target(config_element, arena):
        id = config_element.attrib.get("pkg")
        if id is None:
            return Target.Factory().create(config_element, arena)
        id = id + ".target"
        return TargetFactory.factories[id].create(config_element, arena)
    create_target = staticmethod(create_target)


##########################################################################
# the main agent class
class Target:
    'Definition of a target in 2D space'
    num_targets = 0
    arena      = None

    class Factory:
        def create(self, config_element, arena): return Target(config_element, arena)


    ##########################################################################
    # standard initialisation
    def __init__(self, position, size, arena ):
        # identification target
        self.id = Target.num_targets
    
        # position in meters
        self.position = position
        
        # size of the target (radius)
        self.size = size

        # reference to the arena
        Target.arena = arena

        # inclrease the global counter
        Target.num_targets += 1

        

    ##########################################################################
    # String representaion (for debugging)
    def __repr__(self):
        return 'Target %d(%s, %s)' % ( self.id, self.position.x, self.position.y)


    ##########################################################################
    # set the selected flag to 'status'
    def set_selected_flag( self, status ):
        self.selected = status




# pysage.TargetFactory.add_factory("CRWLEVY.target", CRWLEVYTarget.Factory())



