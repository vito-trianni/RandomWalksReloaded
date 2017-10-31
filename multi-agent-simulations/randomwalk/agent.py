# -*- coding: utf-8 -*-
"""
@author: vtrianni and cdimidov
"""
import sys
import random, math, copy
import numpy as np
import string
from pysage import pysage
#from levy_f import stabrnd
from levy_f import distribution_functions
from scipy.stats import wrapcauchy
from scipy.stats import uniform
import scipy

#import numpy.random as R
#import scipyù print word
#from levy_f import MarkovC

class CRWLEVYAgent(pysage.Agent):
   
    bias = 0;
    linear_speed = 1
    num_motion_steps = 1
    interaction_range = 1

    
    class Factory:
        def create(self, config_element, arena): return CRWLEVYAgent(config_element, arena)

    ##########################################################################
    # standart init function
    ##########################################################################
    def __init__( self, config_element, arena ):
        pysage.Agent.__init__(self, config_element, arena )
        
        # parse custom parameters from configuration file
        
        # control parameter: motion speed
        sspeed = config_element.attrib.get("linear_speed")
        if sspeed is not None:
            CRWLEVYAgent.linear_speed = float(sspeed)
      
        # control parameter: interaction range
        srange = config_element.attrib.get("interaction_range")
        if srange is not None:
             CRWLEVYAgent.interaction_range = float(srange)

        # control parameter : value of CRW_exponent
        cc= config_element.attrib.get("CRW_exponent")
        if cc is not None:
            CRWLEVYAgent.CRW_exponent = float(cc)
            if (CRWLEVYAgent.CRW_exponent < 0) or (CRWLEVYAgent.CRW_exponent >= 1):
                raise ValueError, "parameter for correlated random walk outside of bounds ( should be in [0,1[ )"

        # control parameter : value of alpha that is the Levy_exponent
        salpha= config_element.attrib.get("levy_exponent")
        if salpha is not None:
            CRWLEVYAgent.levy_exponent = float(salpha)

        # control parameter : value of standard deviation 
        ssigma= config_element.attrib.get("std_motion_steps")
        if ssigma is not None:
            CRWLEVYAgent.std_motion_steps = float(ssigma)
        
        # bias_probability
        sbias_probability = config_element.attrib.get("bias_probability")
        if sbias_probability is not None:
            CRWLEVYAgent.bias_probability= float(sbias_probability)

        # counters for the number of steps used for straight
        self.count_motion_steps = 0
        
        # inventory of target in memory, and list of received target from neighbours
        self.inventory = []
        self.received_targets = []
        self.selected_target = 0

        # counter for the visited targets
        self.visited_target_id = []
        
        # counter time enter on the target
        self.step_on_target_time = []
       
        # counter first time enter on target
        self.first_time_step_on_target = []
        self.distance_from_target=[]

	#counter time enter on the central place
	## self.step_on_central_place_time = []

        # flag: true when over target
        self.on_target = False

	# flag: true when over central_place
	self.on_central_place = False
        

    ##########################################################################
    # String representaion (for debugging)
    ##########################################################################
    def __repr__(self):
        return 'CRWLEVY', pysage.Agent.__repr__(self)
       

    ##########################################################################
    # equality operator
    ##########################################################################
    def __eq__(self, other):
        if self.inventory_size() != other.inventory_size():
            return False
        return (self.inventory == other.inventory)
    

    ##########################################################################
    # disequality operator
    ##########################################################################
    def __ne__(self, other):
        if self.inventory_size() != other.inventory_size():
            return True
        return (self.inventory != other.inventory)
   
    
    
    ##########################################################################
    #  initialisation/reset of the experiment variables
    ##########################################################################
    def init_experiment( self ):
        pysage.Agent.init_experiment( self )
        
        #self.count_motion_steps = stabrnd.stabrnd(CRWLEVYAgent.levy_exponent, 0, CRWLEVYAgent.std_motion_steps, CRWLEVYAgent.average_motion_steps, 1, 1)
        self.count_motion_steps = int(math.fabs(distribution_functions.levy(CRWLEVYAgent.std_motion_steps,CRWLEVYAgent.levy_exponent)))
        del self.inventory[:]
        del self.received_targets[:]
        del self.visited_target_id[:]
        del self.step_on_target_time[:]
        del self.distance_from_target[:]
        # del self.step_on_central_place_time[:]

        self.on_target = False
	self.on_central_place = False
        
    ##########################################################################
    # compute the desired motion as a random walk
    ##########################################################################
    def control(self):
        # first check if some target is within the interaction range

        for t in self.arena.targets:
            if (t.position - self.position).get_length() < t.size:
                if not self.on_target:
                    self.on_target = True
		    self.count_motion_steps = 0
                    if t.id not in self.inventory:
                        self.inventory.append(t.id)
                    self.visited_target_id.append(t.id)
                    if len(self.step_on_target_time) > 0 : 			
                        self.step_on_target_time.append(self.arena.num_steps - self.step_on_target_time[-1])
                    else:
                        self.step_on_target_time.append(self.arena.num_steps)
            else:
                self.on_target = False
        if self.arena.arena_type=="unbounded" and (self.arena.num_steps+1)%5000 == 0:
            if self.arena.num_steps>0:
                for t in self.arena.targets:
                    self.distance_from_target.append((t.position - self.position).get_length())

        ## if self.arena_type=="unbounded":
        ##     if(self.arena.central_place.position - self.position).get_length() < self.arena.central_place.size:
        ##         if not self.on_central_place:
        ##             self.on_central_place = True
        ##             self.count_motion_steps = 0	
        ##             if len(self.step_on_central_place_time) > 0 : 
        ##                 self.step_on_central_place_time.append(self.arena.num_steps - self.step_on_central_place_time[-1])
        ##             else:
        ##                 self.step_on_central_place_time.append(self.arena.num_steps)                   
        ## 	else:
        ##         self.on_central_place = False

        # agent basic movement: go straight
        self.apply_velocity = pysage.Vec2d(CRWLEVYAgent.linear_speed,0)
        self.apply_velocity.rotate(self.velocity.get_angle())

        # agent random walk: decide step length and turning angle
        self.count_motion_steps -= 1
        if self.count_motion_steps <= 0:
            # step length
            self.count_motion_steps = int(math.fabs(distribution_functions.levy(CRWLEVYAgent.std_motion_steps,CRWLEVYAgent.levy_exponent)))
            #self.count_motion_steps = math.fabs(stabrnd.stabrnd(CRWLEVYAgent.levy_exponent, 0, CRWLEVYAgent.std_motion_steps, CRWLEVYAgent.average_motion_steps, 1, 1))
            # turning angle
            crw_angle = 0
            if self.arena.arena_type == "unbounded" and (scipy.random.random(1) <= CRWLEVYAgent.bias_probability ) :
                crw_angle=(self.arena.central_place.position - self.position).get_angle() - self.velocity.get_angle()	
            elif CRWLEVYAgent.CRW_exponent == 0:
                crw_angle = random.uniform(0,(2*math.pi))
            else:
                crw_angle = wrapcauchy.rvs(CRWLEVYAgent.CRW_exponent)

            self.apply_velocity.rotate(crw_angle)		 

            

             
    ##########################################################################
    # update the inventory according to the received targets
    ##########################################################################
    def update_inventory( self ):
        for t in self.received_targets:
            if t not in self.inventory:
                self.inventory.append(t)
        del self.received_targets[:]
        

    ##########################################################################
    # engage in communication with neighbouring agents
    ##########################################################################
    def broadcast_target(self):
        # broadcast a target to all neighbours
        self.selected_target = self.select_target()
        if self.selected_target is not None:
            neighbours = self.arena.get_neighbour_agents(self, CRWLEVYAgent.interaction_range)
            for agent in neighbours:
                agent.receive(self.selected_target)

    ##########################################################################
    # get a new target 
    # (possibly within a limited number of alterntives)
    ##########################################################################
    def select_target( self ):
        target = None
        if self.inventory:
            # select a random word from the inventory
            target  = random.choice(self.inventory) 
        return target
        
    ##########################################################################
    # receive an target from a neighbouting agent
    ##########################################################################
    def receive( self, target ):
        self.received_targets.append(target)
        
    ##########################################################################
    # set the selected flag to 'status'
    ##########################################################################
    def set_selected_flag( self, status ):
        self.selected = status
        if self.selected:
            print "agent",  self.id, "inventory:", self.inventory, self.position

    ##########################################################################
    # return the inventory size
    ##########################################################################
    def inventory_size( self ):
       return len(self.inventory)
    


pysage.AgentFactory.add_factory("randomwalk.agent", CRWLEVYAgent.Factory())
