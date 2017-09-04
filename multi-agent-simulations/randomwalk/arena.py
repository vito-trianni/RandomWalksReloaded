# -*- coding: utf-8 -*-
"""
@author: vtrianni and cdimidov
"""
import numpy as np
import math, random
import sys
from pysage import pysage
from results import Results
from agent import CRWLEVYAgent
from target import Target
from collections import defaultdict


class CRWLEVYArena(pysage.Arena):
    
    class Factory:
        def create(self, config_element): return CRWLEVYArena(config_element)
    
    ##########################################################################
    # class init function
    ##########################################################################
    def __init__(self,config_element ):
        pysage.Arena.__init__(self,config_element)

        self.results_filename   = "CRWLEVY.dat" if config_element.attrib.get("results") is None else config_element.attrib.get("results")
        self.results = Results()

        # is the experiment finished?
        self.has_converged = False
        self.convergence_time = float('nan')

        # control parameter: num_targets
        self.target_size = 0.02 if config_element.attrib.get("target_size") is None else float(config_element.attrib["target_size"])

        # control parameter: num_targets
        if config_element.attrib.get("num_targets") is None:
            print "[ERROR] missing attribute 'num_targets' in tag <arena>"
            sys.exit(2)
        self.num_targets = int(config_element.attrib["num_targets"])

        nnruns=  config_element.attrib.get("num_runs")
        if nnruns is not None:
            self.num_runs=int(nnruns)
        else:
            self.num_runs=1

        # create the targets
        self.targets = []
        for i in range(0,self.num_targets):
            self.targets.append(Target(pysage.Vec2d(0,0), self.target_size, self))

        # control flag : value of flag
        self.central_place = None
        sunbounded = config_element.attrib.get("unbounded")
        if sunbounded is not None:
            self.unbounded = int(sunbounded)
    
        #  size_radius
        ssize_radius = config_element.attrib.get("size_radius");
        if ssize_radius is not None:
            self.dimensions_radius = pysage.Vec2d( map(float, ssize_radius.split(',')) )


        # control parameter: target_distance
        starget_distance = config_element.attrib.get("target_distance")
        if starget_distance is not None:
            CRWLEVYAgent.target_distance = float(starget_distance)
        
        # variable in wich is stored the min among first passage time
	self.min_first_time = 0.0
	
        # variable that represents the difference between convergence time and min of first passage time
	#self.conv_time = 0.0
	
	
    ##########################################################################
    # initialisation of the experiment  
    ##########################################################################
    def init_experiment( self ):
        pysage.Arena.init_experiment(self)
        print "Hello!!"

        if self.unbounded:
            # unbounded arena has a central palce
            self.central_place = Target(pysage.Vec2d(0,0), self.target_size, self)
            # agents initialised in the center
            for agent in self.agents:
                agent.position = pysage.Vec2d(0,0)
            # targets initialised in circular sector 
            for target in self.targets:    ##### Different initalisation if bounded or unbounded
                # target_position_radius = random.uniform(self.dimensions_radius.x,self.dimensions_radius.y);
                target_position_angle  = random.uniform(-math.pi,math.pi);
                target.position = pysage.Vec2d(CRWLEVYAgent.target_distance*math.cos(target_position_angle),CRWLEVYAgent.target_distance*math.sin(target_position_angle))
        else:     
            # targets initialised anywhere in the bounded arena
            for target in self.targets:    ##### Different initalisation if bounded or unbounded
                target.position = pysage.Vec2d(random.uniform(-self.dimensions.x/2+CRWLEVYAgent.target_distance,self.dimensions.x/2-CRWLEVYAgent.target_distance),random.uniform(-self.dimensions.y/2+CRWLEVYAgent.target_distance,self.dimensions.y/2-CRWLEVYAgent.target_distance))
             # agents initialised anywhere in the bounded arena
            for agent in self.agents:
                 on_target = True
                 while on_target:  
                       agent.position = pysage.Vec2d(random.uniform(-self.dimensions.x/2,self.dimensions.x/2),random.uniform(-self.dimensions.y/2,self.dimensions.y/2))
                       on_target= False
                       for t in self.targets:   
                           if (t.position - agent.position).get_length() < t.size:                       
                               on_target = True
                               break            
    
        self.inventory_size = 0
        self.has_converged = False
        self.convergence_time = float('nan')
        self.results.new_run()
        self.min_first_time =0.0
        #self.conv_time = 0.0

    ##########################################################################
    # run experiment until finished
    def run_experiment( self ):
        while not self.experiment_finished():
            self.update()
    ##########################################################################
    # updates the status of the simulation
    ##########################################################################
    def update( self ):
        # computes the desired motion and agent state
        for a in self.agents:
            a.control()
            
        # apply the desired motion and update the agent inventory
        self.inventory_size = 0
        for a in self.agents:
            a.update()
            a.update_inventory()
            if self.unbounded == 0:
                if a.position.x < -0.5:
                    a.position.x = -0.5
                elif a.position.x > self.dimensions.x/2:
                    a.position.x = self.dimensions.x/2 

                if a.position.y < -0.5:
                    a.position.y = -0.5
                elif a.position.y > self.dimensions.y/2:
                    a.position.y = self.dimensions.y/2
  
            self.inventory_size += a.inventory_size()

        # broadcast a target 
        for a in self.agents:
           a.broadcast_target()

        # check convergence
        self.has_converged = (self.inventory_size == self.num_agents*self.num_targets)
        if self.has_converged and math.isnan(self.convergence_time):
            self.convergence_time = self.num_steps

        # update simulation step counter
        self.num_steps += 1

    
   
    ##########################################################################
    # return a list of neighbours
    #####################################################################
    def get_neighbour_agents( self, agent, distance_range ):
        neighbour_list = []
        for a in self.agents:
            if (a is not agent) and ((a.position - agent.position).get_length() < distance_range):
                neighbour_list.append(a)
        return neighbour_list

    ##########################################################################
    # compute total_time of first passages
    ##########################################################################
    def compute_total_time( self ):
        first_time = []  
        total_time = 0.0
	self.min_first_time = 0.0
        for a in self.agents:
            first_time.extend(a.step_on_target_time[0:1]) #list that contains all first arrival time
	if first_time !=[]:
 	    self.min_first_time = min(first_time) 
        total_time = sum(first_time)  # sum of the first arrival times 
        return total_time    
     
    #########################################################################
    # compute total_visits
    ##########################################################################
    def compute_total_visits( self ):
        total_visits = 0.0     
        for a in self.agents:
            total_visits += len(a.visited_target_id[0:1])
        return total_visits

    ##########################################################################
    # compute efficiency - Not using this anymore
    ##########################################################################
    def compute_efficiency( self ):
        total_time = self.compute_total_time()
        total_visits = self.compute_total_visits()
        try:
           efficiency = total_visits/total_time
        except ZeroDivisionError:
           efficiency = float('nan')
        return efficiency  # number of different target visited / total_time of first passage

    ##########################################################################
    # compute average_total_time of first passages
    ##########################################################################
    def compute_average_total_time (self):
        total_time = self.compute_total_time()
        total_visits = self.compute_total_visits()
        try:
           average_total_time = total_time/total_visits
        except ZeroDivisionError:
           average_total_time = float('nan')
        return average_total_time 

    ##########################################################################
    # return list of first passage times
    ##########################################################################
    def first_passage_time_list(self):
        first_times=[]
        for a in self.agents:
            try:
                first_times.append(a.step_on_target_time[0:1][0])
            except:
                first_times.append(np.nan) #list that contains all first arrival time
        return first_times
	
            
    ##########################################################################
    # check if the experiment si finished
    ##########################################################################
    def experiment_finished( self ):
        total_visits = self.compute_total_visits()
        conv_time = 0.0
        if ((self.max_steps > 0) and (self.max_steps <= self.num_steps)) or (total_visits == self.num_agents):
	    min_first_time = self.compute_total_time()
            first_passage_times = self.first_passage_time_list()
            conv_time =  self.convergence_time - self.min_first_time
            percentage_tot_agents_with_info = (self.inventory_size*100)/self.num_agents
            total_visits_fraction=total_visits/float(self.num_agents)
            print "run finished: ", self.has_converged, self.convergence_time, conv_time, total_visits_fraction, percentage_tot_agents_with_info, first_passage_times
            self.results.store(self.has_converged, self.convergence_time, conv_time,total_visits_fraction, percentage_tot_agents_with_info,first_passage_times)
            return True
        return False
        
    ##########################################################################
    # save results to file, if any
    ##########################################################################
    def save_results( self ):
        self.results.save(self.results_filename,None)
            
pysage.ArenaFactory.add_factory("randomwalk.arena", CRWLEVYArena.Factory())
