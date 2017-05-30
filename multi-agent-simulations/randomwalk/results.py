"""
@author: vtrianni and cdimidov
"""
import numpy as np
import operator

class Results:
    'A class to store the results '

    def __init__( self, length = 0 ):
        self.convergence = []
        self.convergence_time = []
	self.conv_time = []
        self.efficiency = []
        self.average_total_time = []
        self.total_visits = []
        self.percentage_tot_agents_with_info = []
        self.current_run = -1


    def new_run( self ):
        self.convergence.append(False)
        self.convergence_time.append(0)
	self.conv_time.append(0)
        self.efficiency.append(0)
        self.average_total_time.append(0)
        self.total_visits.append(0)
        self.percentage_tot_agents_with_info.append(0)
        self.current_run += 1
        

    #def update( self,time, n_informations ): 
    #    if( self.store_mean ):
    #        self.num_informations[time] += n_informations

   
            
    def store( self, convergence, time, conv_time, efficiency, average_total_time, total_visits, percentage_tot_agents_with_info):
        self.convergence[self.current_run] = convergence
        self.convergence_time[self.current_run] = time
	self.conv_time[self.current_run] = conv_time
        self.efficiency[self.current_run] = efficiency
        self.average_total_time[self.current_run] = average_total_time
        self.total_visits[self.current_run] = total_visits
        self.percentage_tot_agents_with_info[self.current_run] = percentage_tot_agents_with_info


    #def normalize( self ):
    #    if( self.store_mean ):
    #        self.num_informations /= (self.current_run+1)
          
            
    def save( self, data_filename, run_filename ):
        convergence_array = np.array(self.convergence)
        convergence_time_array = np.ma.array(self.convergence_time, mask=np.logical_not(convergence_array))
	conv_time_array = np.ma.array(self.conv_time, mask=np.logical_not(convergence_array))
        efficiency_array = np.ma.array(self.efficiency, mask=np.logical_not(convergence_array))
        average_total_time_array = np.ma.array(self.average_total_time, mask=np.logical_not(convergence_array))
        total_visits_array = np.ma.array(self.total_visits, mask=np.logical_not(convergence_array))
        percentage_tot_agents_with_info_array = np.ma.array(self.percentage_tot_agents_with_info, mask=np.logical_not(convergence_array))
        
        head = ' '.join(str(e) for e in [np.mean(convergence_array.astype(int)), np.mean(self.convergence_time), np.mean(self.conv_time) , np.mean(self.efficiency), np.mean(self.average_total_time), np.mean(self.total_visits), np.mean(self.percentage_tot_agents_with_info) ])
        np.savetxt(data_filename, np.column_stack((convergence_array,
                                                   convergence_time_array,
						   conv_time_array,
                                                   efficiency_array,
                                                   average_total_time_array,
                                                   total_visits_array,
                                                   percentage_tot_agents_with_info_array
                                                 )), fmt="%d %f %f %.5f %.5f %d %d" )
        
        #  if( self.store_mean ):
        #      np.savetxt(run_filename, np.column_stack((self.num_words)), fmt="%5.2f", header=head)
