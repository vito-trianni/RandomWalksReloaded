"""
@author: vtrianni and cdimidov
"""
import numpy as np
import operator
import scipy.stats as st
import scipy.special as sc

class Results:
    'A class to store the results '

    def __init__( self, length = 0 ):
        self.convergence = []
        self.convergence_time = []
        self.conv_time = []
        self.total_visits_fraction = []
        self.percentage_tot_agents_with_info = []
        self.first_passage_time = []
       
        self.current_run = -1


    def new_run( self ):
        self.convergence.append(False)
        self.convergence_time.append(0)
        self.conv_time.append(0)
        self.total_visits_fraction.append(0)
        self.percentage_tot_agents_with_info.append(0)
        self.first_passage_time.append(0)
        self.current_run += 1
        

    #def update( self,time, n_informations ): 
    #    if( self.store_mean ):
    #        self.num_informations[time] += n_informations

   
            
    def store( self, convergence, time, conv_time, total_visits_fraction, percentage_tot_agents_with_info, first_passage_time_list):
        self.convergence[self.current_run] = convergence
        self.convergence_time[self.current_run] = time
        self.conv_time[self.current_run] = conv_time
        self.total_visits_fraction[self.current_run] = total_visits_fraction
        self.percentage_tot_agents_with_info[self.current_run] = percentage_tot_agents_with_info
        self.first_passage_time[self.current_run]=first_passage_time_list


    #def normalize( self ):
    #    if( self.store_mean ):
    #        self.num_informations /= (self.current_run+1)
          
            
    def save( self, data_filename, run_filename ):
        convergence_array = np.array(self.convergence)
        convergence_time_array = np.ma.array(self.convergence_time)#, mask=np.logical_not(convergence_array))
        conv_time_array = np.ma.array(self.conv_time)#, mask=np.logical_not(convergence_array))
        total_visits_fraction_array = np.ma.array(self.total_visits_fraction)#, mask=np.logical_not(convergence_array))
        percentage_tot_agents_with_info_array = np.ma.array(self.percentage_tot_agents_with_info)#, mask=np.logical_not(convergence_array))
        first_passage_time=np.ma.array(self.first_passage_time)
        #head = ' '.join(str(e) for e in [np.mean(convergence_array.astype(int)), np.mean(self.convergence_time), np.mean(self.conv_time) , np.mean(self.efficiency), np.mean(self.average_total_time), np.mean(self.total_visits), np.mean(self.percentage_tot_agents_with_info) ])
        
        np.savetxt(data_filename, np.column_stack((convergence_array,
                                                   convergence_time_array,
                                                   conv_time_array,
                                                   total_visits_fraction_array,
                                                   percentage_tot_agents_with_info_array,
                                                   first_passage_time
                                                 )), fmt="%d %.0f %.0f %.2f %d"+" %.0f"*first_passage_time.shape[1])#, header=head)
