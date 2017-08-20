#!/usr/bin/env python
"""
This is a script to automate visualising data in a batch.
1. First files in the specified directory are parsed. Add your extension to the tail argument in subtract to feed correct data.
2. Calculate statistics according to the need by specifiying whether truncated or not. Usage: -t 0 or -t 1
3. Specify the logistics of the variables you wish to plot using the plot design module. Subsetting is done on the basis of feed. Currently only support heatmaps and lmplots.
4. Use -u 0 or -1 for arena-specs 
5. Use -l to specify prefix for time limit-specs
6. Use -p for the path where .dat files are stored
author: katanachan
email: Aishwarya Unnikrishnan <shwarya.unnikrishnan@gmail.com>
"""
print __doc__

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import glob
import os
from itertools import groupby
import scipy.stats as st
import scipy.special as sc
import copy

def weib(x,alpha,gamma):
        return 1 - np.exp(-np.power(x/alpha,gamma))

class Weib(st.rv_continuous):
    def __init__(self,lower_bound=0):
        st.rv_continuous.__init__(self,a=lower_bound)
    def _cdf(self,x,alpha,gamma):
        return 1 - np.exp(-np.power(x/alpha,gamma))
def expanded_stats(data):
    time_list=[]
    for i in range(data.shape[0]):
        censored=np.sum(np.isnan(data[i,:]))
        if censored==data.shape[1]:
            time_list.append(np.nan)
        else:
            dataset=data[i,:]
            dataset=dataset[np.argsort(dataset)]
            dataset=dataset[:-censored]
            _,gamma,_,alpha=st.exponweib.fit(dataset,floc=0,f0=1)
            time_list.append(sc.gamma(1+(1/gamma))*alpha)
            
    return np.asarray(time_list).reshape(1,-1)[0]    
        
    
        
def stats(p,truncated=True):
    convergence_array = np.array(p[:,0])
    convergence_time_array = np.ma.array(p[:,1], mask=np.logical_not(convergence_array))
    censored_conv=np.sum(np.logical_not(convergence_array))
    print ("Censored Vallues",censored_conv)
    conv_time_array = np.ma.array(p[:,2], mask=np.logical_not(convergence_array))
    visits_ratio_array = np.ma.array(p[:,3], mask=np.logical_not(convergence_array))
    percentage_tot_agents_with_info_array = np.ma.array(p[:,4], mask=np.logical_not(convergence_array))    
    first_passage_time_array=np.ma.array(expanded_stats(p[:,5:]),mask=np.logical_not(convergence_array))
    
    if truncated==True:
        convergence_time_array.mask=np.ma.nomask
        conv_time_array.mask=np.ma.nomask
        visits_ratio_array.mask=np.ma.nomask
        percentage_tot_agents_with_info_array.mask=np.ma.nomask
        first_passage_time_array.mask=np.ma.nomask
    censored_pass=np.sum(np.isnan(first_passage_time_array))
    dataset = np.column_stack((convergence_array,
                               convergence_time_array,
                               conv_time_array,
                               visits_ratio_array,
                               percentage_tot_agents_with_info_array,
                               first_passage_time_array))
        
    if (censored_conv==0 and censored_pass == 0) or truncated==False:
        head = [np.mean(convergence_array.astype(int)), np.mean(convergence_time_array.compressed()), np.mean(conv_time_array.compressed()) , np.mean(p[:,3]), np.mean(p[:,4]),np.mean(p[:,5])]
    else:
        data=dataset[np.argsort(dataset[:,1])]
        n_est=np.asarray(range(0,conv_time_array.compressed().size-censored_conv))[::-1] + float(censored_conv)
        RT_sync=[]
        for i in range(n_est.size):
            if len(RT_sync)==0:
                RT_sync.append((n_est[i]-1)/n_est[i])
            else:
                RT_sync.append(RT_sync[-1]*((n_est[i]-1)/n_est[i]))
        F=1-np.asarray(RT_sync).reshape(-1,1)
        if censored_conv !=0:
            compute=np.concatenate((data[:-censored_conv,1:3],F),1)
        try:
            m,gamma,c,alpha=st.exponweib.fit(compute[:,0],floc=0,f0=1)
            my=Weib()
            #print alpha,gamma,compute[:,0].shape
            ys=my.cdf(compute[:,0],alpha,gamma)
            #error= np.mean(np.power(ys-F,2))
            #error2=np.mean(np.power(y2-F,2))
            #print "ERROR", error-error2
        
            plt.plot(compute[:,0],ys,'r',linewidth=5)
            plt.plot(compute[:,0],F,'b',linewidth=5)
            #plt.show()
            # Uncomment top line to view the fit

            Tsync=(sc.gamma(1+(1/gamma))*alpha)
            print ("Censored",Tsync)
            convergence_time_array.mask=np.logical_not(convergence_array)
            print ("Uncensored", np.mean(convergence_time_array.compressed()))
            _,gamma,_,alpha=st.exponweib.fit(compute[:,1],floc=0,f0=1)
            Tsynt=sc.gamma(1+(1/gamma))*alpha
        except:
            Tsync=np.nan
            Tsynt=np.nan

        ## First passage time Statistic
        try:
            data=dataset[np.argsort(dataset[:,-1])]
            data=data[:,-1]
            if censored_pass!=0:
                data=data[:-censored_pass]
            _,gamma,_,alpha=st.exponweib.fit(data,floc=0,f0=1)
            Tpass=sc.gamma(1+(1/gamma))*alpha
        except:
            Tpass=np.nan
            
        head = [np.mean(convergence_array.astype(int)), Tsync, Tsynt, np.mean(p[:,3]), np.mean(p[:,4]),Tpass]
    return head

class Variable_Dictionary:
	def __init__(self,variable_name,range_val,constant_val):
		self.name=variable_name
		self.length=range_val
		self.fixed=constant_val
		self.copied=False
                self.average=0
        def get_fixed(self):
                return self.fixed
        def get_copied(self):
                return self.copied

def plot_design(data,x,y,out,plttype,title,unbounded,rho=None,alpha=None,size=None):
	## General preprocessing
	data.index=map(str,np.arange(0,data.shape[0],1))
	plttype=str(plttype)
	title=str(title)
        if unbounded == 0:
                graph_limit=3.5e4
                title+=" for Bounded"
        else:
                title+=" for Unbounded"
                graph_limit=3.5e4

	levy_alpha_range=map(str,np.linspace(1.1,2.0,10))
	crw_rho_range=map(str,np.linspace(0,0.9,7))
	pop_n_range=map(str,np.linspace(10,100,10))

	levy_alpha=Variable_Dictionary("Levy-Exponent-Alpha",levy_alpha_range,alpha)
	crw_rho=Variable_Dictionary("CRW-Exponent-Rho",crw_rho_range,rho)
	pop_n=Variable_Dictionary("Population-Size",pop_n_range,size)
        convergence_time=Variable_Dictionary("Convergence-Time",graph_limit,None)

	dict_variables=[levy_alpha,crw_rho,pop_n,convergence_time]
	z=None
	dict_inputs=[x,y]
	## Additionally add z?

	for i in range(len(dict_inputs)):
		dict_input=dict_inputs[i].split('-')[0]
		for j in range(len(dict_variables)):
			if dict_input==dict_variables[j].name.split('-')[0]:
				dict_inputs[i]=copy.deepcopy(dict_variables[j])
				dict_variables[j].copied=True
        for i in range(len(dict_variables)):
		if dict_variables[i].copied==False and dict_variables[i].fixed!=None:
			z=copy.deepcopy(dict_variables[i])
			dict_inputs.append(z)                
        if np.all(np.logical_not(map(Variable_Dictionary.get_fixed,dict_variables))):
                unused_to_average= np.where(np.logical_not(map(Variable_Dictionary.get_copied,dict_variables[0:3])))[0][0]
                ## Changed this line to choose between out and unused variables
                z=copy.deepcopy(dict_variables[unused_to_average])
                z.average=len(list(set(np.unique(data[z.name])).intersection(map(int,map(float,z.length)))))
        else:
                for i in range(len(dict_inputs)):
                        if dict_inputs[i].fixed != None:
                                data=data[data[dict_inputs[i].name]==dict_inputs[i].fixed]

	## Subset
        x=copy.deepcopy(dict_inputs[0])
        y=copy.deepcopy(dict_inputs[1])

	if plttype=="heatmap":
                ## Make bins
		bin_y=np.asarray((np.asarray(data[y.name])-float(y.length[0]))/(float(y.length[-1])-float(y.length[-2])))
		bin_x=np.asarray((np.asarray(data[x.name])-float(x.length[0]))/(float(x.length[-1])-float(x.length[-2])))
                ## Uncomment this to verify your data
		#data.to_csv('final_data.csv',header=False)
		bin_y=len(y.length)-bin_y-1		
		dummy = np.zeros((len(y.length),len(x.length)))

		## Fill values

		for k in range(data.shape[0]):
                        # I am putting a -1 in these indices because Python2.7 seems to convert float 2.0 to int 1. Weird. Pls remove if you don't have this bug.
                        #print bin_x[k]
                        #print int(bin_x[k])
			dummy[int(bin_y[k]),int(bin_x[k]+1)]+=data[out][k]
                ## A const value check:
                if z.average!=0:
                        dummy/=z.average
                ## Label heat map
                dummy=pd.DataFrame(dummy)
                dummy.columns=x.length
                dummy.index=y.length[::-1]
                sns.heatmap(dummy,annot=True)
        elif plttype=="lmplot":
    	        sns.lmplot(x=x.name,y=y.name,data=data,hue=out,x_jitter=0.05,legend_out=False,fit_reg=True,scatter=True,order=2,markers="x")
                plt.ylim(0,y.length)
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.title(title)
        plt.show()

def subtract(a, b,tail='.dat'):
    a="".join(a.rsplit(tail))
    return "".join(a.rsplit(b))  
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--truncated", help="1 or 0")
    ap.add_argument("-u", "--unbounded", help="0 or 1")
    ap.add_argument("-p", "--path", help="enter path")
    ap.add_argument("-l", "--timelimit", help="Enter unique filename prefix for time limit") 
    args=vars(ap.parse_args())
    if args.get("truncated") is None:
        truncated=0
    else:
        truncated=int(args["truncated"])
    unbounded=int(args["unbounded"])
    path=str(args["path"])
    if args.get("timelimit") is None:
        prefix='result'
    else:
        prefix=str(args["timelimit"])
    log=[]

    for filepath in glob.glob(os.getcwd()+'/'+path+'*.dat'):
    	name_labels=[]
        k=[]
        save=False
        if filepath.split('/')[-1].split('_')[0] == prefix:
            file_dict=subtract(filepath,os.getcwd()+'/'+path+prefix+'_').split('_')
            for i in range(len(file_dict)):
                #name_labels.append(str([''.join(g) for _, g in groupby(file_dict[i], str.isalpha)][0]))
                k.append(float([''.join(g) for _, g in groupby(file_dict[i], str.isalpha)][1]))
            if unbounded == 1 and len(k)==5:
        	    #name_labels=name_labels[1:]
        	    k=k[1:]
        	    save = True
            elif unbounded == 0 and len(k)==4:
                save = True
            print k
            if save:
                b=np.genfromtxt(filepath)
                k+=stats(b)
                log.append(k)
    log=pd.DataFrame(log)
    log.columns=["Bias","Levy-Exponent-Alpha","CRW-Exponent-Rho","Population-Size"]+["Convergence_Count","Convergence-Time","Relative Convergence Time",
                                                                                     "Ratio of Total Visits","Percentage of Total Agents with Info","First Time of Passage"]

    plot_design(log,"CRW-Exponent-Rho","Levy-Exponent-Alpha","First Time of Passage","heatmap","Average First Passage Time for all Populations",unbounded)
    plot_design(log,"CRW-Exponent-Rho","Levy-Exponent-Alpha","Ratio of Total Visits","heatmap","PhiC ratio of Visited to Total Agents",unbounded)
    plot_design(log,"Population-Size","Convergence-Time","CRW-Exponent-Rho","lmplot","Convergence Times for Alpha=1.6",unbounded,alpha=1.6)
    plot_design(log,"Population-Size","Convergence-Time","Levy-Exponent-Alpha","lmplot","Convergence Times for Rho=0.6",unbounded,rho=0.6)
    plot_design(log,"CRW-Exponent-Rho","Levy-Exponent-Alpha","Convergence-Time","heatmap","Total Convergence Time for Population 10",unbounded,size=10)
    plot_design(log,"CRW-Exponent-Rho","Levy-Exponent-Alpha","Convergence-Time","heatmap","Total Convergence Time for Population 100",unbounded,size=100)

if __name__ == '__main__':
    main()
