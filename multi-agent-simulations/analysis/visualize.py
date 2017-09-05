#!/usr/bin/env python
"""
This is a script to automate visualising data in a batch.
1. First, files in the specified directory are parsed. Add your extension to the tail argument (in subtract) to feed correct data.
2. Calculate statistics by specifiying whether truncated or not. Usage: -t 0 or -t 1
3. Specify the logistics of the variables you wish to plot, using the plot design module. Subsetting is done on the basis of feed. Currently only supports heatmaps and lmplots.
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
from scipy.optimize import curve_fit
import matplotlib.backends.backend_pdf

hola=[]
pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")
def weib(x,alpha,gamma):
        return 1 - np.exp(-np.power(x/alpha,gamma))

class Weib(st.rv_continuous):
    def __init__(self,lower_bound=0):
        st.rv_continuous.__init__(self,a=lower_bound)
    def cdf(self,x,alpha,gamma):
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
            n_est=np.asarray(range(0,dataset.size))[::-1] + float(censored)
            RT_sync=[]
            for i in range(n_est.size):
                if len(RT_sync)==0:
                    RT_sync.append((n_est[i]-1)/n_est[i])
                else:
                    RT_sync.append(RT_sync[-1]*((n_est[i]-1)/n_est[i]))
            F=1-np.asarray(RT_sync).reshape(-1,1)
            my=Weib()
            popt,_= curve_fit(my.cdf,xdata=dataset,ydata=np.squeeze(F),bounds=(0,[1000000,10]),method='trf')                
            time_list.append(sc.gamma(1+(1./popt[1]))*popt[0])
            
    return np.asarray(time_list).reshape(1,-1)[0]    

def stats(p,label,truncated=1):
    global hola
    convergence_array = np.array(p[:,0])
    convergence_time_array = np.ma.array(p[:,1], mask=np.logical_not(convergence_array))
    censored_conv=np.sum(np.logical_not(convergence_array))
    print "No. of cases of no convergence",censored_conv
    conv_time_array = np.ma.array(p[:,2], mask=np.logical_not(convergence_array))
    visits_ratio_array = np.ma.array(p[:,3], mask=np.logical_not(convergence_array))
    percentage_tot_agents_with_info_array = np.ma.array(p[:,4], mask=np.logical_not(convergence_array))    
    first_passage_time_array=np.ma.array(expanded_stats(p[:,5:]),mask=np.logical_not(convergence_array))
    
    if truncated==1:
        convergence_time_array.mask=np.ma.nomask
        conv_time_array.mask=np.ma.nomask
        visits_ratio_array.mask=np.ma.nomask
        percentage_tot_agents_with_info_array.mask=np.ma.nomask
        first_passage_time_array.mask=np.ma.nomask
    censored_pass=np.sum(np.isnan(first_passage_time_array))
    print "No of cases of no passage",censored_pass
    dataset = np.column_stack((convergence_array,
                               convergence_time_array,
                               conv_time_array,
                               visits_ratio_array,
                               percentage_tot_agents_with_info_array,
                               first_passage_time_array))
        
    if (censored_conv==0 and censored_pass == 0) or truncated==0:
        head = [np.mean(convergence_array.astype(int)), np.mean(convergence_time_array.compressed()), np.mean(conv_time_array.compressed()) , np.mean(p[:,3]), np.mean(p[:,4]),np.mean(first_passage_time_array)]
    else:
        my=Weib()
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
                
                popt,_= curve_fit(my.cdf,xdata=compute[:,0].compressed(),ydata=np.squeeze(F),bounds=(0,[1000000,10]),method='trf')
                y2=my.cdf(compute[:,0],popt[0],popt[1])
                print popt[0],popt[1]
                Tsync2=sc.gamma(1+(1./popt[1]))*popt[0]
                
                fig=plt.figure()       
                #plt.plot(compute[:,0],ys,'r',linewidth=5,label="exponwebib fit")                
                plt.plot(compute[:,0],y2,'y',linewidth=5,label="curve fit")
                plt.plot(compute[:,0],F,'b',linewidth=5,label="K-M stats")
                plt.legend()
                plt.ylim(0,1)
                label="Alpha "+str(label[1])+" Rho "+str(label[2])+" Population "+str(int(label[3]))+" Time of Convergence"
                plt.title(label)
                plt.xlabel("Number of time steps")
                plt.ylabel("Synchronisation probability")
                pdf.savefig( fig )

                #plt.show()
                # Uncomment top line to view the fit

                #print ("Censored",Tsync)
                print "Censored",Tsync2
                convergence_time_array.mask=np.logical_not(convergence_array)
                print "Uncensored", np.mean(convergence_time_array.compressed())
                popt,_= curve_fit(my.cdf,xdata=compute[:,1].compressed(),ydata=np.squeeze(F),bounds=(0,[100000,10]),method='trf')
                Tsynt2=sc.gamma(1+(1./popt[1]))*popt[0]
            except:
                Tsync2=np.nan
                Tsynt2=np.nan

        else:
            Tsync2=np.mean(convergence_time_array.compressed())
            Tsynt2=np.mean(conv_time_array.compressed())

        data=dataset[np.argsort(dataset[:,-1])]
        data=data[:,-1]    
        ## First passage time Statistic
        if censored_pass!=0:
            try:
                data=data[:-censored_pass]
                n_est=np.asarray(range(0,data.size))[::-1] + float(censored_pass)
                RT_sync=[]
                my=Weib()
                for i in range(n_est.size):
                    if len(RT_sync)==0:
                        RT_sync.append((n_est[i]-1)/n_est[i])
                    else:
                        RT_sync.append(RT_sync[-1]*((n_est[i]-1)/n_est[i]))
                F=1-np.asarray(RT_sync).reshape(-1,1)
                popt,_= curve_fit(my.cdf,xdata=data,ydata=np.squeeze(F),bounds=(0,[1000000,10]),method='trf')
                Tpass2=sc.gamma(1+(1./popt[1]))*popt[0]
                fig=plt.figure()
                y2=my.cdf(data,popt[0],popt[1])
                plt.plot(data,y2,'r',linewidth=5,label="curve fit")
                plt.plot(data,F,'g',linewidth=5,label="K-M stats")
                plt.legend()
                plt.ylim(0,1)
                label="Alpha "+str(label[1])+" Rho "+str(label[2])+" Population "+str(int(label[3]))+" Time of First Passage"
                plt.title(label)
                plt.xlabel("Number of time steps")
                plt.ylabel("Synchronisation probability")
                pdf.savefig( fig )

            except:
                Tpass2=np.nan
        else:
            Tpass2=np.mean(data)
            
        head = [np.mean(convergence_array.astype(int)), Tsync2, Tsynt2, np.mean(p[:,3]), np.mean(p[:,4]),Tpass2]
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
            graph_limit=3.5e5
            title+=" for Bounded"
        else:
            title+=" for Unbounded"
            graph_limit=3.5e5

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
        data.to_csv('final_data.csv',header=False)

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
                fig=plt.figure()
                sns.heatmap(dummy,annot=True)
        elif plttype=="lmplot":
                scatter_kwargs={"s":100}
                data=data[~data.isnull()]
                
                fig=sns.lmplot(x=x.name,y=y.name,data=data,hue=out,legend_out=False,fit_reg=False,scatter=True,markers="o",size=7,aspect=1.3,scatter_kws=scatter_kwargs)
                #plt.ylim(0,y.length)

        sns.plt.xlabel(x.name)
        sns.plt.ylabel(y.name)
        sns.plt.suptitle(title)
        if plttype=="heatmap":
             pdf.savefig( fig )
        elif plttype=="lmplot":
             pdf.savefig( fig.fig )
        #plt.show()

def subtract(a, b,tail='.dat'):
    a="".join(a.rsplit(tail))
    return "".join(a.rsplit(b))  
def main():
    global hola
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
            if save:
                b=np.genfromtxt(filepath)
                k+=stats(b,k)
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
    pdf.close()


if __name__ == '__main__':
    main()
