

import os, fnmatch
import numpy as np
from tkinter import Tcl
from scipy.integrate import simps 


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def filter_files(files, pattern):
    return Tcl().call('lsort', '-dict', fnmatch.filter(files, pattern) )

# # check how many key has nested dict for every dict and return list of keys
# def count(d):
#     layer =[ count(v) if isinstance(v, dict) else 1 for v in d.values() ]
#     # sum 
#     return 



def find_filepaths(system_path, Temp, stype,job_dir="dipole_relax", collect_total=True, inter=True,  pattern=' '):
    dd_dict=dict()
    for T in Temp:
        dd_dict[T]=dict()
        # print("%i K"%T)
        dir_path=f"{system_path}/{T}K/{job_dir}/{stype}/"
        content= listdir_fullpath(dir_path)
        
        for key in pattern:
            pat1=f'*{key}*'
            files= filter_files(content, pat1)
            dd_dict[T][key]=dict()
            
            
            cond="collect_total"
            if inter==True:
                cond+= " and key.split('_')[0] == 'inter'"
            
            if eval(cond):
                pat2=f'*major*'
                res_files= filter_files(files, pat2) 
                dd_dict[T][key]=res_files        
            else:
                for value in pattern[key]:
                    pat2=f'*{value}*'
                    res_files= filter_files(files, pat2) 
                    if len(res_files) > 0:
                        dd_dict[T][key][value]=res_files
                    else:
                        pat2=f'*total*'
                        res_files= filter_files(files, pat2) 
                        dd_dict[T][key]=res_files
                    
                        
                    
                
    return dd_dict


# def integrate_rdf(gr,r,rho):
#     cum_sum=0
#     for gr_i,r_i in zip(gr,r):
#         cum_sum+= gr_i* np.reciprocal(r_i)**4
#     cum_sum*=4*np.pi*rho
#     return cum_sum


def integrate(x_axis, y_axis, prefactor=None):
    N_points = len(x_axis)
    res = np.zeros(N_points)
    
    # print(x_axis,y_axis)
    for x in range(1, N_points):
        res[x] = simps(y_axis[:x], x_axis[:x])

    if prefactor is not None:
        res *= prefactor
        
    return res


def rdf_r4(r,gr):
    N_points=len(gr)
    rdf_r= np.zeros(N_points)
    for count in np.arange(N_points):
        rdf_r[count]= gr[count]* np.reciprocal(r[count])**4
    return rdf_r*4*np.pi



def VFT(T,eta0,B,T0):   
    return  eta0*np.exp( B/(T-T0) )



def lorentz(w, tau,A):
    return    A*tau / (1+(tau*w)**2)


def cole_davison(w, tau,e):
    return np.sin( e* np.arctan(w*tau)) / ( w*(1+(tau*w)**2)**(e/2) )
