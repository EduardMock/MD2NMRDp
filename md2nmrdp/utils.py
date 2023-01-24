

import os, fnmatch
import numpy as np
from tkinter import Tcl
from scipy.integrate import simps 


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

# # check how many key has nested dict for every dict and return list of keys
# def count(d):
#     layer =[ count(v) if isinstance(v, dict) else 1 for v in d.values() ]
#     # sum 
#     return 

        



def find_filepaths(system_path, Temp, stype,job_dir="dipole_relax", collect_total=True,  pattern=' '):
    dd_dict=dict()
    for T in Temp:
        dd_dict[T]=dict()
        # print("%i K"%T)
        dir_path=f"{system_path}/{T}K/{job_dir}/{stype}/"
        content= listdir_fullpath(dir_path)
        
        for key in pattern:
            pat1=f'*{key}*'
            files= Tcl().call('lsort', '-dict', fnmatch.filter(content, pat1) )
            dd_dict[T][key]=dict()
            if collect_total and key.split('_')[0] == 'inter':
                pat2=f'*total*'
                res_files= Tcl().call('lsort', '-dict', fnmatch.filter(files, pat2) )
                dd_dict[T][key]=res_files        
            else:
                for value in pattern[key]:
                    pat2=f'*{value}*'
                    res_files= Tcl().call('lsort', '-dict', fnmatch.filter(files, pat2) )
                    dd_dict[T][key][value]=res_files
                
    return dd_dict


# def integrate_rdf(gr,r,rho):
#     cum_sum=0
#     for gr_i,r_i in zip(gr,r):
#         cum_sum+= gr_i* np.reciprocal(r_i)**4
#     cum_sum*=4*np.pi*rho
#     return cum_sum





def lorentz(w, tau,A):
    return    A*tau / (1+(tau*w)**2)


def cole_davison(w, tau,e):
    return np.sin( e* np.arctan(w*tau)) / ( w*(1+(tau*w)**2)**(e/2) )
