

import os, fnmatch
import numpy as np
import pandas as pd
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



def find_filepaths(system_path, Temp, stype=None,job_dir="dipole_relax", collect_total=True, inter=True,  pattern=' '):
    dd_dict=dict()
    for T in Temp:
        dd_dict[T]=dict()
        # print("%i K"%T)
        if stype is not None:
            dir_path=f"{system_path}/{T}K/{job_dir}/{stype}/"
        else:
            dir_path=f"{system_path}/{T}K/{job_dir}/"
            
        content= listdir_fullpath(dir_path)
        
        for key in pattern:
            pat1=f'*{key}*'
            files= filter_files(content, pat1)
            dd_dict[T][key]=dict()
            
            
            cond="collect_total"
            
            if eval(cond):
                pat2=f'*total*'
                res_files= filter_files(files, pat2)
                
                if len(res_files) == 0:
                    for value in pattern[key]:
                        pat2=f'*{value}*'
                        res_files= filter_files(files, pat2) 
                        if len(res_files) > 0:
                            dd_dict[T][key][value]=res_files
                        else:
                            pat2=f'*major*'
                            res_files= filter_files(files, pat2) 
                            dd_dict[T][key]=res_files
                dd_dict[T][key]=res_files
                
                
            else:
                for value in pattern[key]:
                    pat2=f'*{value}*'
                    res_files= filter_files(files, pat2) 
                    if len(res_files) > 0:
                        dd_dict[T][key][value]=res_files
                    else:
                        pat2=f'*major*'
                        res_files= filter_files(files, pat2) 
                        dd_dict[T][key]=res_files
                    
                        
                    
                
    return dd_dict


def transpose_dict(InputDict):
    df=pd.DataFrame(InputDict)
    return df.T.to_dict()



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



def CD(w, e, t):
    return np.sin(e* np.arctan( w*t)) / ( w* (1+  (w*t)**2)**(e/2) )  

def BBP(w, t):
    return t/ (1+  w**2 * t**2)



from scipy.optimize import differential_evolution
import warnings


def fit_func( func, xData, yData, parameterBounds, bounds, p0=None):
    
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = func(xData, *parameterTuple)
        return np.sum((yData - val) ** 2.0)

    if p0 is None:
        # parameterBounds = []
        # parameterBounds.append([0, 1]) # search bounds for a
        # parameterBounds.append([0, 1000.0]) # search bounds for b

        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        p0= result.x

    if bounds is not None:
        fittedParameters, pcov = curve_fit(func, xData, yData , p0 , bounds=bounds, maxfev=10000 ) 
    else:
        fittedParameters, pcov = curve_fit(func, xData, yData , p0 , maxfev=10000 )
        
    print(fittedParameters)
    points=100
    xmin=min(xData)
    xmax=max(xData)
    
    xlist = np.linspace(xmin, xmax,points)
    ylist = func(xlist, *fittedParameters)

    return (xlist, ylist, fittedParameters) 

