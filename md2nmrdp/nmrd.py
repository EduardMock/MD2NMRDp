
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fftfreq,dct
from scipy.optimize import curve_fit
from . import utils as nmrdu


from copy import deepcopy
import matplotlib.pyplot as plt





class DDrelax():
    
    def __init__(self, system, resname_dict, interactions, Temp, spin=1/2,dt_si=1e-12,dist_si=1e-10):
        
        self.system=system
        self.resnames=resname_dict
        self.Temp=Temp
        self.all_ia= interactions
        
        self.spin= spin
        self.dt_si=dt_si   #unit of timestep 
        self.dist_si=dist_si   #unit of distance 
        
        self.dist_dict= dict()
        
        # simulated regimes
        self.g2s= None  #short
        self.g2i=None   #inter
        self.g2l= None  #long

        
        
        #constants
        perm= 1.25663706212e-06  #N * A^-2
        hbar= 1.0545718176461565e-34 # J*s^-1
        
        self.multiplicity= self.spin*(self.spin+1)
        self.constants= (perm*hbar/(4*np.pi) )**2
        # self.si_conversion=np.reciprocal(self.dist_si)**6 *self.dt_si
        
    
        
        
        
    @staticmethod
    def convert_average(files):
        if len(files)==1:
            xData,yData=np.loadtxt(files[0], usecols=[0,1],unpack=True)
            return xData, yData
        else:
            cum_sum=[]
            for f in files:
                xData,yData=np.loadtxt(f , usecols=[0,1],unpack=True)
                cum_sum.append(yData)
            return xData, np.mean(cum_sum,axis=0)
    
    
    def check_for_att(self, function, restype=''):
        collect=[]
        sim_types=[]
        for stype in ['s','i','l','m']:
            try:
                collect.append(getattr(self, function+stype+restype))
                sim_types.append(stype)
            except:
                pass

        return collect, sim_types
    
    def set_stype(self, function, stype,  value, add_stype= True, restype=''):
        
        if len(stype) > 1 and add_stype:
            setattr(self, function+stype[0]+restype, value)
        elif len(stype) > 1 and  not add_stype:
            setattr(self, function+restype, value)
        else:
            setattr(self, function+stype+restype, value)



    def convert_files(self, filedict, stype):
        con_all=dict()
        con_av=dict()
        dist_dict=dict()
        for T in filedict.keys():
            con_all[T]=dict()
            con_av[T]=dict()
            dist_dict[T]=dict()
            for pat1 in filedict[T].keys() :
                con_all[T][pat1]=dict()
                #check if tupel or dict
                if isinstance( filedict[T][pat1] ,dict): 
                    # sum up contributions of different atomgroups 
                    cum_sum=0
                    
                    for pat2 in filedict[T][pat1].keys():
                        #convert and average intermolecular contribution over residues
                        xData, yData = self.convert_average(filedict[T][pat1][pat2])
                        con_all[T][pat1][pat2]= yData
                        cum_sum+=yData
                    dist_dict[T][pat1]= cum_sum[0]
                    con_av[T][pat1]=cum_sum
                    
                else: 
                    xData, yData = self.convert_average(filedict[T][pat1])
                    dist_dict[T][pat1]= yData[0]
                    con_av[T][pat1]= yData

        if stype in ['short','inter','long']:
            print(f"G2: stype {stype} assigned")
            self.set_stype( 'g2',stype,con_all, restype='all')
            self.set_stype('g2',stype, con_av)
            self.set_stype('dist',stype, dist_dict)
            self.set_stype('ts',stype, xData)
        elif stype=='rdf':
            print('rdf assigned')
            self.set_stype( 'rdf', stype, xData)
            self.set_stype( 'rdf', stype, con_all, add_stype=False, restype='all')
            self.set_stype( 'rdf', stype, con_av, add_stype=False)





    def calc_doac(self, stype='i'):
        
        try:
            rdf=getattr(self, f'rdf')
        except AttributeError:
            raise AttributeError("DDrelax object has no attribute 'rdf'")    
        
        
        doac_dict=dict()
        Iab=dict()
        molV=dict()
        for T in self.rdf.keys():
            doac_dict[T]=dict()
            Iab[T]=dict()
            molV[T]=dict()
            for pat1 in self.rdf[T].keys():
                
                if "inter" in pat1:
                    rdf=self.rdf[T][pat1]
                    rdf_weighted=nmrdu.rdf_r4(self.rdfr,rdf)
                    rdf_w_sum=nmrdu.integrate(self.rdfr, rdf_weighted)
                    Iab[T][pat1]=np.mean(rdf_w_sum[-50:])
                    doac= np.power(4*np.pi/(3 *rdf_w_sum[-1] ),1/3)
                    doac_dict[T][pat1]=doac

                if "intra" in pat1:
                    skip=5
                    rdf=self.rdf[T][pat1]
                    rdf_weighted=nmrdu.rdf_r4(self.rdfr[skip:],rdf[skip:])
                    rdf_w_sum=nmrdu.integrate(self.rdfr[skip:], rdf_weighted)
                    # find last y data point that is larger than 0
                    non_zero = [i for i in rdf if i]
                    #find idx of last non zero element
                    idx = np.where(rdf == non_zero[-1])
                    mol_v=self.rdfr[idx]**3
                    molV[T][pat1]=mol_v[0]
                    Iab[T][pat1]=np.mean(rdf_w_sum[-50:])/mol_v
                    doac= np.power(4*np.pi/(3 *rdf_w_sum[-1] ),1/3)
                    doac_dict[T][pat1]=doac
        
        self.set_stype( 'doac', stype, doac_dict)
        self.set_stype( 'Iab', stype, Iab)
        self.set_stype( 'molV', stype, molV)
        
    

        
    def collect_dist_intra(self,path,stype='i',job_dir="dist_intra"):

        dist_intra_dict=dict()
        for T in self.Temp:
            dist_intra_dict[T]=dict()
            for pat1 in self.all_ia:
                if "intra" in pat1:
                    _, nuc, iontype= pat1.split('_')
                    files=nmrdu.listdir_fullpath(f"{path}/{T}K/{job_dir}")
                    file=nmrdu.filter_files(files,pattern=f"*{nuc}*{iontype}*")
                    dist_intra= np.loadtxt(file[0], skiprows=1, dtype=float)
                    dist_intra_dict[T][pat1]=dist_intra
                    
        self.set_stype( 'dist_intra', stype, dist_intra_dict)
 
        

    def calc_spin_density(self,path,job_dir,n_spin_dict,stype='i'):
        
        spinden_dict=dict()
        box_length_dict=dict()
        for T in self.Temp:
            spinden_dict[T]=dict()
            box_length_dict[T]=dict()
            for pat1 in self.all_ia:
                if "inter" in pat1:
                    job_def=np.loadtxt(f"{path}/{T}K/{job_dir}/myjob.def",dtype=str, usecols=1)
                    box_length=float(job_def[-1])
                    box_length_dict[T][pat1]=box_length
                    n_spin= n_spin_dict[pat1]
                    spinden_dict[T][pat1]=  n_spin/ (box_length**3)

        self.set_stype( 'spin_density', stype, spinden_dict)
        self.set_stype( 'box_length', stype, box_length_dict)
                
        
        
    def calc_sdf(self,stype='i'):

        try:
            g2=getattr(self, f"g2{stype}")
            ts=getattr(self, f"ts{stype}")
        except AttributeError:
            raise AttributeError("DDrelax object has no attribute 'g2' or 'ts' ")  
        

        
        sdf=dict()  #collect all sdf
        # calc timestep
        dt=(ts[1]-ts[0]) #*dt_si
        
        for T in g2.keys():
            sdf[T]=dict()
            for pat1 in g2[T].keys():
                #grep correlation function and normalize
                corr=g2[T][pat1]
                norm_factor=max(corr)
                corr_norm=deepcopy(corr)/norm_factor

                
                #cosine transform
                #NOTE: dt_si will be multiplied later -> better interpolation 
                j=dct(corr_norm,norm=None) *dt 

                
                N=len(j)
                sdf[T][pat1]=j[:N//2]

        #calc frequency axis
        freq=fftfreq(N,dt*self.dt_si)[:N//2]
        
        
        self.set_stype( 'sdf',stype,sdf)
        self.set_stype('freq',stype, freq)
        print(f"SDF: stype {stype} assigned")

            
            
    def interpoalte_parameter(self,parameter_string,ia_key,T):
        class_parameter=getattr(self, parameter_string )
        class_parameter_T= nmrdu.transpose_dict(class_parameter)
        parameter= np.array(list(class_parameter_T[ia_key].values()))
        _interpol1=interp1d( self.Temp ,parameter , bounds_error=False) 
        inpol_parameter= _interpol1(T)
        
        return inpol_parameter
            
            
    def calc_R1(self,_interpol1,b_field,interaction,nuc1,nuc2,dist):
        
        gyros={'H': 42.577*1e6, 'F':40.053*1e6, 'P':17.253*1e6} # Hz*T^-1

        pre_factor= self.constants*gyros[nuc1]**2 *gyros[nuc2]**2 *self.multiplicity*dist*self.dt_si*512
        
        con=[]
        for b in b_field:
            
            w1=gyros[nuc1]*b
            w2=gyros[nuc2]*b
            
            if nuc1 == nuc2:
                spec_av= 2/5
                J_w=_interpol1(w1) 
                J_w2=_interpol1(w2*2)  
                R= spec_av*pre_factor*(J_w+ 4*J_w2)
            else:
                spec_av= 1/15
                J_w=_interpol1( np.abs(w1-w2) )
                J_w2=_interpol1(w1)   
                J_w3=_interpol1(w1+w2)   
                R= spec_av*pre_factor*(J_w+3*J_w2+6*J_w3)
                
            con.append(R)
        
        return con 
    
    
    def calc_Dprofile(self, restype, relevant_ia, freq_list,  parameter_stype='i', stype='i'):
        
        # set freqlist for calculation
        self.set_stype( 'v', stype, freq_list*1e-6,  restype=restype)

        # spectral density
        sdf_dict=getattr(self, f"sdf{stype}")
        freq=getattr(self, f"freq{stype}")

        #prefactor
        doac_dict=getattr(self, f"doac{parameter_stype}")
        spin_density_dict=getattr(self, f"spin_density{parameter_stype}")
        dist_intra_dict=getattr(self, f"dist_intra{parameter_stype}")
        dist_dict=getattr(self, f"dist{parameter_stype}")


        b_field=freq_list/(42.577*1e6)
        R1_dict=dict()
        for T in sdf_dict.keys():
            R1_dict[T]=dict()

            if T in self.Temp: 

                for ia_nuclei in relevant_ia:
                    ia, nuclei,iontype=ia_nuclei.split("_")
                    
                    if ( hasattr(self,f'doac{parameter_stype}') and hasattr(self,f'spin_density{parameter_stype}') ) and ia == "inter":
                        doac=doac_dict[T][ia_nuclei]
                        spin_density=spin_density_dict[T][ia_nuclei]
                        dist= spin_density * 4*np.pi / (doac**3) *np.reciprocal(self.dist_si)**6
                        
                    elif ( hasattr(self,f'dist_intra{parameter_stype}')  )and ia == "intra":  
                        doac_intra= dist_intra_dict[T][ia_nuclei] 
                        dist= 1/(doac_intra*self.dist_si)**6
                        
                    else:
                        dist=dist_dict[T][ia_nuclei] 
                        dist*=np.reciprocal(self.dist_si)**6 

                    
                    sdf=sdf_dict[T][ia_nuclei]
                    _interpol1=interp1d(freq, sdf) 
                    R1=self.calc_R1(_interpol1,b_field,ia,nuclei[0],nuclei[1],dist)
                    R1_dict[T][ia_nuclei]=R1
                

            #if master curve is used -> interpolate to all temperatures
            else:

                for ia_nuclei in relevant_ia:
                    ia, nuclei,iontype=ia_nuclei.split("_")
                            
                    if ( hasattr(self,f'doac{parameter_stype}') and hasattr(self,f'spin_density{parameter_stype}') ) and ia == "inter":
                        doac= self.interpoalte_parameter(f'doac{parameter_stype}',ia_nuclei,T)
                        spin_density=self.interpoalte_parameter(f'spin_density{parameter_stype}',ia_nuclei,T)
                        dist= spin_density * 4*np.pi / (doac**3) *np.reciprocal(self.dist_si)**6
                        
                    elif ( hasattr(self,f'dist_intra{parameter_stype}')  )and ia == "intra":  
                        doac_intra= self.interpoalte_parameter(f'dist_intra{parameter_stype}',ia_nuclei,T)
                        dist= 1/(doac_intra*self.dist_si)**6
                        
                    else:
                        dist=self.interpoalte_parameter(f'dist{parameter_stype}', ia_nuclei,T) *np.reciprocal(self.dist_si)**6 

                    
                    sdf=sdf_dict[T][ia_nuclei]
                    _interpol1=interp1d(freq, sdf) 
                    R1=self.calc_R1(_interpol1,b_field,ia,nuclei[0],nuclei[1],dist)
                    R1_dict[T][ia_nuclei]=R1


        self.set_stype('R1', stype, R1_dict, restype=restype)
        print(f"R1: stype {stype} and restype {restype} assigned")
        R1_array, _ =self.check_for_att( 'R1') 
        
        return R1_array
    


    
    def Tsuperposition(self, file_path, function, T_shift, skip, new_Temp=None):

        xData=getattr(self, function[0])
        yDict=getattr(self, function[1])

        vft_params=np.loadtxt(file_path, dtype=float, skiprows=1)
        Tmax=T_shift
        
        # Tmin=min(Temp)
        eta_max=nmrdu.VFT(Tmax,*vft_params)
        yDict=nmrdu.transpose_dict(yDict)
        # xData_trim=xData

        master=dict()                              
        for nuclei in yDict.keys():
            cum_sum=[]
            Temp=list(yDict[nuclei].keys())[skip:]
            for T in Temp:
                eta=nmrdu.VFT(T,*vft_params)
                
                xData_w=xData *eta_max/eta
                # if T==Temp[0]:
                #     xData_trim=xData_w
                    
                yData=yDict[nuclei][T]
                yData=yData/max(yData)
                
                extrapol=np.mean(yDict[nuclei][Tmax][-200:])
                
                _interpol1=interp1d(xData_w ,yData ,fill_value=(1,extrapol), bounds_error=False) 
                yData_w=_interpol1(xData)
                cum_sum.append(yData_w)         
                            
            master_curve=np.mean(cum_sum, axis=0)    
            master[nuclei]=master_curve
        
        if new_Temp is not None:
            g2_master=dict()  
            for T in new_Temp:
                eta=nmrdu.VFT(T,*vft_params)
                g2_master[T]=dict()
                for nuclei in yDict.keys():
                    yData=master[nuclei]
                    xData_w=xData *eta/eta_max
                    _interpol1=interp1d(xData_w ,yData ) #,fill_value="extrapolate" ) 
                    yData_w=_interpol1(xData)
                    g2_master[T][nuclei]=yData_w

            self.set_stype('ts', stype='m', value=xData)
            self.set_stype('g2', stype='m', value=g2_master)
            print(f"stype m assigned")

                
    


    def plot_Tsuperposition(self, ax, file_path, function, nuclei, T_shift=None, shift=True):
        
        # con_x,con_types=self.check_for_att(function[0] )
        # con_y,con_types=self.check_for_att(function[1], res=res)
        # for (xData,yDict,stype) in zip(con_x, con_y, con_types):
        xData=getattr(self, function[0])
        yDict=getattr(self, function[1])

        vft_params=np.loadtxt(file_path, dtype=float, skiprows=1)
        if T_shift is not None:
            Tref=T_shift
        else:
            Tref=max(yDict.keys())
            
        eta_max=nmrdu.VFT(Tref,*vft_params)
        
        for T in yDict.keys():
            eta=nmrdu.VFT(T,*vft_params)
            
            yData=yDict[T][nuclei]
            
            if "g2" in function[1] and not shift: 
                yData=yData
                xData_w=xData 
            elif "g2" in function[1] and shift:
                yData=yData/max(yData)
                xData_w=xData *eta_max/eta   
            elif "sdf" in function[1]:
                yData=yData #/max(yData)
                xData_w=xData #* eta/eta_max
                
            elif "R1" in function[1]:
                yData=yData
                
            ax.plot(xData_w,yData, label=f'{T}K')
        
        return  ax



        



