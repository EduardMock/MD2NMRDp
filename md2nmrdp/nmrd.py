
import numpy as np
from scipy.interpolate import interp1d
from scipy.fft import fftfreq,dct
from scipy.optimize import curve_fit



from copy import deepcopy
import matplotlib.pyplot as plt



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

class DDrelax():
    
    def __init__(self, system, Temp, spin=1/2,dt_si=1e-12,dist_si=1e-10):
        
        self.system=system
        self.Temp=Temp
        
        self.spin= spin
        self.dt_si=dt_si   #unit of the timestep 
        self.dist_si=dist_si   #unit of the distance/postiton 
        
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
        # print(files)
        if len(files)==1:
            ts,corr=np.loadtxt(files[0] ,unpack=True)
            return ts, corr
        else:
            cum_sum=[]
            for f in files:
                ts,corr=np.loadtxt(f ,unpack=True)
                cum_sum.append(corr)
            return ts, np.mean(cum_sum,axis=0)
    
    
    def check_for_att(self, func, res=''):
        
        collect=[]
        sim_types=[]
        for sim_type in ['s','i','l']:
            try:
                collect.append(getattr(self, func+sim_type+res))
                sim_types.append(sim_type)
            except:
                pass

        return collect, sim_types
    


    def convert_files(self, filedict, stype):
        con=dict()
        con_av=dict()
        for T in filedict.keys():
            print(T)
            con[T]=dict()
            con_av[T]=dict()
            
            for pat1 in filedict[T].keys() :
                con[T][pat1]=dict()
                #check if tupel or dict
                if isinstance( filedict[T][pat1] ,dict): 
                    # sum up contributions of different atomgroups 
                    at_cum_sum=0
                    for pat2 in filedict[T][pat1].keys():
                        #convet and average intermolecular contribution over residues
                        xdata, ref_cum_sum = self.convert_average(filedict[T][pat1][pat2])
                        con[T][pat1][pat2]= ref_cum_sum
                        at_cum_sum+=ref_cum_sum
                    con_av[T][pat1]=at_cum_sum
                else: 
                    xdata, ref_cum_sum = self.convert_average(filedict[T][pat1])
                    con_av[T][pat1]= ref_cum_sum


        if stype=='short':
            print('short assigned')
            self.tss=xdata
            self.g2s=con
            self.g2s_av=con_av
        elif stype=='inter':
            print('inter assigned')
            self.tsi=xdata
            self.g2i=con
            self.g2i_av=con_av
        elif stype=='long':
            print('long assigned')
            self.tsl=xdata
            self.g2l=con
            self.g2l_av=con_av
        elif stype=='rdf':
            print('rdf assigned')
            self.rdfr=xdata
            self.rdf=con




    def calc_doac(self):
        
        try:
            getattr(self, 'rdf')
        except AttributeError:
            raise AttributeError("DDrelax object has no attribute 'rdf'")    
        
        doac_dict=dict()
        for T in self.rdf.keys():
            doac_dict[T]=dict()
            for pat1 in self.rdf[T].keys():
                rdf=self.rdf[T][pat1]
                rdf_weighted=rdf_r4(self.rdfr,rdf)
                rdf_w_sum=integrate(self.rdfr, rdf_weighted)
                doac= np.power(4*np.pi/3 *rdf_w_sum,1/3)
                doac_dict[T][pat1]=doac
        
        self.doac=doac_dict
    

        
    def calc_dist_intra(self,system_path,job_dir):
        
        try:
            getattr(self, 'rdf')
        except AttributeError:
            raise AttributeError("DDrelax object has no attribute 'rdf'")    
        
        dist_intra_dict=dict()
        # for T in self.rdf.keys():
        #     dist_intra_dict[T]=dict()
        #     for pat1 in self.rdf[T].keys():
                # ff_path=f"{system_path}/forcefield/"
                # dcd_path=f"{system_path}/{T}K/{job_dir}/myjob.def"
                # dcd_pattern='*.dcd'                           
                # psf_pattern='*.psf' 


        
        self.dist_intra=dist_intra_dict    
        

    def calc_spin_density(self,system_path,job_dir,n_spin_dict):

        try:
            getattr(self, 'rdf')
        except AttributeError:
            raise AttributeError("DDrelax object has no attribute 'rdf'") 
        
        
        spind_dict=dict()
        for T in self.rdf.keys():
            for pat1 in self.rdf[T].keys():
                job_def=np.loadtxt(f"{system_path}/{T}K/{job_dir}/myjob.def",dtype=str,use_cols=1)
                box_length=float(job_def[-1])
                n_spin= n_spin_dict[pat1]
                spind_dict[T][pat1]=  n_spin*np.reciprocal(box_length)**3
                
        self.spind=spind_dict
                
        
        
    def calc_sdf(self):
        
        con_g2,con_types=self.check_for_att('g2', res='_av')
        con_t,con_types=self.check_for_att('ts' )

        for (ts,g2,stype) in zip(con_t, con_g2, con_types):
            sdf=dict()  #collect all sdf
            
            # calc timestep
            dt=(ts[1]-ts[0]) #*dt_si
            # print(dt)
            
            for T in g2.keys():
                sdf[T]=dict()
                for pat1 in g2[T].keys():
                    #grep correlation function and normalize
                    corr=g2[T][pat1]
                    norm_factor=max(corr)
                    corr_norm=deepcopy(corr)/norm_factor
                    # print(corr_norm[0])
                    
                    #cosine transform
                    #NOTE: dt_si will be multiplied later -> better interpolation 
                    j=dct(corr_norm,norm=None) *dt 

                    
                    N=len(j)
                    sdf[T][pat1]=j[:N//2]

            #calc frequency axis
            freq=fftfreq(N,dt*self.dt_si)[:N//2]
            
            if "s" == stype:
                print('short assigned')
                self.sdfs_av=sdf
                self.freqs=freq
            elif "i"  == stype:
                print('inter assigned')
                self.sdfi_av=sdf
                self.freqi=freq
            elif "l"  == stype:
                print('long assigned')
                self.sdfl_av=sdf
                self.freql=freq

            

            
    def calc_R1(self,_interpol1,b_field,interaction,nuc1,nuc2,dist):
        
        gyros={'H': 42.577*1e6, 'F':40.053*1e6, 'P':17.253*1e6} # Hz*T^-1
        fac1=2/15 #factor 2 is already involed in dct?
        fac2=2/5  #factor 2 is already involed in dct?

        spec_av_intra={ 'FH': fac1,'HF': fac1, 'HP':fac1, 'PH':fac1, 'FF': fac2, 'HH':fac2, 'PP': fac1} 
        spec_av_inter={ 'FH': fac1,'HF': fac1, 'FF': fac2, 'HH':fac2} 
    
    
        nuclei=nuc1+nuc2
        if interaction == 'intra':
            spec_av= spec_av_intra[nuclei]
        elif interaction == 'inter':
            spec_av= spec_av_inter[nuclei]
        
        
        pre_factor= spec_av*self.constants*gyros[nuc1]**2 *gyros[nuc2]**2 *self.multiplicity*dist
        con=[]
        for b in b_field:
            
            w1=gyros[nuc1]*b
            w2=gyros[nuc2]*b
            
            if spec_av == fac2: #1/5
                J_w=_interpol1(w1) 
                J_w2=_interpol1(w2*2)  
                R= pre_factor*(J_w+ 4*J_w2)

            if spec_av == fac1:
                J_w=_interpol1( np.abs(w1-w2) )
                J_w2=_interpol1(w1)   
                J_w3=_interpol1(w1+w2)   
                R= pre_factor*(J_w+3*J_w2+6*J_w3)
                
            con.append(R)
        
        return con 
    
    
    def calc_Dprofile(self,relevant_ia,freq_list):

        con_g2,con_types=self.check_for_att( 'g2', res='_av')
        con_sdf,con_types=self.check_for_att( 'sdf', res='_av')
        con_freq,con_types=self.check_for_att( 'freq' )
        
        dist_dict=dict()
        for (g2, sdf,freq, stype) in zip(con_g2, con_sdf, con_freq,con_types):
            b_field=freq_list/(42.577*1e6)
            R1=dict()
            for T in sdf.keys():
                R1[T]=dict()
                dist_dict[T]=dict()
                for ia_nuclei in relevant_ia:
                    ia, nuclei=ia_nuclei.split("_")
                    
                    if ( hasattr(self,'doac') and hasattr(self,'spind') )and ia == "inter":
                        doac=self.doac[T][ia_nuclei]
                        spin_density=self.spind[T][ia_nuclei]
                        dist= 4*np.pi * np.reciprocal(doac)**3*spin_density *np.reciprocal(self.dist_si)**6 
                        
                    elif ( hasattr(self,'dist_intra')  )and ia == "intra":  
                        doac_intra= self.dist_intra[T][ia_nuclei] 
                        dist= np.reciprocal(doac_intra*self.dist_si)**6
                        
                    else:
                        dist=g2[T][ia_nuclei][0]*np.reciprocal(self.dist_si)**6 
                        dist_dict[T][ia_nuclei]=dist
                        print(f"{ia} {nuclei} r^-6 = {dist:.3e}")
                    
                    sdf_sel=sdf[T][ia_nuclei]
                    N=len(sdf_sel)
                    _interpol1=interp1d(freq, sdf_sel) #, kind='nearest', fill_value="extrapolate")
                    R1_cont=self.calc_R1(_interpol1,b_field,ia,nuclei[0],nuclei[1],dist)
                    
                    R1[T][ia_nuclei]=R1_cont


            if "s" in stype:
                self.R1s=R1
            elif "i" in stype:
                self.R1i=R1
            elif "l" in stype:
                self.R1l=R1
        
        self.dist_dict.update(dist_dict)
        R1_array, _ =self.check_for_att( 'R1') 
        return R1_array
    
    

    def plot_Tsuperposition(self, function, file_path, relevant_nuclei):
        
        con_x,con_types=self.check_for_att(function[0] )
        con_y,con_types=self.check_for_att(function[1], res='_av')


        for (xData,yDict,stype) in zip(con_x, con_y, con_types):
            
            fig, ax=plt.subplots()
            vft_params=np.loadtxt(file_path, dtype=float, skiprows=1)
            eta_max=VFT(max(self.Temp),*vft_params)
            for T in yDict.keys():
                eta=VFT(T,*vft_params)
                
                for ia_nuclei in relevant_nuclei:
                    yData=yDict[T][ia_nuclei]
                    if function == "g2":
                        yData=yData/max(yData)
                        
                    xData_w=xData*eta/eta_max
                    ax.plot(xData_w,yData, label=f'{ia_nuclei} {T}K')
        
        return fig, ax



        



