#==============================================================================
# Imports
#==============================================================================
from math import pi
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

#Import custom modules
from physics.constants import q, hbar, c
from physics.Greek import textmu
from Material_Properties import MaterialProperties,ncrit,ncritmod,MeltThreshold
from Modified_Keldysh import PulseIntegration, FindIntensityThreshold
from Create_Plots import Plot_time_dependent_quantity, EnergyDistributionHistogram, EnergyDistributionCDF


class sim:
    def __init__(self,*,mat,wl,inten,pol,cdir,Tpulse,aoi,
                 time_window=4.0,timesteps=801,multiband=True,VB_depletion=True,Vinogradov=False,Reflection=True,Direct=True,Avalanche=False,VaryRefractiveIndex=True):
        
        MatProp = pd.read_csv('Material_Properties/{Mat}_Properties.csv'.format(Mat=mat))
        
        self.params={
            'Material':mat,
            'MatProp':MatProp,
            'Wavelength':np.array(wl),
            'Intensity':np.array(inten),
            'Polarization':pol,
            'Direction':cdir,
            'PulseDuration':Tpulse,
            'AOI':aoi,
            'intrange':time_window,
            'NumTimesteps':timesteps,
            'Multiband':multiband,
            'VB_depletion':VB_depletion,
            'Vinogradov':Vinogradov,
            'Reflection':Reflection,
            'Direct':Direct,
            'Avalanche':Avalanche,
            'VaryRefractiveIndex':VaryRefractiveIndex
                }
        return
    def run(self):
        """
        Run Modified Keldysh simulation with parameters stored in 'self.params'
        """
        p=self.params
        # First, we reshape quanities for vectorization
        NumWavelengths = p['Wavelength'].size
        NumIntensities = p['Intensity'].size
        
        Wavelength = p['Wavelength'].copy()
        Intensity = p['Intensity'].copy()
        PulseDuration = np.zeros(NumWavelengths) #Pulse duration in fs
        PulseDuration[PulseDuration==0] = p['PulseDuration']
        
        Wavelength = Wavelength.reshape(Wavelength.size,1,1)
        Intensity = Intensity.reshape(1,NumIntensities,1)
        PulseDuration = PulseDuration.reshape(NumWavelengths,1,1)
        

              
        #Print selected inputs
        print('Min wavelength =', MinWavelength, 'um \n',
              'Max wavelength =', MaxWavelength, 'um \n',
              'Wavelength stepsize =',Wavelength_Stepsize, 'um \n',
              'Min intensity =',MinIntensity, 'TW/cm^2 \n',
              'Max intensity =',MaxIntensity, 'TW/cm^2 \n',
              'Num intensities =',NumIntensities, '\n',
              'Polarization =',Polarization, '\n',
              'AOI =', theta, '\n',
              'Crystal direction =',Direction, '\n',
              'Timesteps =',NumTimesteps, '\n',
              'Multiband =',Multiband, '\n',
              'VB depletion =',VB_depletion, '\n',
              'Vinogradov =',Vinogradov, '\n',
              'Reflection =',Reflection, '\n',
              'Direct =', Direct, '\n',
              'Avalanche =',Avalanche, '\n',
              'Vary refractive index =', VaryRefractiveIndex)
        
                #Convert to base units
        Intensity *= 10**12*10**4
        Wavelength *= 10**-6
        PulseDuration *= 10**-15
        theta = p['AOI']*pi/180
        
        runsim(MatProp=p['MatProp'],
               theta=theta,
               Polariation=p['Polarization'],
               Direction=p['Direction'],
               PulseDuration=PulseDuration,
               Wavelength=Wavelength,
               Intensity=Intensity,
               intrange=p['intrange'],
               NumTimesteps=p['NumTimesteps'],
               )

def runsim(MatProp,theta,Polarization,Direction,PulseDuration,
                          Wavelength,Intensity,intrange,NumTimesteps,Instance,
                          VB_depletion,Multiband,Vinogradov,Reflection,Direct,
                          VaryRefractiveIndex,Avalanche):
    
    output = PulseIntegration(MatProp,theta,Polarization,Direction,PulseDuration,
                              Wavelength,Intensity,intrange,NumTimesteps,Instance,
                              VB_depletion,Multiband,Vinogradov,Reflection,Direct,
                              VaryRefractiveIndex,Avalanche)
    
    Time = output['Time']
    Intensity_vs_t = output['Intensity vs t']
    Efield = output['Electric field']
    EffBandgap = output['Effective bandgap']
    PopFactor = output['Population factor']
    SlowlyVaryingAmplitude = output['Slowly varying amplitude']
    WPI = output['Photoionization rate']
    WAv = output['Avalanche ionization rate']
    nCB_PI = output['Photoionization contribution']
    nCB_Av = output['Avalanche contribution']
    nCB = output['Conduction band electron density']
    nCB_contrib = output['CB electron density contributions at each timestep']
    n0 = output['Total electron density']
    AbsorptionRate = output['CB absorption rate']
    AbsorptionRatePerElectron = output['CB absorption rate per electron']
    ArrivalEnergy = output['CB arrival energy']
    EnergyDistribution = output['CB energy distribution']
    nreal = output['Real refractive index']
    nimag = output['Imaginary refractive index']
    Tdrude = output['Avalanche Collision Time']
    Tei = output['e-i Collision Time']
    Tee = output['e-e Collision Time']
    Tf = output['Transmission vs Time']
    
    #Final electron density
    nCB_PI_total = np.sum(nCB_PI,axis=3)
    nCB_Av_total = np.sum(nCB_Av,axis=3)
    nCB_total = np.sum(nCB,axis=3) #Sum of all valence band contributions
    nCB_final = nCB_total[-1] #Final total electron density 
    return output
#%%
#==============================================================================
# Create folder to save files in
#==============================================================================
#Automatic filenaming
filename_p1 = 'Multiband' if Multiband else 'Singleband'
filename_p2 = 'VBdepletion' if VB_depletion else 'NoVBdepletion'
filename_p3 = 'Vinogradov' if Vinogradov else 'NoVinogradov'
filename_p4 = 'Reflection' if Reflection else 'NoReflection'
filename_p5 = 'DirectBG' if Direct else 'IndirectBG'
filename_p6 = 'Avalanche' if Avalanche else 'NoAvalanche'
filename_p7 = 'VaryingRefractiveIndex' if VaryRefractiveIndex else 'ConstantRefractiveIndex'
filename_p8 = 'Melt' if Criterion=='Melt' else 'ncrit' if Criterion=='ncrit' else 'ncritmod'

FolderName = '{Mat}_{p1}_{p2}_{p3}_{p4}_{p5}_{p6}_{p7}_{p8}_{pol}pol'.format(Mat=Material,
           p1=filename_p1,p2=filename_p2,p3=filename_p3,p4=filename_p4,
           p5=filename_p5,p6=filename_p6,p7=filename_p7,p8=filename_p8,
           pol=Polarization)

Directory = './Output/' + FolderName + '/'

filename = FolderName + '_{instance}of{total}'.format(instance=Instance,total=NumInstances)

#Create directory
if SavePlots or SaveData or SaveThreshold and FindThreshold:
    
    #Check if the directory already exists
    if not os.path.exists(Directory):
        os.makedirs(Directory)


#%%
#==============================================================================
# Find threshold intensity
#==============================================================================
#Find threshold
if FindThreshold:
    if Criterion=='ncrit':
        nCB_thresh = ncrit(np.squeeze(Wavelength,axis=2))
    elif Criterion=='ncritmod':
        nCB_thresh = ncritmod(np.squeeze(Wavelength,axis=2),
                              MatPropDict['CB effective mass'][0],
                              np.squeeze(MatPropDict['Refractive index'],axis=2)**2)
    elif Criterion=='Melt':
        nCB_thresh = MeltThreshold(MatProp.loc[0]['NumBonds'],MatProp.loc[0]['LatticeConstant']*10**-10,ThresholdPercentage)
    else:
        ValueError
    
    Intensity_thresh, Thresh_Found = FindIntensityThreshold(Intensity,nCB_final,nCB_thresh)
    #Intensity_thresh *= np.sqrt(pi)/(2*np.sqrt(np.log(2))) #CHANGE THIS BACK
    

 