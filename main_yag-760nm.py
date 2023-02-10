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
from physics.constants import q, hbar, c,me
from physics.Greek import textmu
from Material_Properties import MaterialProperties,ncrit,ncritmod,MeltThreshold
from Modified_Keldysh import PulseIntegration, FindIntensityThreshold
from Create_Plots import Plot_time_dependent_quantity, EnergyDistributionHistogram, EnergyDistributionCDF

#%%
#==============================================================================
# Plot settings
#==============================================================================
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.family'] = 'STIXGeneral'
figsize = (12,8)
Figure_filetype = '.pdf'

#%%
#==============================================================================
# Inputs
#==============================================================================
#Simulation parameters
Instance = 1 #The instance number for this run (for running multiple instances of the code)
NumInstances = 1 #Number of instances to be ran
intrange = 4.0 #integration range in units of the pulse duration
NumTimesteps = 801 #Number of timesteps
Multiband = True #Use multiple valence bands?
VB_depletion = True #Include valence band depletion?
Vinogradov = False #Include Vinogradov absorption?
Reflection = True #Include reflection?
Direct = True #Use direct bandgap?
Avalanche = True #Include avalanche ionization (Drude model)?
VaryRefractiveIndex = True #Vary refractive index?
Criterion = 'ncritmod' #Criterion to use for ablation; either 'ncrit', 'ncritmod', or 'Melt'
ThresholdPercentage = 10 #Percent ionization used for ultrafast melting

#What to output
FindThreshold = True
SaveThreshold = False
Plot_all = False #For plotting everything regardless of what's below
Plot_threshold_vs_wavelength = True
Plot_Efield_vs_time = True
Plot_nCB_vs_time = True
Plot_nCB_final_vs_intensity = True
Plot_photoionization_vs_time = True
Plot_AbsorptionRate_vs_time = False
Plot_ArrivalEnergy_vs_time = False
Plot_EnergyDistribution = False
Plot_bandgap_vs_time = False
Plot_AbsorbedPhotons_vs_time = False
Plot_SlowlyVaryingAmplitude_vs_time = False
Plot_PopFactor_vs_time = False
Plot_refractive_index_vs_time = False
Plot_reflectivity_vs_time = False
Plot_VinogradovRate_vs_energy = False
Plot_collisionrate_vs_time = False
SavePlots = False #Save selected plots
SaveData = False #Save data of selected plots (not currently implemented)

#Material and laser parameters
Material = 'YAG'
MatProp = pd.read_csv('Material_Properties/{Mat}_Properties.csv'.format(Mat=Material))
Direction = '[111]' #Crystal direction along which the polarization is oriented ([110] for s, [111] for p)
theta = 45 #AOI in degrees
Polarization = 'p' #Polarization

Wavelength_Stepsize = 0.01
MinWavelength = .6 #Min wavelength to calculate
MaxWavelength = .9 #Max wavelength to calculate
Wavelength = np.array([.6,.76,.9])
#np.arange(MinWavelength,MaxWavelength+Wavelength_Stepsize,Wavelength_Stepsize)
Wavelength = Wavelength[round(Wavelength.size/NumInstances * (Instance-1)):round(Wavelength.size/NumInstances * Instance)]
#Wavelength = np.array([2.75,3.15,3.75,4.15]) #For specific wavelengths
#Wavelength = np.array([4.15]) #For specific wavelengths
NumWavelengths = Wavelength.size

MinIntensity = 40 #Miniumum peak laser intensity in TW/cm^2
MaxIntensity = 940 #Maximum peak laser intensity in TW/cm^2
DesiredStepSize = 0.1
#DesiredStepSize = 0.1
NumIntensities = 500 #int(np.round((MaxIntensity-MinIntensity)/DesiredStepSize))
#NumIntensities = 150 #Number of intensities
Intensity = np.linspace(MinIntensity,MaxIntensity,NumIntensities)

PulseDuration = np.zeros(NumWavelengths) #Pulse duration in fs
PulseDuration[PulseDuration==0] = 9. #Set to 100 fs for all wavelengths

#Reshape arrays for vectorization
Wavelength = Wavelength.reshape(Wavelength.size,1,1)
Intensity = Intensity.reshape(1,NumIntensities,1)
PulseDuration = PulseDuration.reshape(NumWavelengths,1,1)

#Convert to base units
Intensity *= 10**12*10**4
Wavelength *= 10**-6
PulseDuration *= 10**-15
theta *= pi/180
intrange *= np.max(PulseDuration) #input intrange is in units of the pulse duration
time = np.linspace(-intrange/2,intrange/2,NumTimesteps)

#Define material properties dictionary
MatPropDict = MaterialProperties(MatProp,Direction,Wavelength,Multiband)

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
      'Vary refractive index =', VaryRefractiveIndex, '\n',
      'Criterion =', Criterion)

#%%
#==============================================================================
# Integrate over pulse
#==============================================================================
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
    

#Save values
if SaveThreshold and FindThreshold:
    
    Thresh_vs_wavelength = np.transpose(np.vstack((np.squeeze(Wavelength)*10**6,
                                               Intensity_thresh*10**-16)))
    
    np.savetxt(Directory+filename+'.csv',Thresh_vs_wavelength,delimiter=',')
    
#Plot intensity threshold vs wavelength
if Plot_threshold_vs_wavelength or Plot_all and FindThreshold:
    fig,ax = plt.subplots()
    ax.plot(np.squeeze(Wavelength)*10**6,Intensity_thresh*10**-16,
             linestyle='-',marker='o',color='b')
    ax.set_xlabel('Wavelength ({mu}m)'.format(mu=textmu))
    ax.set_ylabel('Intensity ($\mathrm{TW/cm^2}$)',color='b')
    ax.tick_params('y', colors='b')
    plt.tight_layout()
    
    #plt.savefig(Directory + 'Thresh-vs-wavelength-Plot' + Figure_filetype)
    
    if np.any(np.logical_not(Thresh_Found)):
        try:
            print('WARNING: Threshold not found for wavelengths',
                  np.squeeze(Wavelength)[Thresh_Found==False]*10**6,'um')
        except: #For single wavelengths
            print('WARNING: Threshold not found for wavelengths',
                  Wavelength[0,0,0]*10**6,'um')

#%%
#==============================================================================
# Select wavelength and intensity for outputs
#==============================================================================
SelectWavelength = 3.95 #Desired wavelength to plot (um)

#Convert units
SelectWavelength *= 10**-6

#Find index at which the array values are closest to the desired value
SelectWavelengthIndex = min(range(Wavelength.size),key=lambda i: abs(Wavelength[i,0,0]-SelectWavelength))

#Closest matching values
SelectWavelength = Wavelength[SelectWavelengthIndex,0,0]

#Do the same for intensity
SelectIntensity = Intensity_thresh[SelectWavelengthIndex] #Intensity_thresh[SelectWavelengthIndex] if FindThreshold else Intensity[0,0,0] #TW/cm^2
SelectIntensityIndex = min(range(Intensity.size),key=lambda i: abs(Intensity[0,i,0]-SelectIntensity))
SelectIntensity = Intensity[0,SelectIntensityIndex,0]


#%%
#==============================================================================
# Output final electron density
#==============================================================================
nCB_final_select = nCB_final[SelectWavelengthIndex,SelectIntensityIndex]

print('For {wavelength:.0f} nm, {intensity:.4f} TW/cm^2:'
      .format(wavelength=SelectWavelength*10**9, intensity=SelectIntensity*10**-16))

#Final electron density overall
print('Final overall conduction band electron density:'
      .format(wavelength=SelectWavelength*10**9, intensity=SelectIntensity*10**-16),
      nCB_final_select*10**-6,'cm^-3, ',nCB_final_select/n0*100,'%')


#%%
#==============================================================================
# Create plots of time-dependent quantities
#==============================================================================

#%%
#Photoionization rate vs. time
if Plot_collisionrate_vs_time or Plot_all:
    
    Plot_time_dependent_quantity(figsize,Tdrude,'$t_{Drude} \: (fs)$',
                                 Directory+'DrudeCollisionRate-vs-time-Plot',
                                 Figure_filetype,10**15,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=True,SavePlot=SavePlots)

#Electric field strength vs. time
if Plot_Efield_vs_time or Plot_all:
    
    fig,ax = plt.subplots(figsize=figsize)
    
    ax.plot(Time.squeeze()*10**15,
            Efield[:,SelectWavelengthIndex,SelectIntensityIndex],marker='.',linestyle='None')
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Electric field strength (V/m)')
    
    if SavePlots:
        plt.savefig(Directory + 'ElectricField-vs-time-Plot' + Figure_filetype)
        
#%%
#Electron density vs. time
if Plot_nCB_vs_time or Plot_all:

    Plot_time_dependent_quantity(figsize,nCB,'Electron density ($\mathrm{cm^{-3}}$)',
                                 Directory+'ElectronDensity-vs-time-Plot',
                                 Figure_filetype,10**-6,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)
    plt.semilogy()


#%%
#Photoionization rate vs. time
if Plot_photoionization_vs_time or Plot_all:
    
    Plot_time_dependent_quantity(figsize,WPI,'$W_{PI} \: (\mathrm{1/(fs \cdot cm^3)})$',
                                 Directory+'PhotoionizationRate-vs-time-Plot',
                                 Figure_filetype,10**-21,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)
    
        
#%%
#Vinogradov absorption rate vs. time
if Plot_AbsorptionRate_vs_time and Vinogradov or Plot_all and Vinogradov:
    
    Plot_time_dependent_quantity(figsize,AbsorptionRate,'Absorption rate $(\mathrm{J/(cm^3 \cdot fs)})$',
                                 Directory+'VinogradovRate-vs-time-Plot',
                                 Figure_filetype,10**-21,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)
        
#%%
#Arrival energy vs. time
if Plot_ArrivalEnergy_vs_time and Vinogradov or Plot_all and Vinogradov:
    
    Plot_time_dependent_quantity(figsize,ArrivalEnergy,'Arrival energy (eV)',
                                 Directory+'ArrivalEnergy-vs-time-Plot',
                                 Figure_filetype,1/q,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)

#%%
#Effective bandgap vs. time
if Plot_bandgap_vs_time or Plot_all:
    
    Plot_time_dependent_quantity(figsize,EffBandgap,'Effective bandgap (eV)',
                                 Directory+'EffectiveBandgap-vs-time-Plot',
                                 Figure_filetype,1/q,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)

#%%
#Number of simultaneously absorbed photons vs. time     
if Plot_AbsorbedPhotons_vs_time or Plot_all:
    
    Ephoton = 2*pi*hbar*c/SelectWavelength
    Plot_time_dependent_quantity(figsize,np.ceil(EffBandgap/Ephoton),'Number of photons',
                                 Directory+'AbsorbedPhotons-vs-time-Plot',
                                 Figure_filetype,1,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)

#%%
#Population factor vs. time
if Plot_PopFactor_vs_time or Plot_all:
    
    Plot_time_dependent_quantity(figsize,PopFactor,'Population factor',
                                 Directory+'PopulationFactor-vs-time-Plot',
                                 Figure_filetype,1,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)
        
#%%
#Slowly varying amplitude vs. time
if Plot_SlowlyVaryingAmplitude_vs_time or Plot_all:
    
    Plot_time_dependent_quantity(figsize,SlowlyVaryingAmplitude,'Slowly varying amplitude $Q$',
                                 Directory+'SlowlyVaryingAmplitude-vs-time-Plot',
                                 Figure_filetype,1,MatPropDict,Time,
                                 SelectWavelengthIndex,SelectIntensityIndex,
                                 Total=False,Semilog=False,SavePlot=SavePlots)
        
#%%
#Refractive index vs time
if Plot_refractive_index_vs_time or Plot_all:
    fig,ax = plt.subplots(figsize=figsize)
    ax.plot(Time[:,SelectWavelengthIndex,SelectIntensityIndex,0]*10**15,
            nreal[:,SelectWavelengthIndex,SelectIntensityIndex], 
            label='$\mathrm{Re}[n]$',linestyle='None',marker='.')
#    ax.plot(Time[:,SelectWavelengthIndex,SelectIntensityIndex,0]*10**15,
#            nimag[:,SelectWavelengthIndex,SelectIntensityIndex],
#            label='$\mathrm{Im}[n]$',linestyle='None',marker='.')
    plt.legend()
    if SavePlots:
        plt.savefig(Directory + 'RefractiveIndex-vs-time-Plot' + Figure_filetype)


#%%
#==============================================================================
# Create plots of quantities at specific times
#==============================================================================
SelectTimes = [0,intrange/2] #Select times to plot

#Plot CB energy distribution at peak and end of pulse
if Plot_EnergyDistribution and Vinogradov or Plot_all and Vinogradov:
    
    fig,ax = plt.subplots(nrows=1,ncols=len(SelectTimes),
                          figsize=(8*len(SelectTimes),6))
    
    fig2,ax2 = plt.subplots(nrows=1,ncols=len(SelectTimes),
                          figsize=(8*len(SelectTimes),6))
    
    Numbins=30
    for i,t in enumerate(SelectTimes):
        EnergyDistributionHistogram(ax[i],EnergyDistribution,nCB_contrib,Time, 
                                    SelectWavelengthIndex, 
                                    SelectIntensityIndex,t,
                                    NumBins=Numbins, Total=True)
        EnergyDistributionHistogram(ax[i],EnergyDistribution,nCB_contrib,Time, 
                                    SelectWavelengthIndex, 
                                    SelectIntensityIndex,t,
                                    NumBins=Numbins, Total=False)

            
        ax[i].set_xlabel('Energy (eV)')
        ax[i].set_ylabel('Electron density ($\mathrm{cm^{-3}}$)')
        ax[i].set_title('t={time:.0f} fs'.format(time=t*10**15))
        labels = [MatPropDict['VB name'][i] for i in range(MatPropDict['VB name'].size)]
        
        ax[i].legend(labels)
        fig.tight_layout()
        plt.show()
        
        #Plot CDF:
        CDF = EnergyDistributionCDF(ax2[i], EnergyDistribution, nCB_contrib, Time,
                                SelectWavelengthIndex, SelectIntensityIndex,
                                t,NumBins=Numbins)
        ax2[i].set_xlabel('Energy (eV)')
        ax2[i].set_ylabel('Probability')
        ax2[i].set_title('t={time:.0f} fs'.format(time=t*10**15))
        fig2.tight_layout()
        
        
    if SavePlots:
        plt.savefig(Directory + 'ElectronDensity_vs_Energy-Plot' + Figure_filetype)

        
#%%
#Plot final electron density vs. intensity
if Plot_nCB_final_vs_intensity or Plot_all:
    fontsize = 18
    fig, ax = plt.subplots(figsize=(8,7))
    for i in range(NumWavelengths):
        ax.plot(Intensity.squeeze()*10**-13,nCB_final[i,:]*10**-6,
                label='{wavelength:.2f} {mu}m'.format(
                        wavelength=Wavelength[i,0,0]*10**6, mu=textmu))
    ax.set_xlabel('Peak intensity ($\mathrm{GW/cm^2}$)',
                  fontsize=fontsize)
    ax.set_ylabel('Final CB electron density ($\mathrm{cm^{-3}}$)',
                  fontsize=fontsize)
    ax.tick_params(labelsize=18)
    ax.set_title('Polarization along {} direction'.format(Direction))
#    plt.semilogy()
    plt.legend(fontsize=18)
    
    data = np.vstack((Intensity.squeeze()*10**-13,nCB_final[0,:]*10**-6,nCB_final[1,:]*10**-6,nCB_final[2,:]*10**-6,nCB_final[3,:]*10**-6))
    np.savetxt('Final_nCB_vs_intensity_{}.csv'.format(Direction),data.T, delimiter=',')
    plt.savefig('Final_nCB_vs_intensity_{}.png'.format(Direction))
    if SavePlots:
        plt.savefig(Directory + 'nCB_final_vs_intensity' + Figure_filetype)
    
#%%
#Plot Vinogradov absorption rate per electron vs energy      
if Plot_VinogradovRate_vs_energy and Vinogradov or Plot_all and Vinogradov:
    
    fig,ax = plt.subplots(nrows=1,ncols=len(SelectTimes),
                          figsize=(8*len(SelectTimes),6))
    
    for i,t in enumerate(SelectTimes):
        SelectTimeIndex = min(range(len(Time[:,SelectWavelengthIndex,
                                             SelectIntensityIndex,0])),
                              key=lambda idx: abs(Time[idx,
                                                       SelectWavelengthIndex,
                                                       SelectIntensityIndex,0]
                                                  -SelectTimes[i])) #Closest index
        SelectTime = Time[SelectTimeIndex] #Closest value
        
        Rates = AbsorptionRatePerElectron[SelectTimeIndex,:,SelectWavelengthIndex,SelectIntensityIndex,:]*10**-15/q
        Energies = EnergyDistribution[SelectTimeIndex,:,SelectWavelengthIndex,SelectIntensityIndex,:]/q
        
        RateMask = Rates!=0 #Mask for selecting non-zero values
        Rates_masked = Rates[RateMask].reshape((SelectTimeIndex,Rates.shape[1])) #Applying the mask flattens the array, need to reshape it
        Energies_masked = Energies[RateMask].reshape((SelectTimeIndex,Rates.shape[1])) #Applying the mask flattens the array, need to reshape it
        
        #Reorder arrays for plotting
        Rate_vs_Energy_type = np.dtype([('Rate',Rates_masked.dtype),('Energy',Energies_masked.dtype)])
        Rate_vs_Energy = np.empty(Rates_masked.shape,dtype=Rate_vs_Energy_type)
        Rate_vs_Energy['Rate'] = Rates_masked
        Rate_vs_Energy['Energy'] = Energies_masked
        Rate_vs_Energy = np.sort(Rate_vs_Energy, order='Energy',axis=0)
        
        ax[i].plot(Rate_vs_Energy['Energy'],Rate_vs_Energy['Rate'])
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax[i].set_xlabel('Energy (eV)')
        ax[i].set_ylabel('Absorption rate per electron (eV/fs)')
        ax[i].set_title('$t={time:.0f}$ fs'.format(time=t*10**15))
        
    if SavePlots:
        plt.savefig(Directory + 'VinogradovRatePerElectron_vs_Energy-Plot' + Figure_filetype)

#%%
#==============================================================================
#Avalanche Plot Area
#Plot_time_dependent_quantity(figsize,WAv,'$W_{Av} \: (\mathrm{1/(fs \cdot cm^3)})$',
#                                 Directory+'ImpactionizationRate-vs-time-Plot',
#                                 Figure_filetype,10**-21,MatPropDict,Time,
#                                 SelectWavelengthIndex,SelectIntensityIndex,
#                                 Total=False,Semilog=False,SavePlot=SavePlots)
#
#Plot_time_dependent_quantity(figsize,nCB_Av,'$nCB_{Av} \: (\mathrm{1/(cm^3)})$',
#                                 Directory+'ImpactElectronDensityContribution-vs-time-Plot',
#                                 Figure_filetype,10**-6,MatPropDict,Time,
#                                 SelectWavelengthIndex,SelectIntensityIndex,
#                                 Total=False,Semilog=False,SavePlot=SavePlots)
#
#Plot_time_dependent_quantity(figsize,Tei,'$t_{ei} \: (fs)$',
#                                 Directory+'eiCollisionRate-vs-time-Plot',
#                                 Figure_filetype,10**15,MatPropDict,Time,
#                                 SelectWavelengthIndex,SelectIntensityIndex,
#                                 Total=False,Semilog=True,SavePlot=SavePlots)
#
#Plot_time_dependent_quantity(figsize,Tee,'$t_{ee} \: (fs)$',
#                                 Directory+'eeCollisionRate-vs-time-Plot',
#                                 Figure_filetype,10**15,MatPropDict,Time,
#                                 SelectWavelengthIndex,SelectIntensityIndex,
#                                 Total=False,Semilog=False,SavePlot=SavePlots)

fig,ax = plt.subplots(figsize=figsize)
SelectTime = Time.squeeze()*10**15
SelectQuantity1 = np.sum((10**-21*WAv[:,SelectWavelengthIndex,SelectIntensityIndex]),axis=1)
SelectQuantity2 = np.sum((10**-21*WPI[:,SelectWavelengthIndex,SelectIntensityIndex]),axis=1)
SelectQuantity3 = np.sum((10**-21*(WPI+WAv)[:,SelectWavelengthIndex,SelectIntensityIndex]),axis=1)
SelectQuantity = np.transpose(np.array([SelectQuantity1,SelectQuantity2,SelectQuantity3]))
LineObjects = ax.plot(SelectTime,SelectQuantity)
labels = ['II','PI','Total']
ax.legend(LineObjects, labels)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('$W \: (\mathrm{1/(fs \cdot cm^3)})$')
plt.semilogy()
plt.savefig(Directory+'MixedIonizationRates' + Figure_filetype)

fig,ax = plt.subplots(figsize=figsize)
SelectTime = Time.squeeze()*10**15
SelectQuantity1 = np.sum((10**-6*nCB_Av[:,SelectWavelengthIndex,SelectIntensityIndex]),axis=1)
SelectQuantity2 = np.sum((10**-6*nCB_PI[:,SelectWavelengthIndex,SelectIntensityIndex]),axis=1)
SelectQuantity3 = np.sum((10**-6*nCB[:,SelectWavelengthIndex,SelectIntensityIndex]),axis=1)
SelectQuantity = np.transpose(np.array([SelectQuantity1,SelectQuantity2,SelectQuantity3]))
LineObjects = ax.plot(SelectTime,SelectQuantity)
labels = ['II','PI','Total']
ax.legend(LineObjects, labels)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('$n_{e} \: (\mathrm{1/(cm^3)})$')
plt.semilogy()
plt.savefig(Directory+'MIxedCarrierDensities' + Figure_filetype)

if VaryRefractiveIndex:
    fig,ax = plt.subplots(figsize=figsize)
    SelectTime = Time.squeeze()*10**15
    SelectQuantity1 = (nreal[:,SelectWavelengthIndex,SelectIntensityIndex])
    SelectQuantity2 = (nimag[:,SelectWavelengthIndex,SelectIntensityIndex])
    SelectQuantity = np.transpose(np.array([SelectQuantity1,SelectQuantity2]))
    LineObjects = ax.plot(SelectTime,SelectQuantity)
    labels = ['Re(n)','Im(n)']
    ax.legend(LineObjects, labels)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('$n \: (\mathrm{arb. u.})$')
    #plt.semilogy()
    plt.savefig(Directory+'RefractiveIndex' + Figure_filetype)
    
#    Plot_time_dependent_quantity(figsize,Tf,'$T \: (\mathrm{arb. u.})$',
#                                     Directory+'Tf-vs-time-Plot',
#                                     Figure_filetype,1,MatPropDict,Time,
#                                     SelectWavelengthIndex,SelectIntensityIndex,
#                                     Total=False,Semilog=False,SavePlot=SavePlots) 
    
    SelectWavelength2 = 2.75
    SelectWavelength2 *= 10**-6
    SelectWavelengthIndex2 = min(range(Wavelength.size),key=lambda i: abs(Wavelength[i,0,0]-SelectWavelength2))
    SelectWavelength2 = Wavelength[SelectWavelengthIndex2,0,0]*10**6
    
    SelectWavelength3 = 3.15
    SelectWavelength3 *= 10**-6
    SelectWavelengthIndex3 = min(range(Wavelength.size),key=lambda i: abs(Wavelength[i,0,0]-SelectWavelength3))
    SelectWavelength3 = Wavelength[SelectWavelengthIndex3,0,0]*10**6
    
    SelectWavelength4 = 3.75
    SelectWavelength4 *= 10**-6
    SelectWavelengthIndex4 = min(range(Wavelength.size),key=lambda i: abs(Wavelength[i,0,0]-SelectWavelength4))
    SelectWavelength4 = Wavelength[SelectWavelengthIndex4,0,0]*10**6
    
    SelectWavelength5 = 4.15
    SelectWavelength5 *= 10**-6
    SelectWavelengthIndex5 = min(range(Wavelength.size),key=lambda i: abs(Wavelength[i,0,0]-SelectWavelength5))
    SelectWavelength5 = Wavelength[SelectWavelengthIndex5,0,0]*10**6
    
    fig,ax = plt.subplots(figsize=figsize)
    SelectTime = Time.squeeze()*10**15
    SelectQuantity1 = (Tf[:,SelectWavelengthIndex2,SelectIntensityIndex])
    SelectQuantity2 = (Tf[:,SelectWavelengthIndex3,SelectIntensityIndex])
    SelectQuantity3 = (Tf[:,SelectWavelengthIndex4,SelectIntensityIndex])
    SelectQuantity4 = (Tf[:,SelectWavelengthIndex5,SelectIntensityIndex])
    SelectQuantity = np.transpose(np.array([SelectQuantity1,SelectQuantity2,SelectQuantity3,SelectQuantity4]))
    LineObjects = ax.plot(SelectTime,SelectQuantity)
    labels = ['{} {}m'.format(np.round(SelectWavelength2,2),textmu),
              '{} {}m'.format(np.round(SelectWavelength3,2),textmu),
              '{} {}m'.format(np.round(SelectWavelength4,2),textmu),
              '{} {}m'.format(np.round(SelectWavelength5,2),textmu)]
    ax.legend(LineObjects, labels)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('$T \: (\mathrm{arb. u.})$')
#    plt.semilogy()
    plt.savefig(Directory+'Tf 4 WL' + Figure_filetype)
    
#Convert units


#Find index at which the array values are closest to the desired value


#Closest matching values

#%%
#==============================================================================
# Show figures
#==============================================================================
plt.show()

#%%
#==============================================================================
# Create and save dataframe from all output instances
#==============================================================================
#data_output_list = []
#for i in range(NumInstances):
#    data_output = pd.read_csv(Directory + 
#                              filename.replace('{instance}'.format(instance=Instance),
#                                               '{idx}'.format(idx=i+1)) + '.csv',header=None).values
#    data_output_list.append(data_output)
#data_output_array = np.vstack((data_output_list))
#data_output_dataframe = pd.DataFrame(data_output_array,columns=['Wavelength','Intensity','Intensity'])
#
#data_output_dataframe.to_csv(Directory + 
#           filename.replace('_{instance}of{total}'.format(instance=Instance,total=NumInstances),''),sep=',')
#data_output_dataframe.plot(x='Wavelength',subplots=True,sharex=True)


np.savetxt('thresholds.csv',Intensity_thresh,delimiter=',')
np.savetxt('wavelengths.csv',Wavelength[:,0,0],delimiter=',')     