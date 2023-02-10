#==============================================================================
# Imports
#==============================================================================
from math import pi, log
import numpy as np
from tqdm import tqdm
import os

#Import custom modules
from physics.constants import c,epsilon0
from Keldysh_Photoionization import (DensityOfStates, KeldyshParameter,
                                     EffectiveBandgap,PhotoionizationRate)
from Vinogradov import MinimumEnergy, VinogradovEnergyRate
from Avalanche_Ionization import (AvalancheIonization,DrudeCollisionTime,
                                  ElectronIonCollisionTime,
                                  ElectronElectronCollisionTime)
from Fresnel import transmission
from Drude import Realn, Imagn
from Material_Properties import MaterialProperties


#%%
#==============================================================================
# Pulse integration
#==============================================================================
def PulseIntegration(MatProp,theta,Polarization,Direction,PulseDuration,
                     Wavelength,Intensity,intrange,NumTimesteps,Instance=1,VB_depletion=False,
                     Multiband=False,Vinogradov=False,Reflection=False,
                     Direct=False,VaryRefractiveIndex=False,Avalanche=False):
    """Integrates over the duration of the pulse."""
    
   
    #Material properties
    MatPropDict = MaterialProperties(MatProp,Direction,Wavelength,Multiband)
    mvb = MatPropDict['VB effective mass']
    mcb = MatPropDict['CB effective mass'] if Direct else MatPropDict['Indirect CB effective mass']
    Bandgap = MatPropDict['Bandgap'] if Direct else MatPropDict['Indirect bandgap']
    ValenceBandShape = MatPropDict['VB shape']
    n0 = MatPropDict['Total electron density']
    ncomplex_unexcited = MatPropDict['Refractive index']
    tau = MatPropDict['CollisionTime']
    DeformationPotential = MatPropDict['Deformation potential']
    Density = MatPropDict['Density']
    SpeedofSound = MatPropDict['Speed of sound']
    PhononFrequency = MatPropDict['Phonon frequency']
    Tphonon = MatPropDict['Phonon temperature']
    epsilonInf = MatPropDict['High frequency permittivity']
    epsilonStatic = MatPropDict['Static permittivity']
        
    #Calculated parameters
    omega = 2*pi*c/Wavelength #Laser frequency
    I0 = Intensity #Peak laser intensity
    time = np.linspace(-intrange/2,intrange/2,NumTimesteps)
    dt = np.gradient(time,axis=0) #timestep
    NumTimesteps = dt.shape[0] #Number of timesteps
    NumWavelengths = Wavelength.size #Number of wavelengths
    NumIntensities = Intensity.size #Number of intensities
    NumValenceBands = ValenceBandShape.size #Number of valence bands
    meff = (mcb*mvb)/(mcb+mvb) #Optical effective mass
    Gamma = 1/tau #Electron-phonon collision rate (Drude model)
    
    
    #Electron density available in each valence band (m^-3). Axis 0 is the 
    #wavelength, axis 1 the intensity, and axis 2 the valence band.
    nVB = np.full((NumWavelengths,NumIntensities,NumValenceBands),n0/NumValenceBands)
    
    """Conduction band electron density (m^-3). Each row is the contribution 
    for a given transition while each column corresponds with a given time. 
    There will be some initial value that will be treated as though it 
    originated as an equal contribution from each band."""
    nCB_initial = 1*10**14 #Should make this an input material property
    nCB_contrib = np.zeros((NumTimesteps,NumWavelengths,NumIntensities,NumValenceBands))
    nCB_PI_contrib = np.zeros((NumTimesteps,NumWavelengths,NumIntensities,NumValenceBands))
    nCB_Av_contrib = np.zeros((NumTimesteps,NumWavelengths,NumIntensities,NumValenceBands))
    nCB_contrib[0,:,:,:] = nCB_initial/NumValenceBands #The zeroth element of axis 0 represents the intial value
    nCB_PI = np.cumsum(nCB_PI_contrib,axis=0) #Total electron density for each valence band over time (PI contribution)
    nCB_Av = np.cumsum(nCB_Av_contrib,axis=0) #Total electron density for each valence band over time (Av contribution)
    nCB = np.cumsum(nCB_contrib,axis=0) #Total electron density for each valence band over time
    
    """Distribution of energies in the conduction band (J/electron). This array
    has five dimensions, one for time, one for the iteration number (which can 
    be matched with the electron density contribution), one for wavelength, one
    for intensity, and one for the valence bands. Axis 0 is time, axis 1 is 
    iteration number, axis 2 is the wavelength, axis 3 is the intensity, and axis 
    4 is the valence band. There will be some initial value for the initial, 
    intrinsic carrier concentration. Memory usage can easily exceed available 
    RAM when a large number of timesteps, wavelengths, and/or intensities are 
    used, so numpy.memmap is used to store this array locally as a dat file."""
    if Vinogradov:
        E_initial = 0
        filename = os.path.join('EnergyDistribution_CB_{instance}.dat'.format(instance=Instance))
        EnergyDistribution_CB = np.memmap(filename, dtype='float64', mode='w+', 
                                          shape=(NumTimesteps,NumTimesteps,NumWavelengths,NumIntensities,NumValenceBands))
        EnergyDistribution_CB[0,0,:,:,:] = E_initial
    else:
        EnergyDistribution_CB = np.zeros((NumTimesteps,NumWavelengths,NumIntensities,NumValenceBands))
    
    #Arrays for keeping track of time evolution of various quantities
    Bandgap_vs_t = np.zeros(nCB.shape) #Effective bandgap vs time
    PopFactor_vs_t = np.zeros(nCB.shape) #Population factor vs time
    SlowlyVaryingAmplitude_vs_t = np.zeros(nCB.shape)
    WPI_vs_t = np.zeros(nCB.shape) #Photoionization rate vs time
    WAv_vs_t = np.zeros(nCB.shape) #Avalanche ionization rate vs time
    CollisionTimes_vs_t = np.zeros(nCB.shape) #Avalanche ionization rate vs time
    ei_CollisionTimes_vs_t = np.zeros(nCB.shape)
    ee_CollisionTimes_vs_t = np.zeros(nCB.shape)
    ArrivalEnergy_vs_t = np.zeros(nCB.shape) #Arrival energy after transitioning to the CB
    AbsorptionRatePerElectron = np.zeros(EnergyDistribution_CB.shape) #Vinogradov absorption rate per electron
    I_vs_t = np.zeros((NumTimesteps,NumWavelengths,NumIntensities)) #Intensity vs time
    Tf_vs_t = np.zeros((NumTimesteps,NumWavelengths,NumIntensities))
    Efield_vs_t = np.zeros((NumTimesteps,NumWavelengths,NumIntensities)) #Electric field vs time
    nreal_vs_t = np.full((NumTimesteps,NumWavelengths,NumIntensities),ncomplex_unexcited.real.reshape(1,NumWavelengths,1)) #Real refractive index vs time
    nimag_vs_t = np.full((NumTimesteps,NumWavelengths,NumIntensities),ncomplex_unexcited.imag.reshape(1,NumWavelengths,1)) #Imaginary refractive index vs time
    ncomplex_vs_t = nreal_vs_t + nimag_vs_t*1.0j
    nreal = ncomplex_unexcited.real #Initial value of real refractive index
    nimag = ncomplex_unexcited.imag #Initial value of imag refractive index
    
    #Loop over time
    for idx,t in enumerate(tqdm(time)):
        
        #Total conduction band electron density at previous timestep
        nCB_previous_total = np.sum(nCB[idx-1],axis=2)
        
        
        #Laser intensity
        I = I0*np.exp(-4*log(2)*t**2/PulseDuration**2) #Incident laser intensity
        I_vs_t[idx] = I.squeeze(axis=2)

        
        #Refractive index change
        if VaryRefractiveIndex:
            nreal = Realn(omega,ncomplex_unexcited.real**2,
                      nCB_previous_total[...,np.newaxis],mcb[0],0,Gamma,I)
            nimag = Imagn(omega,ncomplex_unexcited.real**2,
                      nCB_previous_total[...,np.newaxis],mcb[0],0,Gamma,I)
            nreal_vs_t[idx] = nreal.squeeze(axis=2)
            nimag_vs_t[idx] = nimag.squeeze(axis=2)
            ncomplex_vs_t[idx] = nreal_vs_t[idx] + nimag_vs_t[idx]*1.0j
        
        
        #Incident laser field envelope
        Incident_Efield = np.sqrt(2*I/(c*epsilon0))
        
        
        #Laser field envelope within the medium
        if Reflection:
            Tf = transmission(ncomplex_vs_t[idx-1,...,np.newaxis],theta,Polarization) #Field transmission
            #Rf = reflection(ncomplex_vs_t[idx-1,...,np.newaxis],theta,Polarization) #Field reflection
            Efield = np.abs(Tf)*Incident_Efield
        else:
            Efield = Incident_Efield/(np.sqrt(nreal))
        Efield_vs_t[idx] = Efield.squeeze(axis=2)
        Tf_vs_t[idx] = np.abs(Tf).squeeze(axis=2)
    
        #Gamma factors
        gammas = KeldyshParameter(Efield,omega,meff,Bandgap,subscript='All')
        
        #Effective bandgap
        Delta_eff = EffectiveBandgap(Bandgap,ValenceBandShape,gammas)
        Bandgap_vs_t[idx] = Delta_eff
        
        #Photoionization rates
        SlowlyVaryingAmplitude, WPI = PhotoionizationRate(omega,meff,Delta_eff,
                                                          ValenceBandShape,gammas) #Photoionization
        SlowlyVaryingAmplitude_vs_t[idx] = SlowlyVaryingAmplitude
        WPI_vs_t[idx] = WPI
        
        #Avalanche ionization rates
        if Avalanche:
#            WAv = AvalancheIonization(omega,nCB[idx-1],meff,Bandgap,I,
#                                      ncomplex_unexcited.real**2)*nCB[idx-1]
            WAv = AvalancheIonization(omega,nCB[idx-1],meff,Bandgap,Efield)*nCB[idx-1]
        else:
            WAv = 0
        WAv_vs_t[idx] = WAv
        
        if Avalanche:
            CollisionTimes = DrudeCollisionTime(nCB[idx-1],meff,Bandgap)
            ei_collision_times = ElectronIonCollisionTime(nCB[idx-1],meff,Bandgap)
            ee_collision_times = ElectronElectronCollisionTime(nCB[idx-1],meff,Bandgap)
        else:
            CollisionTimes = 0
            ei_collision_times = 0
            ee_collision_times = 0
        CollisionTimes_vs_t[idx] = CollisionTimes
        ei_CollisionTimes_vs_t[idx] = ei_collision_times
        ee_CollisionTimes_vs_t[idx] = ee_collision_times
        
        #Population factor
        PopFactor = DensityOfStates(nVB,nCB[idx-1])
        PopFactor_vs_t[idx] = PopFactor
        
        #Electron density contributions at each time step
        if VB_depletion:
            nCB_PI_contrib[idx] = PopFactor*WPI*dt[idx]
            nCB_Av_contrib[idx] = PopFactor*WAv*dt[idx]
        else:
            nCB_PI_contrib[idx] = WPI*dt[idx]
            nCB_Av_contrib[idx] = WAv*dt[idx]
        nCB_contrib[idx] = nCB_PI_contrib[idx] + nCB_Av_contrib[idx]
        
        #Conduction band electron density
        nCB_PI[idx] = nCB_PI[idx-1] + nCB_PI_contrib[idx] #PI contribution
        nCB_Av[idx] = nCB_Av[idx-1] + nCB_Av_contrib[idx] #Av contribution
        nCB[idx] = nCB[idx-1] + nCB_PI_contrib[idx] + nCB_Av_contrib[idx] #Total

        if Vinogradov:
            #Conduction band energy absorption rates per electron
            PreviousEnergyDistribution_CB = EnergyDistribution_CB[idx-1] #Energy distribution at the previous timestep
            A = np.zeros(PreviousEnergyDistribution_CB.shape) #Initialize array of Vinogradov rates
        
            #Assign appropriate values for energy levels that are populated
            A[:idx] = VinogradovEnergyRate(Efield,omega,meff,
                            PreviousEnergyDistribution_CB[:idx],epsilonStatic,
                            epsilonInf,Density,DeformationPotential,
                            SpeedofSound,PhononFrequency,Tphonon)
            AbsorptionRatePerElectron[idx] = A

            #Minimum energy of electrons excited to the conduction band
            Emin = MinimumEnergy(omega,meff,mcb,Bandgap,Delta_eff,ValenceBandShape,gammas)
            ArrivalEnergy_vs_t[idx] = Emin
        
            #Conduction band electron energy distribution per electron
            EnergyDistribution_CB[idx,:idx] = EnergyDistribution_CB[idx-1,:idx] + A[:idx]*dt[np.newaxis,idx]
            EnergyDistribution_CB[idx,idx] = Emin
    
    if Vinogradov:
        #Conduction band electron energy absorption rates
        AbsoluteEnergyDistribution_CB = EnergyDistribution_CB*nCB_contrib #Absolute energy distribution (Joules/m^3)
        EnergyAbsorbed_CB = np.sum(AbsoluteEnergyDistribution_CB,axis=1) #Energy absorbed by CB electrons over time
        AbsorptionRate = np.gradient(EnergyAbsorbed_CB, axis=0)/PulseDuration #Energy absorption rate by CB electrons over time
    else:
        AbsorptionRate = np.zeros(nCB.shape)
    
       
    Dict = {'Time':time, 'Intensity vs t':I_vs_t, 'Electric field':Efield_vs_t, 
            'Effective bandgap':Bandgap_vs_t, 'Photoionization rate':WPI_vs_t,
            'Avalanche ionization rate':WAv_vs_t,
            'Slowly varying amplitude':SlowlyVaryingAmplitude_vs_t,
            'Population factor':PopFactor_vs_t,
            'Conduction band electron density':nCB,
            'Photoionization contribution':nCB_PI,
            'Avalanche contribution':nCB_Av,
            'CB electron density contributions at each timestep':nCB_contrib,
            'CB absorption rate per electron':AbsorptionRatePerElectron,
            'CB absorption rate':AbsorptionRate,
            'CB arrival energy':ArrivalEnergy_vs_t,
            'CB energy distribution':EnergyDistribution_CB,
            'Total electron density':n0,
            'Real refractive index':nreal_vs_t,
            'Imaginary refractive index':nimag_vs_t,
            'Avalanche Collision Time':CollisionTimes_vs_t,
            'e-i Collision Time':ei_CollisionTimes_vs_t,
            'e-e Collision Time':ee_CollisionTimes_vs_t,
            'Transmission vs Time':Tf_vs_t}

    return Dict

#%%
#==============================================================================
# Find threshold
#==============================================================================

def FindIntensityThreshold(Intensity,nCB,nCB_thresh):
    """Given an array of the final conduction band electron density (with axis 
    0 corresponding to wavelength and axis 1 to intensity), this function finds 
    the indices where the electron density best matches a given threshold. It
    then gives the corresponding intensity.
    
    Intensity: Intensities over which nCB was calculated (W/m^2)
    
    nCB: Final total conduction band electron density in m^-3 (numpy array with
    shape (NumWavelengths, NumIntensities))
    
    nCB_thresh: Threshold electron density in m^-3."""
    
    #Find the indices whose values are closest to the threshold
    ClosestIndices = np.argmin(np.abs((nCB - nCB_thresh)),axis=1) #Find indices for each intensity (axis 1)
    
    #Find the corresponding Intensity values
    Intensity_closest = Intensity.flat[ClosestIndices]
    
    #Check if the threshold was within the range of nCB
    ThresholdFound = np.logical_and(np.max(nCB,axis=1)>nCB_thresh, 
                                    np.min(nCB,axis=1)<nCB_thresh)
    
    return Intensity_closest, ThresholdFound
