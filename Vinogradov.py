#==============================================================================
# Imports
#==============================================================================
from math import pi
import numpy as np
from physics.constants import q, hbar, kb

#%%
#==============================================================================
# Vinogradov equations
#==============================================================================
def MinimumEnergy(omega,meff,mcb,Bandgap,Effective_Bandgap,ValenceBandShape,gammas):
    """The lowest energy level in the conduction band at which the electrons 
    arrive. Used in the calculation of the Vinogradov energy rate equation. 
    Inputs not already specified by previous functions:
    
    mcb: Conduction band effective mass (kg)"""

    KaneMask = ValenceBandShape=='Kane'
    ParabolicMask = ValenceBandShape=='Parabolic'

    MinimumEnergy = np.zeros(Effective_Bandgap.shape)

    #For Kane-shaped valence bands:
    MinimumEnergy[:,:,KaneMask] = meff[KaneMask]/(2*mcb[KaneMask])*Bandgap[KaneMask]*( 
                (hbar*omega/Bandgap[KaneMask] * np.floor(Effective_Bandgap[:,:,KaneMask]/(hbar*omega)+1))**2 - 1 )

    #For parabolic-shaped valence bands:
    MinimumEnergy[:,:,ParabolicMask] = meff[ParabolicMask]/mcb[ParabolicMask] * Bandgap[ParabolicMask] * \
                                   ( hbar*omega/Bandgap[ParabolicMask] * \
                                    np.floor(Effective_Bandgap[:,:,ParabolicMask]/(hbar*omega)+1) - 1 )

    return MinimumEnergy
        
        
def Psi(x,FunctionNumber):
    """Functions used in the Vinogradov equation. FunctionNumber determines 
    which function to use, Psi1 or Psi2"""
    
    xzero = x[x==0]
    xsmall = x[x<=1]
    xlarge = x[x>1]
        
    #First Psi function
    if FunctionNumber==1:
        
        Psi1 = np.zeros(x.shape)
        
        Psi1[x<=1] = (17/30*np.sqrt(1-xsmall**2) + np.arcsin(xsmall)/(10*xsmall) + \
                       4/15*xsmall**2*np.sqrt(1-xsmall**2) + 2/3*xsmall*np.arcsin(xsmall) - \
                       1/15*xsmall**4*np.arccosh(1/xsmall))
        Psi1[x==0] = np.full(xzero.shape,2./3.)
        Psi1[x>1] = pi*xlarge/3 + pi/(20*xlarge)
        
        return Psi1
        
    
    #Second Psi function
    elif FunctionNumber==2:

        Psi2 = np.zeros(x.shape)

        Psi2[x<=1] = 4/3*np.sqrt(1-xsmall**2) + 2*np.arcsin(xsmall)/(3*xsmall) + \
                      2/3*xsmall**2*np.arccosh(1/xsmall)
        Psi2[x==0] = np.full(xzero.shape,2.0)
        Psi2[x>1] = pi/(3*xlarge)
        
        return Psi2
    
    else:
        raise ValueError
        

def VinogradovEnergyRate(Efield,omega,mcb,Energy,epsilonStatic,epsilonInf,
                         Density,DeformationPotential,SpeedofSound,
                         PhononFrequency,Tphonon):
    """Vinogradov equation for the rate of energy absorption (at time t) of a 
    single conduction band electron due to phonon collisions during 
    ponderomotive oscillations from an incident laser pulse. Inputs:
    
    Efield: Electric field strength of the pulse envelope (V/m)
    
    omega: Laser frequency (Hz)
    
    mcb: Conduction band effective mass (kg)
    
    Energy: Electron energy level. At t=0, it is the lowest energy level in the
    conduction band at which the electrons arrive. (J)
    
    epsilonStatic: Static dielectric permittivity (F/m)
    
    epsilonInf: High-frequency dielectric permittivity (F/m)
    
    Density: Material density (kg/m^3)
    
    DeformationPotential: Deformation potential for the electron-phonon 
    collisions (J)
    
    SpeedofSound: Longitudinal speed of sound for the material (m/s)
    
    PhononFrequency: Polar optical phonon frequency in the long-wavelength 
    limit (Hz)
    
    Tphonon: Phonon temperature (K)"""
    
    return ( (q*Efield/omega)**3 * 2*DeformationPotential**2*kb*Tphonon/(pi**2*hbar**4*Density*SpeedofSound**2)
            * Psi(omega*np.sqrt(2*mcb*Energy)/(q*Efield),1) + (q*Efield/omega) * 
            (q**2*PhononFrequency*(epsilonStatic-epsilonInf))/(pi*hbar*epsilonStatic*epsilonInf) * 
           1/np.tanh(hbar*PhononFrequency/(2*kb*Tphonon)) * Psi(omega*np.sqrt(2*mcb*Energy)/(q*Efield),2) )
