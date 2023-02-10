#==============================================================================
# Imports
#==============================================================================
from math import pi, sqrt
import numpy as np
from scipy.special import dawsn,ellipe,ellipk
from physics.constants import q, hbar


#%%
#==============================================================================
# Density of states and Keldysh adiabatic parameter
#==============================================================================
def DensityOfStates(ReservoirDensity,ExcitedStateDensity):
    """Returns the density of states coefficient for a transition from a 
    reservoir of electron density ReservoirDensity to an excited state of 
    density ExcitedStateDensity."""

    return (ReservoirDensity - ExcitedStateDensity)/ReservoirDensity


def KeldyshParameter(Efield,omega,meff,Bandgap,subscript='All'):
    """Keldysh parameters. The subscript indicates which one to use (original 
    or first or second auxillary). If the subscript is 'All', a numpy array of 
    all three gammas will be output."""

    gamma = omega*np.sqrt(meff*Bandgap)/(q*Efield)

    if subscript=='None':
        return gamma
    elif subscript==1:
        return gamma**2/(1+gamma**2)
    elif subscript==2:
        return 1/(1+gamma**2)
    elif subscript=='All':
        return np.array([gamma,gamma**2/(1+gamma**2),1/(1+gamma**2)])
    else:
        raise ValueError

#%%
#==============================================================================
# Effective bandgap
#==============================================================================
def EffectiveBandgap(Bandgap,ValenceBandShape,gammas):
    """The effective energy gap between two bands. Inputs not already specified
    by previous functions:

    Bandgap: Ordinary bandgap (J)

    ValenceBandShape: Shape used to approximate the valence band (array of 
    strings, either 'Kane' or 'Parabolic')

    gammas: array of gamma factors as output by the KeldyshParameter function 
    when subscript='All'."""

    gamma = gammas[0]
    gamma1 = gammas[1]
    gamma2 = gammas[2]

    KaneMask = ValenceBandShape=='Kane'
    ParabolicMask = ValenceBandShape=='Parabolic'

    Delta_eff = np.zeros(gammas.shape[1:]) #The zeroth element of gammas.shape refers to the gamma type

    #Assign the appropriate value for each element based on the valence band shape
    Delta_eff[:,:,KaneMask] = ( 2*Bandgap[KaneMask]/(pi*np.sqrt(gamma1[:,:,KaneMask]))*
    							ellipe(gamma2[:,:,KaneMask]) )
    Delta_eff[:,:,ParabolicMask] = Bandgap[ParabolicMask]*(1+1/(4*gamma[:,:,ParabolicMask]**2))

    return Delta_eff

#%%
#==============================================================================
# Keldysh photoionization rate equations
#==============================================================================
def KeldyshSum(x,ValenceBandShape,gammas):
    """Sum used in calculation of Keldysh photoionization rate"""

    gamma = gammas[0]
    gamma1 = gammas[1]
    gamma2 = gammas[2]

    NumElements = 1000 #Number of terms to calculate in the summation
    k = np.arange(0,NumElements).reshape((NumElements,1,1,1)) #array of sum indices

    KaneMask = ValenceBandShape=='Kane'
    ParabolicMask = ValenceBandShape=='Parabolic'

    SumElements = np.zeros((NumElements,) + x.shape) #Include another axis for the element number

    #For Kane-shaped valence bands:
    SumElements[:,:,:,KaneMask] = ( np.exp(-k*pi*(ellipk(gamma1[:,:,KaneMask])-
    ellipe(gamma1[:,:,KaneMask]))/ellipe(gamma2[:,:,KaneMask]))*
    dawsn(np.sqrt(pi**2*(np.floor(x[:,:,KaneMask]+1)-
    x[:,:,KaneMask]+k)/(2*ellipk(gamma2[:,:,KaneMask])*
    ellipe(gamma2[:,:,KaneMask])))) )

    #For parabolic-shaped valence bands:
    SumElements[:,:,:,ParabolicMask] = ( np.exp(-2*k*(np.arcsinh(gamma[:,:,ParabolicMask]*sqrt(2))-
    		gamma[:,:,ParabolicMask]*sqrt(2)/
    		(np.sqrt(1+2*gamma[:,:,ParabolicMask]**2))))*
    		dawsn(np.sqrt(2*(k+np.floor(x[:,:,ParabolicMask]+1)-
    		x[:,:,ParabolicMask])*gamma[:,:,ParabolicMask]*sqrt(2)/
    		(np.sqrt(1 + 2*gamma[:,:,ParabolicMask]**2)))) )

    Sum = np.sum(SumElements,axis=0)

    return Sum


def PhotoionizationRate(omega,meff,Effective_Bandgap,ValenceBandShape,gammas):
    """Calculates the photoionization rate from the Keldysh equations."""

    gamma = gammas[0]
    gamma1 = gammas[1]
    gamma2 = gammas[2]

    KaneMask = ValenceBandShape=='Kane'
    ParabolicMask = ValenceBandShape=='Parabolic'

    x = Effective_Bandgap/(hbar*omega)
    Q = np.zeros(Effective_Bandgap.shape)
    WPI = np.zeros(Effective_Bandgap.shape)

    Sum = KeldyshSum(x,ValenceBandShape,gammas)

    #For Kane-shaped valence bands:
    Q[:,:,KaneMask] = np.sqrt(pi/(2*ellipk(gamma2[:,:,KaneMask])))*Sum[:,:,KaneMask]

    WPI[:,:,KaneMask] = ( 2*2*2*omega/(9*pi)*(meff[KaneMask]*omega/
    				(hbar*np.sqrt(gamma1[:,:,KaneMask])))**1.5*
    				Q[:,:,KaneMask]*np.exp(-pi*np.floor(x[:,:,KaneMask]+1)*
    				(ellipk(gamma1[:,:,KaneMask])-ellipe(gamma1[:,:,KaneMask]))/
    				ellipe(gamma2[:,:,KaneMask])) )

    #For parabolic-shaped valence bands:
    Q[:,:,ParabolicMask] = ( np.sqrt(np.sqrt(1 +
    				2*gamma[:,:,ParabolicMask]**2)/
    				(gamma[:,:,ParabolicMask]*sqrt(2)))*
    				Sum[:,:,ParabolicMask] )

    WPI[:,:,ParabolicMask] = ( 2*2*omega/(8*pi)*(meff[ParabolicMask]*omega/hbar)**1.5*
    							np.exp(-2*np.floor(x[:,:,ParabolicMask]+1)*
    							(np.arcsinh(gamma[:,:,ParabolicMask]*sqrt(2))-
    							gamma[:,:,ParabolicMask]*sqrt(2)/(np.sqrt(1 + 2*gamma[:,:,ParabolicMask]**2)))-
    							2*x[:,:,ParabolicMask]*
    							2*sqrt(2)*gamma[:,:,ParabolicMask]**3/(np.sqrt(1 +
    							2*gamma[:,:,ParabolicMask]**2)*(1 +
    							4*gamma[:,:,ParabolicMask]**2)))*Q[:,:,ParabolicMask] )

    return Q,WPI

