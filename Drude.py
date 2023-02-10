#==============================================================================
# Imports
#==============================================================================
import numpy as np
from physics.constants import q, epsilon0


#%%
#==============================================================================
# Drude model
#==============================================================================
def PlasmaFrequency(ne,meff):
    """Plasma frequency"""
    return np.sqrt(ne*q**2/(meff*epsilon0))

def Real_epsilon(epsilonc,omega,omegap,Gamma,n2,Intensity):
    """Real part of permittivity"""
    return ( epsilonc - omegap**2/(omega**2+Gamma**2) + 
            2*np.sqrt(epsilonc)*n2*Intensity )

def Imaginary_epsilon(omega,omegap,Gamma):
    """Imaginary part of permittivity"""
    return omegap**2*Gamma/(omega*(omega**2+Gamma**2))

def Realn(omega,epsilonc,ne,meff,n2,Gamma,Intensity):
    """Real part of refractive index"""
    omegap = PlasmaFrequency(ne,meff)

    ep = Real_epsilon(epsilonc,omega,omegap,Gamma,n2,Intensity)
    epp = Imaginary_epsilon(omega,omegap,Gamma)

    return np.sqrt(0.5*(np.sqrt(ep**2 + epp**2)+ep))

def Imagn(omega,epsilonc,ne,meff,n2,Gamma,Intensity):
    """Imaginary part of refractive index"""
    omegap = PlasmaFrequency(ne,meff)

    ep = Real_epsilon(epsilonc,omega,omegap,Gamma,n2,Intensity)
    epp = Imaginary_epsilon(omega,omegap,Gamma)

    return np.sqrt(0.5*(np.sqrt(ep**2 + epp**2)-ep))
