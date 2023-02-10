import numpy as np
from math import pi
from physics.constants import q,epsilon0

#%%
#==============================================================================
# Avalanche ionization
#==============================================================================
#def DrudeCollisionTime(ne,meff,Bandgap):
#    return 16*pi*epsilon0**2*np.sqrt(meff*(0.1*Bandgap)**3)/(np.sqrt(2)*q**4*ne)

def ElectronIonCollisionTime(ne,meff,Bandgap):
    return 16*pi*epsilon0**2*np.sqrt(meff*(0.1*Bandgap)**3)/(np.sqrt(2)*q**4*ne)

def ElectronElectronCollisionTime(ne,meff,Bandgap):
    return ((3*(q**2)*np.sqrt(meff))/(16*pi*epsilon0))*((0.1*Bandgap)**(-3/2))

def DrudeCollisionTime(ne,meff,Bandgap):
    return (ElectronIonCollisionTime(ne,meff,Bandgap)**(-1)+ElectronElectronCollisionTime(ne,meff,Bandgap)**(-1))**(-1)

#def CrossSection(omega,meff,tau,epsilonc):
#    return q**2/(c*epsilon0*np.sqrt(epsilonc)*meff)*(tau/(1 + omega**2*tau**2))

#def CrossSection(omega,epsilonc,ne,meff,n2,Gamma,Intensity):
#    nimag = Imagn(omega,epsilonc,ne,meff,n2,Gamma,Intensity)
#    alpha = 2*omega*nimag/c
#    
#    return alpha/ne

def AvalancheIonization(omega,ne,meff,Bandgap,Efield):
    """Drude model avalnche ionization"""
    tau = DrudeCollisionTime(ne,meff,Bandgap)
#    tau = 32*(10**-15)
#    Cross_Section = CrossSection(omega,epsilonc,ne,meff,n2,Gamma,Intensity)
#    Cross_Section = CrossSection(omega,meff,tau,epsilonc)
    conductivityfactor = ((q**2)*tau)/(2*meff*(1+(omega**2)*(tau**2)))
    
# multiplication by 'ne' is effective done in 172 of 'Modified_Keldysh.py'. If you try to include it in the return if causes an error in the calculation for some reason.    
    return (conductivityfactor*(Efield**2))/Bandgap
