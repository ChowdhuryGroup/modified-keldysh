#==============================================================================
# Imports
#==============================================================================
import numpy as np

#%%
#==============================================================================
# Fresnel transmission and reflection (field)
#==============================================================================
def transmission(n,AOI,pol):
    """Returns the Fresnel transmission coefficient for the complex electric 
    field amplitude. Inputs:
        
    n: complex refractive index
    AOI: angle of incidence (radians)
    pol: polarization ('s' or 'p')"""
    
    cos_theta = np.cos(AOI)    
    cos_thetat = 1/n*np.sqrt(n**2 - np.sin(AOI)**2)
    if pol=='p':
        return 2*cos_theta/(cos_thetat + n*cos_theta)
    elif pol=='s':
        return 2*cos_theta/(cos_theta + n*cos_thetat)
    else:
        raise ValueError
        
def reflection(n,AOI,pol):
    """Returns the Fresnel reflection coefficient for the complex electric 
    field amplitude. Inputs:
        
    n: complex refractive index
    AOI: angle of incidence (radians)
    pol: polarization ('s' or 'p')"""
    
    cos_theta = np.cos(AOI)
    cos_thetat = 1/n*np.sqrt(n**2 - np.sin(AOI)**2)
    if pol=='p':
        return (n*cos_theta-cos_thetat)/(cos_thetat + n*cos_theta)
    elif pol=='s':
        return (cos_theta-n*cos_thetat)/(cos_theta + n*cos_thetat)
    else:
        raise ValueError

