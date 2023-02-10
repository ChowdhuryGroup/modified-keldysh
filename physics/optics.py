import math
import cmath
import numpy as np


def Reflectivity(n,theta,polarization):
    """Returns the reflectivity given the complex refractive index n, the
    angle of incidence theta, and the polarization"""

    if polarization=='p':
        return abs((cmath.sqrt(1-(1/n*math.sin(theta))**2)-n*math.cos(theta))/\
        (cmath.sqrt(1-(1/n*math.sin(theta))**2)+n*math.cos(theta)))**2

    elif polarization=='s':
        return abs((math.cos(theta)-n*cmath.sqrt(1-(1/n*math.sin(theta))**2))/\
        (math.cos(theta)+n*cmath.sqrt(1-(1/n*math.sin(theta))**2)))**2

    else: #unpolarized
        Rs = abs((math.cos(theta)-n*cmath.sqrt(1-(1/n*math.sin(theta))**2))/\
        (math.cos(theta)+n*cmath.sqrt(1-(1/n*math.sin(theta))**2)))**2

        Rp = abs((cmath.sqrt(1-(1/n*math.sin(theta))**2)-n*math.cos(theta))/\
        (cmath.sqrt(1-(1/n*math.sin(theta))**2)+n*math.cos(theta)))**2

        return (Rs+Rp)/2

def BrewsterAngle(n):
    """Returns the Brewster angle given the refractive index n"""
    return math.tan(n.real)*180/(math.pi)

def Sellmeier(wavelength,SellmeierCoefficients,form=1):
    """Calculates the refractive index of a material given the wavelength in
    microns and the Sellmeier coefficients. The input 'form' refers to the two
    common forms of the Sellmeier equation:

    n^2 = 1 + B1*wavelength**2/(wavelength**2-C1) +
            B2*wavelength**2/(wavelength**2-C2) +
            B3*wavelength**2/(wavelength**2-C3)     (form 1)

    and

    n^2 = A + B1*wavelength**2/(wavelength**2-C1) +
            B2*wavelength**2/(wavelength**2-C2)     (form 2)"""

    if form==1:
        B1 = SellmeierCoefficients[0]
        C1 = SellmeierCoefficients[1]
        B2 = SellmeierCoefficients[2]
        C2 = SellmeierCoefficients[3]
        B3 = SellmeierCoefficients[4]
        C3 = SellmeierCoefficients[5]

        nsquared = ( 1 + B1*wavelength**2/(wavelength**2-C1) +
                    B2*wavelength**2/(wavelength**2-C2) +
                    B3*wavelength**2/(wavelength**2-C3) )

    elif form==2:
        A = SellmeierCoefficients[0]
        B1 = SellmeierCoefficients[1]
        C1 = SellmeierCoefficients[2]
        B2 = SellmeierCoefficients[3]
        C2 = SellmeierCoefficients[4]

        nsquared = ( A + B1*wavelength**2/(wavelength**2-C1) +
                    B2*wavelength**2/(wavelength**2-C2) )

    elif form==3:
        A = SellmeierCoefficients[0]
        B = SellmeierCoefficients[1]
        C = SellmeierCoefficients[2]
        D = SellmeierCoefficients[3]
        E = SellmeierCoefficients[4]
        
        L = 1/(wavelength**2 - 0.028)

        nsquared = (A + B*L + C*L**2 + D*wavelength**2 + E*wavelength**4)**2
        
    elif form==4:
        A = SellmeierCoefficients[0]
        B1 = SellmeierCoefficients[1]
        C1 = SellmeierCoefficients[2]
        B2 = SellmeierCoefficients[3]
        C2 = SellmeierCoefficients[4]

        nsquared = ( A + B1/(wavelength**2-C1) +B2/(wavelength**2-C2) )
        
    elif form==5:
        A = SellmeierCoefficients[0]
        B = SellmeierCoefficients[1]
        C = SellmeierCoefficients[2]
        D = SellmeierCoefficients[3]

        nsquared = A+B/(wavelength**2-C)-D*wavelength**2
    else:
        raise ValueError

    n = np.sqrt(nsquared)

    return n
