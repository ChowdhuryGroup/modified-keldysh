#==============================================================================
# Imports
#==============================================================================
import numpy as np
import scipy as sp
from math import pi
from physics.constants import me,c,epsilon0,q
from physics.optics import Sellmeier


#%%
#==============================================================================
# Threshold electron densities
#==============================================================================
def ElectronDensity_VB(NumBonds,LatticeSpacing):
    """Calculates the valence band electron density in m^-3 assuming two 
    valence electrons per bond and a cubic lattice.
    
    NumBonds: Number of bonds per unit cell
    LatticeSpacing: Side length of unit cell (m)"""
    
    return 2*NumBonds*LatticeSpacing**-3

def MeltThreshold(NumBonds,LatticeSpacing,ThresholdPercentage=10):
    """Calculates the electron density threshold for ultrafast melting. This is
    ~10% of the valence band electron density, which is found assuming two
    valence electrons per bond and a cubic lattice.
    
    NumBonds: Number of bonds per unit cell
    LatticeSpacing: Side length of unit cell (m)
    ThresholdPercentage: percent ionization to use"""
    
    nVB = ElectronDensity_VB(NumBonds,LatticeSpacing) #Valence band electron density
    
    return ThresholdPercentage/100 * nVB

def ncrit(Wavelength):
    """Returns the critical density in m^-3 given a wavelength in meters.
    Assumes meff=me and epsilonc=1"""
    omega = 2*pi*c/Wavelength
    return epsilon0*me/q**2*omega**2

def ncritmod(Wavelength, meff, epsilonc):
    """Returns the modified critical density in m^-3. Does not assume meff=me 
    or epsilonc=1
    
    Wavelength: laser wavelength (m)
    meff: effective mass (kg)
    epsilonc = unexcited permittivity"""
    
    omega = 2*pi*c/Wavelength
    return omega**2*meff*epsilon0*np.real(epsilonc)/q**2


#%%
#==============================================================================
# Material properties
#==============================================================================
def MaterialProperties(MatProp,Direction,Wavelength,Multiband=False):
    """Returns a dictionary of material properties given the following:
    
    MatProp: pandas DataFrame of material properties. Many of these properties
    are given in non-base units (e.g. eV, microns, etc) and will be converted
    here.
    
    Direction: crystal direction along which the polarization is oriented
    
    Wavelengths: array of wavelengths
    
    Multiband: boolean determining whether or not multiple bands are to be used
    .
    This dict is to be passed to the function PulseIntegration."""
    
    #Band-dependent properties
    if Multiband:
        mvb = MatProp[(MatProp.CrystalDirection == Direction)].mvb.values*me #Valence band effective masses
        mcb = MatProp[(MatProp.CrystalDirection == Direction)].mcb.values*me #Conduction band effective masses
        mcb_indirect = MatProp[(MatProp.CrystalDirection == Direction)].mcb_indirect.values*me #Conduction band effective masses for indirect transitions
        Bandgap = MatProp[(MatProp.CrystalDirection == Direction)].Bandgap.values*q #Bandgap for each transition
        IndirectBandgap = MatProp[(MatProp.CrystalDirection == Direction)].IndirectBandgap.values*q #Indirect bandgap for each transition
        SpeedofSound = MatProp[(MatProp.CrystalDirection == Direction)].SpeedofSound.values #Speed of sound (m/s)
        ValenceBandNames = MatProp[(MatProp.CrystalDirection == Direction)].ValenceBand.values #VB names
        ValenceBandShape = MatProp[(MatProp.CrystalDirection == Direction)].ValenceBandShape.values #Valence band shapes
    else: #This is gross, find a way to make it prettier
        mvb = MatProp[(MatProp.CrystalDirection == Direction) & 
                                   (MatProp.ValenceBand == 'HeavyHole')].mvb.values*me #Valence band effective masses
        mcb = MatProp[(MatProp.CrystalDirection == Direction) & 
                                   (MatProp.ValenceBand == 'HeavyHole')].mcb.values*me #Conduction band effective masses
        mcb_indirect = MatProp[(MatProp.CrystalDirection == Direction) & 
                                   (MatProp.ValenceBand == 'HeavyHole')].mcb_indirect.values*me #Conduction band effective masses
        Bandgap = MatProp[(MatProp.CrystalDirection == Direction) & 
                                   (MatProp.ValenceBand == 'HeavyHole')].Bandgap.values*q #Bandgap for each transition
        IndirectBandgap = MatProp[(MatProp.CrystalDirection == Direction) & 
                                  (MatProp.ValenceBand == 'HeavyHole')].IndirectBandgap.values*q #Indirect bandgap for each transition
        SpeedofSound = MatProp[(MatProp.CrystalDirection == Direction) & 
                                  (MatProp.ValenceBand == 'HeavyHole')].SpeedofSound.values #Speed of sound (m/s)
        ValenceBandNames = MatProp[(MatProp.CrystalDirection == Direction) & 
                                   (MatProp.ValenceBand == 'HeavyHole')].ValenceBand.values #Bandgap for each transition
        ValenceBandShape = MatProp[(MatProp.CrystalDirection == Direction) & 
                                   (MatProp.ValenceBand == 'HeavyHole')].ValenceBandShape.values #Valence band shapes
    
    
    #Constant collision time for Drude model
    CollisionTime = MatProp.loc[0]['CollisionTime']*10**-15 #Electron-phonon collision time (s)
    
    
    #Unexcited valence band electron density
    NumBonds = MatProp.loc[0]['NumBonds'] #Number of bonds per unit cell
    LatticeConstant = MatProp.loc[0]['LatticeConstant']*10**-10 #Lattice constant (m)
    Density = MatProp.loc[0]['Density'] #Density (kg/m^3)
    MolarMass = MatProp.loc[0]['MolarMass (g/mol)']*1e-3 #units: kg/mol
    NA = 6.02214076e23 #Avogadro's number
    #n0 = NA*Density/MolarMass
    n0 = ElectronDensity_VB(NumBonds,LatticeConstant)
    
    
    
    #Refractive index
    MinSellmeierWavelength = MatProp.loc[0]['MinWavelength']*10**-6 #Min wavelength where the Sellmeier equation can be used
    SellmeierForm = MatProp.loc[0]['SellmeierForm'] #Form of the Sellmeier equation to use
    if SellmeierForm==1:
        SellmeierCoeff = MatProp.loc[0]['B1':'C3'].values.astype('float64') #Sellmeier coefficients
    elif SellmeierForm==2:
        SellmeierCoeff = MatProp.loc[0]['A':'C2'].values.astype('float64') #Sellmeier coefficients
    elif SellmeierForm==3:
        SellmeierCoeff = MatProp.loc[0]['A':'E'].values.astype('float64') #Sellmeier coefficients
    
    elif SellmeierForm==4:
        SellmeierCoeff = MatProp.loc[0]['A':'C2'].values.astype('float64') #Sellmeier coefficients
    
    elif SellmeierForm==5:
        SellmeierCoeff = MatProp.loc[0]['A':'D'].values.astype('float64') #Sellmeier coefficients
    
    n = np.zeros(Wavelength.shape) #Real part of refractive index
    n[Wavelength>=MinSellmeierWavelength] = Sellmeier(Wavelength*10**6, SellmeierCoeff,
                                            SellmeierForm)[Wavelength>=MinSellmeierWavelength]
    n[Wavelength<MinSellmeierWavelength] = sp.interp(Wavelength[Wavelength<MinSellmeierWavelength],
     MatProp['Wavelength'].values*10**-9,MatProp['n'].values)
    kappa = np.zeros(Wavelength.shape) #Imaginary part of refractive index
    kappa[Wavelength>=MinSellmeierWavelength] = 0
    kappa[Wavelength<MinSellmeierWavelength] = sp.interp(Wavelength[Wavelength<MinSellmeierWavelength],
         MatProp['Wavelength'].values*10**-9,MatProp['kappa'].values)
    ncomplex = n + kappa*1.0j

    
    #Properties for Vinogradov absorption
    DeformationPotential = MatProp.loc[0]['DeformationPotential']*q #Deformation potential (J)

    PhononFrequency = MatProp.loc[0]['PhononFrequency']*10**12 #Polar optical phonon frequency (Hz)
    Tphonon = MatProp.loc[0]['Tphonon'] #Phonon temperature (K)
    epsilonInf = MatProp.loc[0]['epsilonInf'] #Unexcited permittivity (high-frequency limit)
    epsilonStatic = MatProp.loc[0]['epsilonStatic'] #Unexcited permittivity (low-frequency limit)
    
    
    return {'VB effective mass':mvb, 'CB effective mass':mcb, 
            'Indirect CB effective mass':mcb_indirect,
            'CollisionTime':CollisionTime,'Bandgap':Bandgap, 
            'Indirect bandgap':IndirectBandgap,'VB name':ValenceBandNames,
            'VB shape':ValenceBandShape,'Total electron density':n0,
            'Refractive index':ncomplex,
            'Deformation potential':DeformationPotential,
            'Density':Density,'Speed of sound':SpeedofSound,
            'Phonon frequency':PhononFrequency,'Phonon temperature':Tphonon,
            'High frequency permittivity':epsilonInf,
            'Static permittivity':epsilonStatic}
