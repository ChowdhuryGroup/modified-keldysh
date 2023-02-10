# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:48:27 2021

@author: tripepi.4
"""
import numpy as np
import labclerk as lck
import Material_Properties as mp
import orchard as orc

def tableprep(inten_array,data_arrays,headers,delim='\t'):
    """
    Takes array output from simulation and prepares it to be put in a table for export.
    'headers' is a list for data arrays. Intensity array is already labeled.
    """
    inten=np.squeeze(inten_array)*1e-13 #intensity in units of GW/cm^2
    to_vstack=[inten]
    for d in data_arrays:
        to_vstack.append(d*1e-6) #nCB in units of cm^-3
    final=np.vstack(to_vstack)
    fn=lck.savedlg()
    head='Intensity (GW/cm^2)'
    for h in headers:
        head=head+delim+h
    np.savetxt(fn,np.transpose(final),delimiter=delim,header=head)
    return final,fn

def thresholds(Wavelength,MatProp,MatPropDict,delim='\t'):
    """
    Calculate all the thresholds at once.
    """
    
    mpd=MatPropDict
    wl=np.squeeze(Wavelength,axis=2)
    ncr=mp.ncrit(wl)*1e-6 #units: cm^-3
    mncr=mp.ncritmod(wl,mpd['CB effective mass'][0],np.squeeze(mpd['Refractive index'],axis=2)**2)*1e-6 #units: cm^-3
    melt=np.ones(wl.shape)*mp.MeltThreshold(MatProp.loc[0]['NumBonds'],MatProp.loc[0]['LatticeConstant']*10**-10,10)*1e-6 #units: cm^-3
    head='Wavelength (nm)'
    headers=['ncrit (cm^-3)','ncritmod (cm^-3)','Melt10% (cm^-3)']
    for h in headers:
        head=head+delim+h
    fn=lck.savedlg()
    final=np.hstack((wl*1e9,ncr,mncr,melt))
    np.savetxt(fn,final,delimiter=delim,header=head)
    return final,fn

def ncbtotplot():
    
    return