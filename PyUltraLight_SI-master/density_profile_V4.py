# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:19:01 2023

author: Emily Kendall (main algorithm for converting the data into a radial density profile)
Modified and upgraded by Matthias Stallovits (includes units, fitting, mass calculation, plotting and inclusion in the main package)
"""

import numpy as np
from scipy.optimize import curve_fit

### Define constants
hbar = 1.0545718e-34  # m^2 kg/s
parsec = 3.0857e16  # m
solar_mass = 1.989e30  # kg
axion_mass = 1e-22 * 1.783e-36  # kg
G = 6.67e-11  # N m^2 kg^-2
omega_m0 = 0.31
H_0 = 67.7 * 1e3 / (parsec * 1e6)  # s^-1

##units
length_unit = (8 * np.pi * hbar ** 2 / (3 * axion_mass ** 2 * H_0 ** 2 * omega_m0)) ** 0.25 # length code unit
mass_unit = (3 * H_0 ** 2 * omega_m0 / (8 * np.pi)) ** 0.25 * hbar ** 1.5 / (axion_mass ** 1.5 * G) # mass code unit

kpc=length_unit/(parsec*10**3) # for conversion: length code unit into kpc
sm=mass_unit/solar_mass # for conversion: mass code unit into solar masses
rho_unit=sm*(kpc)**(-3) # for the conversion of density code unit into solar masses per kpc^3

### Functions
## function to convert radius and density from code units to physical units
def conversion(quantity,list_quantity):
     if quantity == "radius":
         radius_unit=kpc
         return [i*radius_unit for i in list_quantity]
     elif quantity == "density":
         density_unit=rho_unit
         return [j*density_unit for j in list_quantity]

## function for the point system to decide which core-fit is better    
def point_sys(argument1, argument2, Counter_list): # argument1 for Gauß, argument2 for Schive
    Counter1=Counter_list[0] # Counter for Gauß later
    Counter2=Counter_list[1] # Counter for Schive later
    if argument1>argument2:
        Counter2=Counter_list[1]+1
    else:
        Counter1=Counter_list[0]+1
    return [Counter1,Counter2]

## function to split the radius list for the two differnt fits (one for core, one for envelope)
def split_fit(index,list_quantity):
    core=[]
    envelope=[]
    for i in range(0,index+1):
            core.append(list_quantity[i])
    for j in range(index,len(list_quantity)):
            envelope.append(list_quantity[j])
    return [core,envelope]

## function for the Gauß-profile (for the core)
def Gauß(r,a,sig):
    return a*np.exp(-np.power(r,2)/(2*sig**2))

## function for the Schive-profile (for the core)
def Schive(r,delta_s,r_sol):
    return delta_s/np.power(1+np.power((r/r_sol),2),8)

## function for the NFW-profile (for the envelope)
def NFW(r, delta_NFW, r_s):
    return delta_NFW/((r/r_s)*np.power(1+(r/r_s),2))

## function for the fitting routine
def fitting(radius,density, solitons):
    # calculation of the core-fit-parameters
    popt1,pcov1= curve_fit(Gauß, radius, density,p0=[density[0],radius[0]])
    perr1=np.sqrt(np.diag(pcov1))

    popt2,pcov2 = curve_fit(Schive, radius, density,p0=[density[0],radius[0]])
    perr2=np.sqrt(np.diag(pcov2))
    
    # set everything up for the point system
    res_g=[]
    res_s=[]
    for i in range(0,len(radius)):
        res_g.append(abs(density[i]-Gauß(radius[i], *popt1)))
        res_s.append(abs(density[i]-Schive(radius[i], *popt2)))
        res_g_max=max(res_g)
        res_s_max=max(res_s) 
    
    Counter=[0,0]

    # apply point system
    Counter=point_sys(res_g_max, res_s_max, Counter)
    Counter=point_sys(perr1[0], perr2[0], Counter)
    Counter=point_sys(perr1[1], perr2[1], Counter)


    # determining the connection point
    factor=[]
    if Counter[0]>Counter[1]:
        Core_Fit="Gauß"
        for i in range(0,len(radius)):
            factor.append(abs(radius[i]-(2.575829*popt1[1])))
    else:
        Core_Fit="Schive"
        for i in range(0,len(radius)):
            factor.append(abs(radius[i]-(2.575829*popt1[1])))
    
    # for one soliton a connection point is not needed
    if len(solitons) == 1:
        fit_radius=[radius,"NoEnvelope"]    # only first entry needed

        return [Core_Fit, fit_radius, popt1, popt2]
    
    # for more than one soliton we need the connection point as a starting point for the NFW-model
    elif len(solitons) > 1:
        #split the radii at the connection point
        connection=factor.index(min(factor))
        fit_radius=split_fit(connection, radius) #0 for core; 1 for envelope
        fit_density=split_fit(connection, density) #0 for core; 1 for envelope

        # calculating the envelope-fit-parameters
        popt3,pcov3= curve_fit(NFW, fit_radius[1], fit_density[1], p0=[fit_density[1][0],fit_radius[1][0]], bounds=([fit_density[1][0]/2,fit_radius[1][0]/2], [2*fit_density[1][0], 2*fit_radius[1][0]]))

        return [Core_Fit, fit_radius, popt1, popt2, popt3]
    else:
        raise NameError("Solitons-list has no entries.")
        
## function to prepare the simulation data for the fitting routine        
def Fitpreperation(loc, state, length, resol): 
        data = np.load('{}{}{}{}'.format(loc, '/rho_#', state, '.npy'))
        gridspace = length/resol
        
        # find halo centres and truncate if displaced
        centre = np.unravel_index(np.argmax(data), data.shape)
        trunc = min(resol-centre[0], centre[0], resol-centre[1], centre[1], centre[2], resol-centre[2])

        # setting up x, y, z vectors to determine radial distance from centre of halo
        xvec = np.array([[np.linspace((0 - centre[0]) * gridspace, (resol - (centre[0]+1)) * gridspace, resol)]]).reshape(resol,1, 1)
        yvec = np.array([[np.linspace((0 - centre[1]) * gridspace, (resol - (centre[1]+1)) * gridspace, resol)]]).reshape(1,resol,1)
        zvec = np.array([[np.linspace((0 - centre[2]) * gridspace, (resol - (centre[2]+1)) * gridspace, resol)]])
        rvals = np.sqrt(xvec ** 2 + yvec ** 2 + zvec ** 2)

        # binning distances and calculation of the spherical averages
        bins = np.zeros(trunc + 1)
        bins[0] = 0
        bins[1] = gridspace/2
        for k in range(2,trunc + 1):
            bins[k] = bins[k-1] + gridspace
        avg = np.zeros(trunc)
        for j in range(int(trunc)):
            avg[j] = np.average(data[np.logical_and(bins[j] <= rvals, rvals < bins[j+1])])
        avg = avg.tolist()
        bins = bins.tolist()
        bins.pop(0)

        # nan-cancelling; useful, when the avg-list contains nans due to the averaging process
        # e = 0
        
        # while e < len(avg):
        #     if math.isnan(avg[e]):
        #         avg.pop(e)
        #         bins.pop(e)
        #         e = e-1
        #     e += 1
        
        # convert units
        radius = conversion("radius", bins)
        density = conversion("density", avg)
     
        return [radius, density]

## function to calculate the total mass and the radius, that contains p*100 % of the mass
def mass_and_R_p_calculation(radius, density, p, solitons, method):
    found_R = False
    total_mass = 0
    
    for soliton in solitons:
        total_mass += soliton[0]
        
    Mass_D = total_mass*sm*solar_mass
    
    M=0
    b=0
    
    if method == 'S':
        # calculating the radius R_p and total mass, using integration of spherical shells
        method_used = "integration with spherical shells"
        
        while b<len(radius):
            if b == 0:
                new_M = density[b]*4*np.pi*(radius[b])**2
            else:
                new_M = density[b]*4*np.pi*(radius[b]**2)*(radius[b]-radius[b-1])
            M=M+new_M
            b += 1
            if M >= p*Mass_D/solar_mass and found_R == False:
                R_p = radius[b]
                found_R = True
    elif method == 'T':
        # calculating the radius R_p and total mass, using the trapez rule
        method_used = "integration with the trapez rule"
        
        while b < len(radius):
            if b == 0:
                new_M = 0
            else:
                new_M = 4*np.pi*(radius[b]-radius[b-1])*(1/2)*((radius[b])**2*density[b]+(radius[b-1])**2*density[b-1])
            M = M+new_M
            b +=1
            if M >= p*Mass_D/solar_mass and found_R == False:
                R_p = radius[b]
                found_R = True
    else:
        raise NameError("Unsupported method used.")

    print("For the mass and R_p calculation the " + method_used + "-method was used.")
    print("The total mass should be " + str(total_mass) + " CE.") 
    print("The calculated total mass is " + str(round(M/sm,3)) + " CE.")
    print("The radius, that contains " + str(p*100) + "% of the mass is " + str(round(R_p,3)) + " kpc.")

    return R_p

# It could be shown that in most cases the trapezoidal rule delivered better
# results, which is why it was chosen as the standard method.