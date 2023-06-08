'''
@David Pojunas

Runs backwards simulations
'''

import xarray as xr
import numpy as np
import pandas as pd
import math 

from gom_mp_kernels import *

from parcels import ParcelsRandom
from parcels import (FieldSet, Field, ParticleSet, JITParticle, ErrorCode, Variable ,GeographicPolar,Geographic, VectorField)


from glob import glob
from datetime import timedelta 

### Global Constants
MIN_LON = -98.0
MAX_LON = -76.400024
MIN_LAT = 18.12
MAX_LAT = 31.92

#cell size
CELL_SIZE = 1/25

#earth radius in km 
EARTH_RADIUS_KM = 6371

#testing domain
tc_lat, tc_lon = 29.4, -89.400024
ti_latmin, ti_latmax, ti_lonmin, ti_lonmax =  272, 292, 205, 225
lon_min, lon_max, lat_min, lat_max = -90.2, -88.64, 28.6, 30.16

######################
### SET FIELDSETS  ###
######################

def get_hycom_fieldset(indices = {}):
    data_dir = 'data/'
    hycom_files = sorted(glob(data_dir + '*_GOM_HYCOM_raw.nc'))
    filenames = {'U': hycom_files, 'V': hycom_files}
    variables = {'U': 'water_u', 'V': 'water_v'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    return FieldSet.from_netcdf(filenames, variables, dimensions, indices = indices)

def get_stokes(indices = {}):
    data_dir = 'data/'
    stokes_files = sorted(glob(data_dir + '*_GOM_STOKES_raw.nc'))
    filenames = {'Ust': stokes_files, 'Vst': stokes_files}
    variables = {'Ust': 'VSDX', 'Vst': 'VSDY'}
    dimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    interp_method = {'Ust' : 'linear','Vst' : 'linear'}
    return FieldSet.from_netcdf(filenames, variables, dimensions, indices = indices, allow_time_extrapolation = True)
    
def set_stokes_sumfieldset(fieldset):
    stokes_fieldset = get_stokes()
    stokes_fieldset.Ust.units = GeographicPolar()
    stokes_fieldset.Vst.units = Geographic()
    fieldset = FieldSet(U=fieldset.U + stokes_fieldset.Ust,
                        V=fieldset.V + stokes_fieldset.Vst)
    return fieldset

def set_stokes_fieldset(fieldset):
    stokes_fieldset = get_stokes()
    stokes_fieldset.Ust.units = GeographicPolar()
    stokes_fieldset.Vst.units = Geographic()
    
    fieldset.add_field(stokes_fieldset.Ust)
    fieldset.add_field(stokes_fieldset.Vst)
    
    vectorfield_stokes = VectorField('UVst', fieldset.Ust, fieldset.Vst)
    fieldset.add_vector_field(vectorfield_stokes)
    return fieldset

def set_displacement_field(fieldset):
    # ADD BOUNDRY PROPERTIES
    fieldset.interp_method = {'U': 'freeslip', 'V': 'freeslip'}
    
    # ADD DISPLACEMENT FIELD PROPERTIES
    gom_masks = xr.open_dataset('data/gom_masks_w_inputs.nc')
    u_displacement = gom_masks.disp_vx.values
    v_displacement = gom_masks.disp_vy.values
    d2s = gom_masks.d2s.values

    fieldset.add_field(Field('dispU', data=u_displacement,
                            lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                            mesh='spherical', allow_time_extrapolation = True))
    fieldset.add_field(Field('dispV', data=v_displacement,
                            lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat,
                            mesh='spherical', allow_time_extrapolation = True))
    fieldset.dispU.units = GeographicPolar()
    fieldset.dispV.units = Geographic()
    
    fieldset.add_field(Field('distance2shore', d2s,
                            lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat, 
                            mesh='spherical', allow_time_extrapolation = True))
    return fieldset

def set_smagdiff_fieldset(fieldset, diff = 0.1):
    fieldset.add_field(Field(name='cell_areas', data=fieldset.U.cell_areas(), lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat))
    fieldset.add_constant('Cs', diff)
    return fieldset 

#########################
### LOAD PARTICLE SET ###
#########################

def monte_carlo_multi_pr(vals, n, size = CELL_SIZE, seed = 1001):
    #np.random.seed(seed)
    assert(len(vals) == len(n)) # sanity check
    release_locs = np.repeat(vals, n, axis = 0)
    lls = release_locs[:, :2]
    r = size * np.sqrt(np.random.rand(np.shape(lls)[0], np.shape(lls)[1]))
    theta = 2 * math.pi * (np.random.rand(np.shape(lls)[0]))
    theta = np.array([np.sin(theta), np.cos(theta)]).T
    lls = (lls + theta * r).T
    # RETURNS LATS, LONS, TIME
    return lls[0], lls[1], release_locs.T[2]


def get_particle_set(fieldset, release_time_days):
    class Nurdle(JITParticle):
        dU = Variable('dU')
        dV = Variable('dV')
        d2s = Variable('d2s', initial=1e3)
        age = Variable('age', initial=0.0)
        # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach
        beached = Variable('beached', initial = 0.0)
        unbeachCount = Variable('unbeachCount', dtype=np.int32, initial=0.0)
    
    # load nurdle dataset
    df_all_nurdles = pd.read_csv('data/nurdle_release.csv')
    # limit the particles released given 
    release_time_seconds = release_time_days * 24 * 60 * 60
    df_all_nurdles = df_all_nurdles[df_all_nurdles['time'] >= release_time_seconds]
    # compute starting locations
    vals = df_all_nurdles[['lat', 'lon', 'time']].values
    n = df_all_nurdles.nurdle_count.values
    lats, lons, time = monte_carlo_multi_pr(vals, n)
    
    return ParticleSet(fieldset = fieldset, pclass = Nurdle, lon = lons, lat = lats, time = time,)

###############
### KERNELS ###
###############
def AdvectionRK4(particle, fieldset, time):
    if particle.beached == 0:
        (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)

        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)

        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)

        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
        particle.beached = 2
        
def StokesUV(particle, fieldset, time):
    if particle.beached == 0:
        (u_uss, v_uss) = fieldset.UVst[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_uss * particle.dt
        particle.lat += v_uss * particle.dt
        particle.beached = 3

def SmagDiff(particle, fieldset, time):
    if particle.beached == 0:
        dx = 0.01
        # gradients are computed by using a local central difference.
        updx, vpdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon+dx]
        umdx, vmdx = fieldset.UV[time, particle.depth, particle.lat, particle.lon-dx]
        updy, vpdy = fieldset.UV[time, particle.depth, particle.lat+dx, particle.lon]
        umdy, vmdy = fieldset.UV[time, particle.depth, particle.lat-dx, particle.lon]

        dudx = (updx - umdx) / (2*dx)
        dudy = (updy - umdy) / (2*dx)
        
        dvdx = (vpdx - vmdx) / (2*dx)
        dvdy = (vpdy - vmdy) / (2*dx)

        A = fieldset.cell_areas[time, 0, particle.lat, particle.lon]
        sq_deg_to_sq_m = (1852*60)**2*math.cos(particle.lat*math.pi/180)
        A = A / sq_deg_to_sq_m
        Kh = fieldset.Cs * A * math.sqrt(dudx**2 + 0.5*(dudy + dvdx)**2 + dvdy**2)

        dlat = ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Kh) 
        dlon = ParcelsRandom.normalvariate(0., 1.) * math.sqrt(2*math.fabs(particle.dt)* Kh) 

        particle.lat += dlat
        particle.lon += dlon
        
        particle.beached = 3

def BeachTest(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-14 and fabs(v) < 1e-14:
            if particle.beached == 2:
                particle.beached = 4
            else:
                dispUab, dispVab = fieldset.dispU[time, particle.depth, particle.lat,particle.lon], fieldset.dispV[time, particle.depth, particle.lat,particle.lon]
                dtt = -1*particle.dt
                particle.lon += dispUab*dtt
                particle.lat += dispVab*dtt
                particle.beached = 1  
        else:
            particle.beached = 0

def Unbeach(particle, fieldset, time):    
    if particle.beached == 4:
        dispUab, dispVab = fieldset.dispU[time, particle.depth, particle.lat,particle.lon], fieldset.dispV[time, particle.depth, particle.lat,particle.lon]
        dtt = -1*particle.dt
        particle.lon += dispUab*dtt
        particle.lat += dispVab*dtt
        particle.beached = 0

def Ageing(particle, fieldset, time):
    particle.age += particle.dt

def DeleteParticle(particle, fieldset, time):
    particle.delete()
    
######################
### RUN SIMULATION ###
######################

def run_gom_mp_backwards(outfile, total_sim_time_days = 30, sim_dt_hours = 2, output_dt_hours = 24, fw = -1):
    # FIELD SET
    fieldset = get_hycom_fieldset()
    fieldset = set_stokes_fieldset(fieldset)
    fieldset = set_displacement_field(fieldset)
    fieldset = set_smagdiff_fieldset(fieldset)
    
    # PARTICLE SET
    release_time_days = total_sim_time_days / 2
    pset = get_particle_set(fieldset, release_time_days)
    
    # KERNELS
    kernels = (pset.Kernel(AdvectionRK4) + pset.Kernel(BeachTesting) + pset.Kernel(DisplaceB) + pset.Kernel(StokesUV) + pset.Kernel(BeachTesting) +
               pset.Kernel(SmagDiffBeached) + pset.Kernel(Ageing) + pset.Kernel(BeachTesting))
    
    pfile = pset.ParticleFile(name=outfile, outputdt=timedelta(hours=output_dt_hours))
    pset.execute(kernels, runtime=timedelta(days=total_sim_time_days), dt=fw*timedelta(hours=sim_dt_hours), output_file=pfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
    pfile.close()
    return fieldset, pset