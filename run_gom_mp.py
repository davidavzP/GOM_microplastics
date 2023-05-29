'''
@David Pojunas

Contains main simulation loop
'''

import xarray as xr
import numpy as np
import pandas as pd
import math 

from gom_mp_kernels import *

from parcels import ParcelsRandom
from parcels import (FieldSet, Field, ParticleSet, JITParticle, AdvectionRK4, ErrorCode,
                     DiffusionUniformKh, AdvectionDiffusionM1, AdvectionDiffusionEM, Variable ,GeographicPolar,Geographic)


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
t_latmin, t_latmax, t_lonmin, t_lonmax = 29.0, 29.76,-89.79999, -89.03998


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
    
def set_stokes_fieldset(fieldset):
    stokes_fieldset = get_stokes()
    stokes_fieldset.Ust.units = GeographicPolar()
    stokes_fieldset.Vst.units = Geographic()
    fieldset=FieldSet(U=fieldset.U + stokes_fieldset.Ust,
                        V=fieldset.V + stokes_fieldset.Vst)
    return fieldset

def set_displacement_field(fieldset, base_fieldsetU, masks, indices = {}):
    if indices:
        masks = masks.isel(Latitude = indices['lat'], Longitude = indices['lon'])
        
    # EXTRACT DATA
    u_displacement = masks.disp_vx
    v_displacement = masks.disp_vy
    landmask = masks.landmask
    d2s = masks.d2s
    
    # ADD BOUNDRY PROPERTIES
    fieldset.interp_method = {'U': 'freeslip', 'V': 'freeslip'}
    
    # ADD DISPLACEMENT FIELD
    fieldset.add_field(Field('dispU', data=u_displacement,
                            lon=base_fieldsetU.grid.lon, lat=base_fieldsetU.grid.lat,
                            mesh='spherical', allow_time_extrapolation = True))
    fieldset.add_field(Field('dispV', data=v_displacement,
                            lon=base_fieldsetU.grid.lon, lat=base_fieldsetU.grid.lat,
                            mesh='spherical', allow_time_extrapolation = True))
    fieldset.dispU.units = GeographicPolar()
    fieldset.dispV.units = Geographic()

    # ADD FIELD PROPERTIES
    fieldset.add_field(Field('landmask', landmask,
                            lon=base_fieldsetU.grid.lon, lat=base_fieldsetU.grid.lat,
                            mesh='spherical', allow_time_extrapolation = True))
    fieldset.add_field(Field('distance2shore', d2s,
                            lon=base_fieldsetU.grid.lon, lat=base_fieldsetU.grid.lat, 
                            mesh='spherical', allow_time_extrapolation = True))
    
    return fieldset

def set_smagdiff_fieldset(fieldset, base_fieldsetU, diff = 0.1):
    fieldset.add_field(Field(name='cell_areas', data=base_fieldsetU.cell_areas(), lon=base_fieldsetU.grid.lon, lat=base_fieldsetU.grid.lat))
    fieldset.add_constant('Cs', diff)
    return fieldset

def set_coast_fieldset(fieldset, base_fieldsetU, masks, indices = {}):
    if indices:
        masks = masks.isel(Latitude = indices['lat'], Longitude = indices['lon'])
    coast = masks.coastal_id_mask
    fieldset.add_field(Field('coast', coast,
                            lon=base_fieldsetU.grid.lon, lat=base_fieldsetU.grid.lat,
                            mesh='spherical', allow_time_extrapolation =True, interp_method='nearest'))
    return fieldset

def monte_carlo_particle_release(clat, clon, total_num_particles, size = CELL_SIZE, seed = 1001):
    np.random.seed(seed)
    r = size * np.sqrt(np.random.rand(total_num_particles))
    theta = np.random.rand(total_num_particles)* 2 * math.pi
    return clat + r * np.sin(theta), clon + r * np.cos(theta)

def monte_carlo_multi_pr(vals, n, size = CELL_SIZE, seed = 1001):
    #np.random.seed(seed)
    assert(len(vals) == len(n)) # sanity check
    release_locs = np.repeat(vals, n, axis = 0)
    lls = release_locs[:, :2]
    r = size * np.sqrt(np.random.rand(np.shape(lls)[0], np.shape(lls)[1]))
    theta = 2 * math.pi * (np.random.rand(np.shape(lls)[0]))
    theta = np.array([np.sin(theta), np.cos(theta)]).T
    lls = (lls + theta * r).T
    # RETURNS LATS, LONS, IDS
    return lls[0], lls[1], release_locs.T[2]

def test_2_pset(gom_masks):
    # Define a release region
    lon_min, lon_max, lat_min, lat_max = -90.2, -88.64, 28.6, 30.16

    # EXTRACT DATA
    river_input_mon = gom_masks.sel(Longitude = slice(lon_min, lon_max), Latitude = slice(lat_min, lat_max))
    rim_idx = np.where(river_input_mon.river_input_mon > 0.0)
    rim_lats = river_input_mon.Latitude[rim_idx[0]].values
    rim_lons =  river_input_mon.Longitude[rim_idx[1]].values
    rim_vals = river_input_mon.river_input_mon.values[rim_idx]
    rim_ids = river_input_mon.coastal_id_mask.values[rim_idx]

    # PREPROCESS FOR MCR
    vals = np.array([rim_lats, rim_lons, rim_ids]).T
    n = np.array(np.ceil(rim_vals), dtype = int)
    
    return monte_carlo_multi_pr(vals, n)



def get_particle_set(fieldset, testing, gom_masks, empty = False):
    class PlasticParticle(JITParticle):
        dU = Variable('dU')
        dV = Variable('dV')
        d2s = Variable('d2s', initial=1e3)
        age = Variable('age', initial=0.0)
        coast = Variable('coast', initial=fieldset.coast) 
        
    if empty:
        return ParticleSet(fieldset = fieldset, pclass = PlasticParticle, lon = [], lat = [], coast = [])
        
    if testing == 1:
        lats, lons = monte_carlo_particle_release(tc_lat, tc_lon, 100)
        return ParticleSet(fieldset = fieldset, pclass = PlasticParticle, lon = lons, lat = lats)
    elif testing == 2:
        lats, lons, ids = test_2_pset(gom_masks)
        return ParticleSet(fieldset = fieldset, pclass = PlasticParticle, repeatdt = timedelta(days = 30), lon = lons, lat = lats, coast = ids)
    else:
        raise ValueError('Not Testing')

# DOES NOT WORK MOMENTARILY
def run_gom_mp(outfile, disp = True, stokes = False, diff = 0.0, indices = {}, testing = 0):
    gom_masks = xr.open_dataset('data/gom_masks_w_inputs.nc')
    
    # SET FIELDSETS
    fieldset = get_hycom_fieldset(indices)
    base_fieldsetU = fieldset.U
    
    if stokes:
        fieldset = set_stokes_fieldset(fieldset)
    if disp:
        fieldset = set_displacement_field(fieldset, base_fieldsetU, gom_masks, indices)
    if diff > 0.0:
        fieldset = set_smagdiff_fieldset(fieldset, base_fieldsetU, diff)
        
    fieldset = set_coast_fieldset(fieldset, base_fieldsetU, gom_masks, indices)
    
    # SET PARTICLESETS
    pset = get_particle_set(fieldset, testing, gom_masks)
    
    kernels = pset.Kernel(SampleCoast) + pset.Kernel(AdvectionRK4)
    if disp:
        kernels = pset.Kernel(Displace) + kernels
    if diff > 0.0:
        kernels += pset.Kernel(SmagDiff)
    if disp:
        kernels += pset.Kernel(SetDisplacement)

    # RUN SIMULATION
    if testing > 0:
        pfile = pset.ParticleFile(name=outfile, outputdt=timedelta(hours=3))
        pset.execute(kernels, runtime=timedelta(hours=500), dt=timedelta(hours=1), output_file=pfile,  recovery={ErrorCode.ErrorOutOfBounds: DeleteParticlePrint})
        pfile.close()
    else:
        raise ValueError('Not Testing')

    return fieldset, pset


def run_gom_mp_repeat_test(outfile, disp = True, stokes = False, diff = 0.0, indices = {}, testing = 0):
    gom_masks = xr.open_dataset('data/gom_masks_w_inputs.nc')
    
    # SET FIELDSETS
    fieldset = get_hycom_fieldset(indices)
    base_fieldsetU = fieldset.U
    
    if stokes:
        fieldset = set_stokes_fieldset(fieldset)
    if disp:
        fieldset = set_displacement_field(fieldset, base_fieldsetU, gom_masks, indices)
    if diff > 0.0:
        fieldset = set_smagdiff_fieldset(fieldset, base_fieldsetU, diff)
        
    fieldset = set_coast_fieldset(fieldset, base_fieldsetU, gom_masks, indices)
    
    # SET PARTICLESETS
    pset = get_particle_set(fieldset, testing, gom_masks, empty = False)
    
    kernels = pset.Kernel(SampleCoast) + pset.Kernel(AdvectionRK4)
    if disp:
        kernels = pset.Kernel(Displace) + kernels
    if diff > 0.0:
        kernels += pset.Kernel(SmagDiff)
    if disp:
        kernels += pset.Kernel(SetDisplacement)
    kernels += pset.Kernel(Ageing)

    # RUN SIMULATION
    if testing > 0:
        pfile = pset.ParticleFile(name=outfile, outputdt=timedelta(hours=3))
        pset.execute(kernels, runtime=timedelta(days=180), dt=timedelta(hours=1), output_file=pfile,  recovery={ErrorCode.ErrorOutOfBounds: DeleteParticlePrint})
        pfile.close()
        # outputdt = timedelta(hours=3).total_seconds() # write the particle data every 3 hour
        # repeatdt = timedelta(days=30).total_seconds() # release each set every month...
        # runtime = timedelta(days=360).total_seconds() 
        
        # output_file = pset.ParticleFile(name=outfile)
        # for time in np.arange(0, runtime, outputdt):
        #     if np.isclose(np.fmod(time, repeatdt), 0):
        #        # NOTE: TIME IS NOT BEING INCLUDED FOR THIS CALL...
        #         pset_init = get_particle_set(fieldset, testing, gom_masks) 
        #         pset_init.execute(SampleCoast, dt=0)    # record the initial temperature of the particles
        #         pset.add(pset_init)                       # add the newly released particles to the total particleset  
        #         print(f'Length of pset at time {time}: {len(pset)}')
            
        #     output_file.write(pset,time)
        #     pset.execute(kernels, runtime=outputdt, dt=timedelta(hours=1))
            
            
        # output_file.write(pset, time+outputdt)
        # output_file.close()

    else:
        raise ValueError('Not Testing')

        
        

    return fieldset, pset