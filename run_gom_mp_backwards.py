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


def get_particle_set(fieldset):
    class Nurdle(JITParticle):
        dU = Variable('dU')
        dV = Variable('dV')
        d2s = Variable('d2s', initial=1e3)
        age = Variable('age', initial=0.0)
    
    # load nurdle dataset
    df_all_nurdles = pd.read_csv('data/nurdle_release.csv')
    
    # compute starting locations
    vals = df_all_nurdles[['lat', 'lon', 'time']].values
    n = df_all_nurdles.nurdle_count.values
    lats, lons, time = monte_carlo_multi_pr(vals, n)
    
    return ParticleSet(fieldset = fieldset, pclass = Nurdle, lon = lons, lat = lats, time = time,)


def run_gom_mp_backwards(outfile, disp = True, stokes = False, diff = 0.0, indices = {}, fw = -1):
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
            
    # SET PARTICLESETS
    pset = get_particle_set(fieldset)
    
    kernels = pset.Kernel(AdvectionRK4)
    if disp:
        kernels = pset.Kernel(Displace) + kernels
    if diff > 0.0:
        kernels += pset.Kernel(SmagDiff)
    if disp:
        kernels += pset.Kernel(SetDisplacement)
    
    pfile = pset.ParticleFile(name=outfile, outputdt=timedelta(hours=24))
    pset.execute(kernels, runtime=timedelta(days=365), dt=fw*timedelta(hours=2), output_file=pfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticlePrint})
    pfile.close()
    
    return fieldset, pset

    # # RUN SIMULATION
    # if testing > 0:
    #     pfile = pset.ParticleFile(name=outfile, outputdt=timedelta(hours=3))
    #     pset.execute(kernels, runtime=timedelta(hours=500), dt=timedelta(hours=1), output_file=pfile,  recovery={ErrorCode.ErrorOutOfBounds: DeleteParticlePrint})
    #     pfile.close()
    # else:
    #     raise ValueError('Not Testing')

    # return fieldset, pset

