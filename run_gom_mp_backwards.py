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
    
    fieldset.add_field(stokes_fieldset.Ust)
    fieldset.add_field(stokes_fieldset.Vst)
    
    vectorfield_stokes = VectorField('UVst', fieldset.Ust, fieldset.Vst)
    fieldset.add_vector_field(vectorfield_stokes)
    
def set_unbeach_field(fieldset):
    file = 'data/gom_masks_w_inputs.nc'
    variables = {'unBeachU': 'disp_vx',
                 'unBeachV': 'disp_vy'}
    dimensions = {'lat': 'Latitude', 'lon': 'Longitude'}
    fieldsetUnBeach = FieldSet.from_netcdf(file, variables, dimensions, allow_time_extrapolation = True)
    fieldset.add_field(fieldsetUnBeach.unBeachU)
    fieldset.add_field(fieldsetUnBeach.unBeachV)
    
    UVunbeach = VectorField('UVunbeach', fieldset.unBeachU, fieldset.unBeachV)
    fieldset.add_vector_field(UVunbeach)
    

def set_smagdiff_fieldset(fieldset, diff = 0.1):
    fieldset.add_field(Field(name='cell_areas', data=fieldset.U.cell_areas(), lon=fieldset.U.grid.lon, lat=fieldset.U.grid.lat))
    fieldset.add_constant('Cs', diff)


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
        # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach
        beached = Variable('beached', initial = 0.0)
        unbeachCount = Variable('unbeachCount', dtype=np.int32, initial=0.0)
    
    # load nurdle dataset
    df_all_nurdles = pd.read_csv('data/nurdle_release.csv')
    
    # compute starting locations
    vals = df_all_nurdles[['lat', 'lon', 'time']].values
    n = df_all_nurdles.nurdle_count.values
    lats, lons, time = monte_carlo_multi_pr(vals, n)
    
    return ParticleSet(fieldset = fieldset, pclass = Nurdle, lon = lons, lat = lats, time = time,)


def run_gom_mp_backwards(outfile, stokes = True, diff = 0.1, fw = -1):
    #gom_masks = xr.open_dataset('data/gom_masks_w_inputs.nc')
    
    # SET FIELDSETS
    fieldset = get_hycom_fieldset()
    
    if stokes:
        set_stokes_fieldset(fieldset)
    if diff > 0.0:
        set_smagdiff_fieldset(fieldset, 0.1)
        
    set_unbeach_field(fieldset)
            
    # SET PARTICLESETS
    pset = get_particle_set(fieldset)
    
    kernels = pset.Kernel(AdvectionRK4) 
    kernels += pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching)
    if stokes:
        kernels += pset.Kernel(StokesUV) + pset.Kernel(BeachTesting)
    if diff:
        kernels += pset.Kernel(SmagDiff2) + pset.Kernel(BeachTesting)
    kernels += pset.Kernel(Ageing2)    
    
    pfile = pset.ParticleFile(name=outfile, outputdt=timedelta(hours=24))
    pset.execute(kernels, runtime=timedelta(days=60), dt=fw*timedelta(hours=2), output_file=pfile, recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle})
    pfile.close()

    
    return fieldset, pset

