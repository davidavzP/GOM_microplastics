'''
@ David Pojunas

Kernels defining Particles in the Gulf of Mexico
'''

# https://nbviewer.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_diffusion.ipynb
def SmagDiff(particle, fieldset, time):
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

# https://nbviewer.org/github/OceanParcels/parcels/blob/master/parcels/examples/documentation_unstuck_Agrid.ipynb
def SetDisplacement(particle, fieldset, time):
    particle.d2s = fieldset.distance2shore[time, particle.depth,
                               particle.lat, particle.lon]
    if  particle.d2s < 0.5:
        dispUab = fieldset.dispU[time, particle.depth, particle.lat,
                               particle.lon]
        dispVab = fieldset.dispV[time, particle.depth, particle.lat,
                               particle.lon]
        particle.dU = dispUab
        particle.dV = dispVab
    else:
        particle.dU = 0.
        particle.dV = 0.
    
def Displace(particle, fieldset, time):    
    if  particle.d2s < 0.5:
        particle.lon += particle.dU*particle.dt
        particle.lat += particle.dV*particle.dt
        
def Ageing(particle, fieldset, time):
    particle.age += particle.dt
    # We do not want to remove the particle to have more flexibility in the cut off time
    # 15552000 = 180 days
    # 6912000 = 80 days
    if particle.age > 6912000:
        particle.delete()
    
def SampleCoast(particle, fieldset, time):
    particle.coast = fieldset.coast[time, particle.depth, particle.lat, particle.lon]

def DeleteParticlePrint(particle, fieldset, time):
    print("Particle [%d] lost !! (%g %g %g %g)" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
    particle.delete()
    
def DeleteParticle(particle, fieldset, time):
    particle.delete()