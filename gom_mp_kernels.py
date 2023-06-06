'''
@ David Pojunas

Kernels defining Particles in the Gulf of Mexico
'''


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
        #particle.beached = 2

def StokesUV(particle, fieldset, time):
    if particle.beached == 0:
        (u_uss, v_uss) = fieldset.UVst[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_uss * particle.dt
        particle.lat += v_uss * particle.dt
        #particle.beached = 3

def SmagDiffBeached(particle, fieldset, time):
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
        
def SetDisplacementB(particle, fieldset, time):
    if particle.beached == 0:
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

def c(particle, fieldset, time):    
    if particle.d2s < 0.5 and particle.beached == 0:
        dtt = -1*particle.dt
        particle.lon += particle.dU*dtt
        particle.lat += particle.dV*dtt
        
def BeachTesting(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if fabs(u) < 1e-14 and fabs(v) < 1e-14:
            particle.beached = 1
            particle.d2s = fieldset.distance2shore[time, particle.depth,
                                particle.lat, particle.lon]
            if particle.d2s < 0.5:
                dtt = 1*particle.dt
                particle.lon += particle.dU*dtt
                particle.lat += particle.dV*dtt   
        else:
            particle.beached = 0     

# THIS IS NOT WORKING...            
def UnBeaching(particle, fieldset, time):
    if particle.beached == 4:
        print("Particle [%d] UnBeached !! (%g %g %g %g)" % (particle.id, particle.lon, particle.lat, particle.depth, particle.time))
        dtt = -1*particle.dt
        (ub, vb) = fieldset.UVunbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += ub * dtt
        particle.lat += vb * dtt
        particle.beached = 0.0
        particle.unbeachCount += 1.0
        
def OutOfBounds(particle, fieldset, time):
    particle.lon = 0.0
    particle.lat = 0.0
    particle.beached = 1.0
    
def Ageing2(particle, fieldset, time):
    particle.age += particle.dt


def SmagDiff2(particle, fieldset, time):
    if particle.beached == 0.0:
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

        particle.beached = 3.0

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