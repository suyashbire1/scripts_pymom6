import eulerian_velocities_loop as evl
import eulerian_transport as et
import comparison as cp
import xarray as xr
import sys
import numpy as np


def save_euvel(expt):
    vels = evl.get_velocities(
        expt.fil0,
        tokm=None,
        z=np.linspace(-3000, 0, 100),
        geometry=expt.geometry)
    ds = xr.Dataset(dict(u=vels['u'], v=vels['v'], w=vels['w']))
    ds.to_netcdf(expt.name + '_eulerian_vels.nc')


def save_eutrans(expt):
    vels = et.get_transport(
        expt.fil0,
        tokm=None,
        z=np.linspace(-3000, 0, 50),
        geometry=expt.geometry)
    ds = xr.Dataset(dict(uh=vels['uh'], vh=vels['vh']))
    ds.to_netcdf(expt.name + '_eulerian_trans.nc')


if __name__ == '__main__':
    print(sys.argv[1])
    expt = getattr(cp, sys.argv[1])
    save_eutrans(expt)
